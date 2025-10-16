#!/usr/bin/env python3

import os
import sys
import io
import math
import json
import base64
import tempfile
import traceback
import librosa
import copy
from datetime import datetime
import asyncio
import threading
import queue
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import Image
import time
import numpy as np
import torchvision.transforms as transforms
from collections import deque
import subprocess
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from peft import LoraConfig, inject_adapter_in_model, PeftModel

from OmniAvatar.utils.args_config import parse_args
from scripts.inference import match_size, resize_pad
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4, load_state_dict
from OmniAvatar.distributed.fsdp import shard_model
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms as TT
from transformers import Wav2Vec2FeatureExtractor
from PIL import Image
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.prompters import WanPrompter
from OmniAvatar.schedulers.flow_match import FlowMatchScheduler
from scripts.causal_inference import CausalInferencePipeline

class PipelinedCausalInferencePipeline(nn.Module):
    """
    流水线化的CausalInferencePipeline实现
    主要改进：
    1. prompt encoding提前到times循环外
    2. causal denoising inference和VAE decoding实现流水线并行
    3. 支持KV cache的因果推理
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        print(f"self.args: {self.args}")
        # set device and dtype
        self.device = torch.device(f"cuda:{args.rank}")
        if args.dtype=='bf16':
            self.dtype = torch.bfloat16
        elif args.dtype=='fp16':
            self.dtype = torch.float16
        else:   
            self.dtype = torch.float32
        
        # load causal inference pipeline
        self.causal_pipe = self.load_causal_model()
        
        # 流水线相关参数
        self.denoising_queue = queue.Queue()  # denoising任务队列
        self.vae_queue = queue.Queue()  # decoding任务队列
        self.result_buffer = {}  # 保持字典结构，用于按chunk_id排序
        # 多线程控制
        self.denoising_thread = None
        self.vae_thread = None
        self.result_lock = threading.Lock()
        # 事件触发同步机制
        self.denoising_events = deque()
        self.vae_events = deque()
        self.current_clock = 0
    
    def load_causal_model(self):
        """Load causal inference pipeline"""
        # Initialize model manager
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models(
            [
                self.args.dit_path.split(","),   # load dit
                self.args.text_encoder_path,     # load text encoder
                self.args.vae_path               # load vae
            ],
            torch_dtype=self.dtype,
            device='cpu',
        )
        
        # Create causal inference pipeline
        causal_pipe = CausalInferencePipeline.from_model_manager(
            model_manager=model_manager,
            args=self.args,
            device=self.device
        )
        
        # Move VAE to different GPU for pipeline parallelism
        causal_pipe.vae.to("cuda:1")
        print(f"Move VAE to cuda:1 for pipeline parallelism")
        
        return causal_pipe
    
    def run_pipeline(
            self,
            noise_blocks,
            text_prompts,
            image_path,
            audio_path,
            initial_latent,
            streaming_callback,
            session_id,
            return_latents=False
        ):
        """
        Run pipelined causal inference
        
        Args:
            noise_blocks: List of noise tensors for each block
            text_prompts: Text prompts for generation
            image_path: Path to reference image
            audio_path: Path to audio file
            initial_latent: Initial latent for I2V
            streaming_callback: Callback function for streaming
            session_id: Session ID
            return_latents: Whether to return latents
        """
        # clear pipeline state
        self.current_clock = 0
        self.denoising_queue.queue.clear()
        self.vae_queue.queue.clear()
        self.result_buffer.clear()
        self.denoising_events.clear()
        self.vae_events.clear()
        
        # prepare
        num_blocks = len(noise_blocks)
        completed_chunks = 0
        total_frames_generated = 0
        
        # run async threads
        self.denoising_thread = threading.Thread(target=self._causal_denoising_worker, daemon=True)
        self.vae_thread = threading.Thread(target=self._vae_worker, daemon=True)
        self.denoising_thread.start()
        self.vae_thread.start()
        
        # Prepare conditioning (similar to causal_inference.py forward method)
        # prepare image_condition
        img_lat = None
        if image_path is not None:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.causal_pipe.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
            self.causal_pipe.vae.to(self.device)
            img_lat = self.causal_pipe.vae.encode(videos=image.to(dtype=self.dtype), device=self.device)
            # Expand to match the total number of frames
            total_frames = sum([noise.shape[1] for noise in noise_blocks])
            img_lat = img_lat.repeat(1, 1, total_frames, 1, 1)
            msk = torch.zeros_like(img_lat)[:, :1]
            msk[:, :, 1:] = 1
            img_lat = torch.cat([img_lat, msk], dim=1)
            print("img_lat:", img_lat.shape)
            
        # prepare audio_condition
        audio_emb = None
        if audio_path is not None:
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
            
            # Trim audio to 5 seconds
            max_duration = 5.0  # 5 seconds
            max_samples = int(max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                print(f"Audio trimmed to {max_duration} seconds")
            
            input_values = np.squeeze(
                self.causal_pipe.wav_feature_extractor(audio, sampling_rate=16000).input_values
            )
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            ori_audio_len = audio_len = 81
            input_values = input_values.unsqueeze(0)
            
            with torch.no_grad():
                self.causal_pipe.audio_encoder.to(self.device)
                hidden_states = self.causal_pipe.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
                audio_emb = audio_embeddings.permute(0, 2, 1)[:, :, :, None, None]
                audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2) # 1, 768, 44, 1, 1
                audio_emb = self.causal_pipe.generator.audio_proj(audio_emb.to(self.dtype))
                audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.causal_pipe.generator.audio_cond_projs], 0)
                print("audio_shape:", audio_emb.shape)
        else:
            print("Detect No audio input!!")
            audio_emb = None
        
        # Submit first block
        self.denoising_events.append(threading.Event())
        
        # Calculate frame ranges for each block
        current_start_frame = 0
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        
        for block_id in range(num_blocks):
            print(f"===================== Processing block {block_id} =====================")
            
            noise = noise_blocks[block_id]
            block_num_frames = noise.shape[1]
            
            # Prepare block-specific conditioning
            block_img_lat = None
            block_audio_emb = None
            
            if img_lat is not None:
                block_img_lat = img_lat[:, :, current_start_frame:current_start_frame + block_num_frames]
            
            if audio_emb is not None:
                # For audio, we need to slice appropriately based on frame timing
                block_audio_emb = audio_emb[:, current_start_frame:current_start_frame + block_num_frames]
            
            # Prepare block initial latent
            block_initial_latent = None
            if initial_latent is not None and current_start_frame < num_input_frames:
                end_frame = min(current_start_frame + block_num_frames, num_input_frames)
                if end_frame > current_start_frame:
                    block_initial_latent = initial_latent[:, current_start_frame:end_frame]
            
            # Create denoising task
            denoising_task = {
                "block_id": block_id,
                "noise": noise.clone(),
                "text_prompts": text_prompts,
                "img_lat": block_img_lat.clone() if block_img_lat is not None else None,
                "audio_emb": block_audio_emb.clone() if block_audio_emb is not None else None,
                "initial_latent": block_initial_latent.clone() if block_initial_latent is not None else None,
                "current_start_frame": current_start_frame,
                "return_latents": return_latents
            }
            
            self.denoising_queue.put(denoising_task)
            
            # Wait for denoising completion
            self.denoising_events[self.current_clock].wait()
            
            # For blocks after the first, also wait for previous VAE completion
            if block_id > 0:
                self.vae_events[self.current_clock - 1].wait()
                # Send previous chunk frames
                total_frames_generated = self._send_chunk_frames(
                    block_id - 1, streaming_callback, session_id, 
                    total_frames_generated, num_blocks
                )
            
            # Prepare for next block
            if block_id < num_blocks - 1:
                self.current_clock += 1
                self.denoising_events.append(threading.Event())
            
            current_start_frame += block_num_frames
        
        # Send last chunk
        total_frames_generated = self._send_chunk_frames(
            num_blocks - 1, streaming_callback, session_id, 
            total_frames_generated, num_blocks
        )
        
        # Collect all results
        all_results = []
        for block_id in range(num_blocks):
            while block_id not in self.result_buffer:
                time.sleep(0.1)
            all_results.append(self.result_buffer[block_id])
        
        return all_results
    
    def forward(self, 
                prompt,             # text prompt
                image_path=None,    # reference image
                audio_path=None,    # reference audio
                num_blocks=None,    # number of blocks for causal generation
                frames_per_block=None,  # frames per block
                height=720, 
                width=720,
                num_steps=None,
                negative_prompt=None,
                guidance_scale=None,
                audio_scale=None,
                streaming_callback=None,  # 流式生成回调函数
                streaming_mode=False,    # 是否启用流式模式
                session_id=None,         # 会话ID
                return_latents=False):   # 是否返回latents
        
        # Set default values
        num_blocks = num_blocks if num_blocks is not None else getattr(self.args, 'num_blocks', 7)
        frames_per_block = frames_per_block if frames_per_block is not None else getattr(self.args, 'num_frame_per_block', 3)
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale
        
        print(f"Pipelined causal inference with {num_blocks} blocks, {frames_per_block} frames per block")
        
        # Calculate total frames
        total_frames = num_blocks * frames_per_block
        
        # Prepare noise blocks
        noise_blocks = []
        for block_id in range(num_blocks):
            block_noise = torch.randn([1, frames_per_block, 16, height//8, width//8], 
                                    device=self.device, dtype=self.dtype)
            noise_blocks.append(block_noise)
        
        # Prepare initial latent (for I2V)
        initial_latent = None
        if image_path is not None:
            # For I2V, we typically use the first frame as initial latent
            # This is a simplified version - you might need to adjust based on your specific needs
            initial_latent = torch.zeros([1, 1, 16, height//8, width//8], 
                                       device=self.device, dtype=self.dtype)
        
        # 流式生成准备
        output_dir = f'demo_out/causal_streaming_videos_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Video will be saved to: {output_dir}")

        # 发送开始事件
        if streaming_callback:
            streaming_callback({
                'type': 'start',
                'session_id': session_id,
                'total_blocks': num_blocks,
                'message': 'Pipelined causal video generation started',
                'output_dir': output_dir
            })

        # 第一阶段：提前进行prompt encoding（只执行一次）
        print("Phase 1: Prompt encoding (executed once)")
        start_time = time.time()
        # Text conditioning is handled inside the causal pipeline
        prompt_time = time.time() - start_time
        print(f"Prompt encoding preparation completed in {prompt_time:.4f}s")
        
        # 发送prompt encoding完成事件
        if streaming_callback:
            streaming_callback({
                'type': 'prompt_encoding_complete',
                'session_id': session_id,
                'time_taken': prompt_time,
                'message': 'Prompt encoding preparation completed'
            })

        # 第二阶段：流水线化的block生成
        print("Phase 2: Pipelined causal block generation")
        results = self.run_pipeline(
            noise_blocks=noise_blocks,
            text_prompts=prompt,
            image_path=image_path,
            audio_path=audio_path,
            initial_latent=initial_latent,
            streaming_callback=streaming_callback,
            session_id=session_id,
            return_latents=return_latents
        )
        
        return results
    
    def _causal_denoising_worker(self):
        """
        Causal denoising worker
        从队列中获取任务并执行因果推理
        """
        while True:
            try:
                task = self.denoising_queue.get(timeout=0.1)
                block_id = task["block_id"]
                print(f"Processing causal denoising inference for block {block_id}")
                
                # Perform causal inference for this block
                result = self.causal_pipe.inference(
                    noise=task['noise'],
                    text_prompts=task['text_prompts'],
                    img_lat=task['img_lat'],
                    audio_embed=task['audio_emb'],
                    initial_latent=None,
                    # initial_latent=task['initial_latent'],
                    return_latents=task['return_latents']
                )
                
                if task['return_latents']:
                    video, latents = result
                else:
                    video = result
                    latents = None
                
                # Trigger denoising event
                self.denoising_events[self.current_clock].set()
                
                # Create VAE event for this block
                self.vae_events.append(threading.Event())
                
                # Add VAE decoding task (video is already decoded by causal_pipe.inference)
                # But we still need to process it for streaming
                vae_task = {
                    "block_id": block_id,
                    "video": video,
                    "latents": latents
                }
                self.vae_queue.put(vae_task)
                
                print(f"Causal denoising inference completed for block {block_id}")
                self.denoising_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in causal denoising inference for block {block_id}: {e}")
                traceback.print_exc()
                self.denoising_queue.task_done()
    
    def _vae_worker(self):
        """
        VAE processing worker (mainly for format conversion and streaming)
        """
        while True:
            try:
                task = self.vae_queue.get(timeout=0.1)
                block_id = task["block_id"]
                print(f"Processing video formatting for block {block_id}")
                
                video = task["video"]
                latents = task["latents"]
                
                # Trigger VAE event
                self.vae_events[self.current_clock - 1].set()
                
                # Store result in buffer with lock
                with self.result_lock:
                    self.result_buffer[block_id] = {
                        "video": video,
                        "latents": latents
                    }
                
                print(f"Video formatting completed for block {block_id}")
                self.vae_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in video formatting for block {block_id}: {e}")
                traceback.print_exc()
                self.vae_queue.task_done()

    def _send_chunk_frames(self, block_id, streaming_callback, session_id, total_frames_generated, total_blocks):
        """Send frames for a completed block"""
        print(f"Sending block {block_id} frames")
        
        # Wait for result to be available
        while block_id not in self.result_buffer:
            time.sleep(0.1)
        
        block_data = self.result_buffer[block_id]
        video = block_data["video"]
        
        if streaming_callback is None:
            return total_frames_generated + video.shape[1]
        
        try:
            # Convert video tensor to frames and create video chunk
            video_base64 = self._create_video_chunk(video)
            
            total_frames_generated += video.shape[1]
            progress_overall = (total_frames_generated / (total_blocks * self.args.num_frame_per_block)) * 100
            
            streaming_callback({
                "type": "video_chunk",
                "session_id": session_id,
                "video_data": video_base64,
                "block_number": block_id,
                "total_blocks": total_blocks,
                "progress": progress_overall,
                "frames_in_block": video.shape[1],
                "message": f"Block {block_id} completed (causal pipelined)"
            })
            
        except Exception as e:
            print(f"Error creating video chunk for block {block_id}: {e}")
            # Fallback to frame-by-frame sending
            total_frames_generated = self._send_frames_fallback(
                video, block_id, streaming_callback, 
                session_id, total_frames_generated, total_blocks
            )
        
        return total_frames_generated

    def _create_video_chunk(self, video):
        """Create a video chunk from video tensor"""
        # Convert video tensor to numpy
        video_np = (video.squeeze(0).permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        
        # Create a simple MP4 video chunk using imageio or cv2
        import imageio
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            imageio.mimsave(tmp_file.name, video_np, fps=getattr(self.args, 'fps', 16))
            
            # Read the video file and encode as base64
            with open(tmp_file.name, 'rb') as f:
                video_bytes = f.read()
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            return video_base64

    def _send_frames_fallback(self, video, block_id, streaming_callback, session_id, total_frames_generated, total_blocks):
        """Fallback method: send frames one by one"""
        print(f"Using fallback method for block {block_id}")
        
        for frame_idx in range(video.shape[1]):
            frame_data = video[:, frame_idx]
            # Convert to base64
            frame_np = (frame_data.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            frame_pil = Image.fromarray(frame_np)
            buffer = io.BytesIO()
            frame_pil.save(buffer, format='JPEG', quality=85)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            total_frames_generated += 1
            progress_overall = (total_frames_generated / (total_blocks * self.args.num_frame_per_block)) * 100
            
            streaming_callback({
                "type": "video_frame",
                "session_id": session_id,
                "frame_data": frame_base64,
                "frame_number": total_frames_generated,
                "block_number": block_id,
                "progress": progress_overall,
                "block_progress": ((frame_idx + 1) / video.shape[1]) * 100
            })
        
        # Block complete
        streaming_callback({
            "type": "block_complete",
            "session_id": session_id,
            "block_number": block_id,
            "total_blocks": total_blocks,
            "frames_in_block": video.shape[1],
            "total_frames_generated": total_frames_generated,
            "progress": (total_frames_generated / (total_blocks * self.args.num_frame_per_block)) * 100,
            "message": f"Block {block_id} completed (causal pipelined)"
        })
        
        return total_frames_generated


def main():
    """Main function to test the pipelined causal inference pipeline."""
    torch.set_grad_enabled(False)
    args = parse_args()
    # Set device based on rank
    device = torch.device(f"cuda:{getattr(args, 'rank', 0)}")
    
    # Create pipelined causal inference pipeline
    pipeline = PipelinedCausalInferencePipeline(args)
    print("Pipelined causal inference pipeline initialized successfully!")
    
    # Prepare test parameters
    text_prompts = getattr(args, 'prompt', "A realistic video of a man speaking directly to the camera on a sofa, with dynamic and rhythmic hand gestures that complement his speech.")
    
    print(f"Text prompts: {text_prompts}")
    
    # Test streaming callback
    def test_streaming_callback(data):
        print(f"Streaming callback: {data['type']} - {data.get('message', '')}")
    
    # Perform pipelined causal inference
    print("Starting pipelined causal inference...")
    
    results = pipeline(
        prompt=text_prompts,
        image_path="/data2/jdsu/projects/OmniAvatar/examples/images/0000.jpeg",
        audio_path="/data2/jdsu/projects/OmniAvatar/examples/audios/0000.MP3",
        num_blocks=7,
        frames_per_block=3,
        streaming_callback=test_streaming_callback,
        session_id="test_session",
        return_latents=False
    )
    
    print(f"Generated {len(results)} blocks")
    
    # Save results
    output_path = "generated_causal_pipelined_video.mp4"
    if results:
        # Concatenate all video blocks
        all_videos = []
        for result in results:
            all_videos.append(result["video"])
        
        # Concatenate along time dimension
        final_video = torch.cat(all_videos, dim=1)
        
        # Save video
        import imageio
        video_np = (final_video.squeeze(0).permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        imageio.mimsave(output_path, video_np, fps=getattr(args, 'fps', 16))
        print(f"Video saved to: {output_path}")
    
    print("Pipelined causal inference completed successfully!")


if __name__ == "__main__":
    main()