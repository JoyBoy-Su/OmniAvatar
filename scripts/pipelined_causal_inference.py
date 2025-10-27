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
from typing import List, Optional, Dict, Any
from collections import defaultdict

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
        
        # 性能统计相关
        self.timing_stats = defaultdict(list)  # 存储各步骤的耗时统计
        self.timing_lock = threading.Lock()  # 保护timing_stats的线程锁
    
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
            noise: torch.Tensor,
            batch_size: int,
            num_blocks: int,
            num_input_frames: int,
            initial_latent: torch.Tensor,
            conditional_dict: dict,
            img_lat: torch.Tensor,
            output: torch.Tensor,
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
        
        # 清空性能统计
        with self.timing_lock:
            self.timing_stats.clear()
        
        # run async threads
        self.denoising_thread = threading.Thread(target=self._causal_denoising_worker, daemon=True)
        self.vae_thread = threading.Thread(target=self._vae_worker, daemon=True)
        self.denoising_thread.start()
        self.vae_thread.start()
        
        # Step 2: cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            print("INITIAL_LATENT is not None!!")
            raise ValueError
        
        # Step 3: temporal denoising loop
        print(f"NUM BLOCKS is {num_blocks}")
        all_num_frames = [self.args.num_frame_per_block] * num_blocks
        total_frames_generated = 0
        for current_num_frames in all_num_frames:
            print(f"===================== Current clock: {self.current_clock} ======================")
            print(f"Processing frame {current_start_frame - num_input_frames} to {current_start_frame + current_num_frames - num_input_frames}.")
            noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            y_input = conditional_dict["image"][:, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            audio_input = conditional_dict["audio"][:,current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            block_conditional_dict = conditional_dict.copy()
            block_conditional_dict.update(image=y_input.clone(), audio=audio_input.clone())
            
            denoising_task = {
                "chunk_id": self.current_clock,
                "current_start_frame": current_start_frame,
                "current_num_frames": current_num_frames,
                "img_lat": img_lat.clone(),
                "batch_size": batch_size,
                "noisy_input": noisy_input.clone(),
                "block_conditional_dict": block_conditional_dict,
                "output": output.clone()
            }
            self.denoising_queue.put(denoising_task)
            # wait denoising clock and decoding clock - 1
            self.denoising_events.append(threading.Event())
            self.denoising_events[self.current_clock].wait()
            if self.current_clock >= 1:
                self.vae_events[self.current_clock - 1].wait()
            # send current chunk, TODO: update interface
            total_frames_generated = self._send_chunk_frames(self.current_clock - 1, streaming_callback, session_id, total_frames_generated, num_blocks)
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            self.current_clock += 1
        # TODO: wait decoding the last chunk?
        self.vae_events[self.current_clock - 1].wait()
        # send the last chunk, TODO: update interface
        total_frames_generated = self._send_chunk_frames(self.current_clock - 1, streaming_callback, session_id, total_frames_generated, num_blocks)
        
        # 打印性能统计报告
        self._print_timing_report()
  
    def _record_timing(self, step_name: str, duration: float, chunk_id: int = None):
        """记录步骤耗时"""
        with self.timing_lock:
            key = f"{step_name}_chunk_{chunk_id}" if chunk_id is not None else step_name
            self.timing_stats[key].append(duration)
    
    def _print_timing_report(self):
        """打印性能统计报告"""
        print("\n" + "="*80)
        print("PERFORMANCE TIMING REPORT")
        print("="*80)
        
        with self.timing_lock:
            # 按类别分组统计
            forward_steps = {}
            denoising_steps = {}
            vae_steps = {}
            
            for key, times in self.timing_stats.items():
                if key.startswith('forward_'):
                    forward_steps[key] = times
                elif 'denoising' in key or 'generator_forward' in key:
                    denoising_steps[key] = times
                elif 'vae' in key or 'decode' in key:
                    vae_steps[key] = times
            
            # 打印Forward阶段统计
            if forward_steps:
                print("\n📊 FORWARD PHASE TIMING:")
                print("-" * 50)
                total_forward_time = 0
                for step, times in forward_steps.items():
                    avg_time = np.mean(times)
                    total_time = np.sum(times)
                    total_forward_time += total_time
                    print(f"  {step:<35}: {avg_time:>8.3f}s (total: {total_time:>8.3f}s, count: {len(times)})")
                print(f"  {'TOTAL FORWARD TIME':<35}: {total_forward_time:>8.3f}s")
            
            # 打印Denoising阶段统计
            if denoising_steps:
                print("\n🔄 DENOISING PHASE TIMING:")
                print("-" * 50)
                total_denoising_time = 0
                for step, times in denoising_steps.items():
                    avg_time = np.mean(times)
                    total_time = np.sum(times)
                    total_denoising_time += total_time
                    print(f"  {step:<35}: {avg_time:>8.3f}s (total: {total_time:>8.3f}s, count: {len(times)})")
                print(f"  {'TOTAL DENOISING TIME':<35}: {total_denoising_time:>8.3f}s")
            
            # 打印VAE阶段统计
            if vae_steps:
                print("\n🎬 VAE DECODE PHASE TIMING:")
                print("-" * 50)
                total_vae_time = 0
                for step, times in vae_steps.items():
                    avg_time = np.mean(times)
                    total_time = np.sum(times)
                    total_vae_time += total_time
                    print(f"  {step:<35}: {avg_time:>8.3f}s (total: {total_time:>8.3f}s, count: {len(times)})")
                print(f"  {'TOTAL VAE TIME':<35}: {total_vae_time:>8.3f}s")
            
            # 打印总体统计
            print("\n📈 OVERALL STATISTICS:")
            print("-" * 50)
            total_pipeline_time = sum(np.sum(times) for times in self.timing_stats.values())
            print(f"  {'TOTAL PIPELINE TIME':<35}: {total_pipeline_time:>8.3f}s")
            
            # 分析瓶颈
            print("\n🔍 BOTTLENECK ANALYSIS:")
            print("-" * 50)
            step_totals = {step: np.sum(times) for step, times in self.timing_stats.items()}
            sorted_steps = sorted(step_totals.items(), key=lambda x: x[1], reverse=True)
            for i, (step, total_time) in enumerate(sorted_steps[:5]):
                percentage = (total_time / total_pipeline_time) * 100 if total_pipeline_time > 0 else 0
                print(f"  {i+1}. {step:<30}: {total_time:>8.3f}s ({percentage:>5.1f}%)")
        
        print("="*80)

    @torch.no_grad()
    def forward(
        self,
        noise: torch.Tensor,
        text_prompts: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        streaming_callback=None,  # 流式生成回调函数
        session_id=None
    ) -> torch.Tensor:
        """
        Forward pass with automatic conditioning initialization.
        
        Args:
            noise: Input noise tensor [batch_size, num_output_frames, channels, height, width]
            text_prompts: Text prompts for generation
            image_path: Path to reference image (optional)
            audio_path: Path to audio file (optional)
            initial_latent: Initial latent for I2V [batch_size, num_input_frames, channels, height, width]
            return_latents: Whether to return latents
            
        Returns:
            Generated video tensor [batch_size, num_frames, channels, height, width]
        """
        # prepare image condition
        if image_path is not None:
            start_time = time.time()
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.causal_pipe.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
            image_prep_time = time.time() - start_time
            self._record_timing("forward_image_preprocessing", image_prep_time)
            
            start_time = time.time()
            self.causal_pipe.vae.to("cuda:1")
            img_lat = self.causal_pipe.vae.encode(videos=image.to(dtype=self.dtype),device="cuda:1").repeat(1,1,21,1,1)
            msk = torch.zeros_like(img_lat)[:,:1]
            msk[:, :, 1:] = 1
            img_lat = torch.cat([img_lat, msk], dim=1)
            image_encode_time = time.time() - start_time
            self._record_timing("forward_image_vae_encode", image_encode_time)
            print("img_lat:",img_lat.shape)
        
        # prepare audio_condition
        if audio_path is not None:
            start_time = time.time()
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
            audio_len = (noise.shape[1] - 1) * 4 + 1
            input_values = input_values.unsqueeze(0)
            audio_prep_time = time.time() - start_time
            self._record_timing("forward_audio_preprocessing", audio_prep_time)
            
            start_time = time.time()
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
                print("audio_shape:",audio_emb.shape)
            audio_encode_time = time.time() - start_time
            self._record_timing("forward_audio_encoding", audio_encode_time)
        else:
            print("Detect No audio input!!")
            audio_embeddings = None
        
        # inference (prepare for run_pipeline)
        batch_size, num_frames, num_channels, height, width = noise.shape
        # frame block calculations
        assert num_frames % self.args.num_frame_per_block == 0
        num_blocks = num_frames // self.args.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        # text conditioning
        start_time = time.time()
        self.causal_pipe.text_encoder.to("cuda")
        conditional_dict = self.causal_pipe.encode_text_prompts(text_prompts, positive=True)
        conditional_dict["image"] = img_lat
        conditional_dict["audio"] = audio_emb
        text_encode_time = time.time() - start_time
        self._record_timing("forward_text_encoding", text_encode_time)
        
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        # step 1: initialize KV caches
        start_time = time.time()
        self.causal_pipe.setup_caches(batch_size, noise.dtype, noise.device)
        cache_setup_time = time.time() - start_time
        self._record_timing("forward_cache_setup", cache_setup_time)
        
        # run pipeline
        self.run_pipeline(
            noise=noise,
            batch_size=batch_size,
            num_blocks=num_blocks,
            num_input_frames=num_input_frames,
            initial_latent=initial_latent,
            conditional_dict=conditional_dict,
            img_lat=img_lat,
            output=output,
            streaming_callback=streaming_callback,
            session_id=session_id,
            return_latents=return_latents
        )
    
    @torch.no_grad()
    def _causal_denoising_worker(self):
        """
        Causal denoising worker
        从队列中获取任务并执行因果推理
        """
        while True:
            try:
                task = self.denoising_queue.get(timeout=0.1)
                chunk_id = task["chunk_id"]
                print(f"Processing causal denoising inference for chunk {chunk_id}")
                
                # 记录整个denoising任务的开始时间
                denoising_start_time = time.time()
                
                current_start_frame = task["current_start_frame"]
                current_num_frames = task["current_num_frames"]
                img_lat = task["img_lat"]
                batch_size = task["batch_size"]
                noisy_input = task["noisy_input"]
                block_conditional_dict = task["block_conditional_dict"]
                output = task["output"]
                
                # Step 3.1: Spatial denoising loop
                denoising_loop_start = time.time()
                for index, current_timestep in enumerate(self.causal_pipe.denoising_step_list):
                    step_start_time = time.time()
                    
                    if current_start_frame == 0:
                        noisy_input[:, :1] = img_lat[:, :16, :1].permute(0, 2, 1, 3, 4)
                    timestep = torch.ones([batch_size, current_num_frames], device=noisy_input.device, dtype=torch.int64) * current_timestep
                    
                    # generate
                    generator_start_time = time.time()
                    v, denoised_pred = self.causal_pipe.generator_forward(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=block_conditional_dict,
                        timestep=timestep,
                        kv_cache=self.causal_pipe.kv_cache1,
                        crossattn_cache=self.causal_pipe.crossattn_cache,
                        current_start=current_start_frame * self.causal_pipe.frame_seq_length
                    )
                    generator_time = time.time() - generator_start_time
                    self._record_timing(f"denoising_generator_forward_step_{index}", generator_time, chunk_id)
                    
                    if index < len(self.causal_pipe.denoising_step_list) - 1:
                        noise_start_time = time.time()
                        next_timestep = self.causal_pipe.denoising_step_list[index + 1]
                        noisy_input = self.causal_pipe.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones([batch_size * current_num_frames], device=noisy_input.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                        noise_time = time.time() - noise_start_time
                        self._record_timing(f"denoising_add_noise_step_{index}", noise_time, chunk_id)
                    
                    step_time = time.time() - step_start_time
                    self._record_timing(f"denoising_total_step_{index}", step_time, chunk_id)
                
                denoising_loop_time = time.time() - denoising_loop_start
                self._record_timing("denoising_loop_total", denoising_loop_time, chunk_id)
                
                # Step 3.2: record the model's output
                if current_start_frame == 0:
                    denoised_pred[:, :1] = img_lat[:, :16, :1].permute(0, 2, 1, 3, 4)
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred # latents: denoised_pred
                
                # Step 3.3: return with timestep zero to update KV cache using clean context
                context_start_time = time.time()
                context_timestep = torch.ones_like(timestep) * 0
                self.causal_pipe.generator_forward(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=block_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.causal_pipe.kv_cache1,
                    crossattn_cache=self.causal_pipe.crossattn_cache,
                    current_start=current_start_frame * self.causal_pipe.frame_seq_length,
                )
                context_time = time.time() - context_start_time
                self._record_timing("denoising_context_update", context_time, chunk_id)
                
                # 记录整个denoising任务的总时间
                total_denoising_time = time.time() - denoising_start_time
                self._record_timing("denoising_total", total_denoising_time, chunk_id)
                
                # Trigger denoising event
                self.denoising_events[self.current_clock].set()
                
                # Create VAE event for this block
                self.vae_events.append(threading.Event())
                
                # Add VAE decoding task (video is already decoded by causal_pipe.inference)
                # But we still need to process it for streaming
                # if current_start_frame > 0:
                #     task.update({
                #         "latents": output[:, current_start_frame - 1:current_start_frame + current_num_frames].clone()
                #     })
                # else:
                task.update({
                    "latents": denoised_pred.clone()
                })
                self.vae_queue.put(task)
                
                print(f"Causal denoising inference completed for block {chunk_id}")
                self.denoising_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in causal denoising inference for block {chunk_id}: {e}")
                traceback.print_exc()
                self.denoising_queue.task_done()
    
    @torch.no_grad()
    def _vae_worker(self):
        """
        VAE processing worker (mainly for format conversion and streaming)
        """
        while True:
            try:
                task = self.vae_queue.get(timeout=0.1)
                chunk_id = task["chunk_id"]
                print(f"Processing video formatting for block {chunk_id}")
                
                # 记录整个VAE任务的开始时间
                vae_start_time = time.time()
                
                latents: torch.Tensor = task["latents"]
                
                # 数据准备阶段
                prep_start_time = time.time()
                latents = latents.permute(0, 2, 1, 3, 4)    # (b, c, t, h, w)
                latents = latents.to("cuda:1")
                prep_time = time.time() - prep_start_time
                self._record_timing("vae_data_preparation", prep_time, chunk_id)
                
                # VAE解码阶段
                decode_start_time = time.time()
                video = self.causal_pipe.vae.decode(latents, device="cuda:1").permute(0, 2, 1, 3, 4)
                decode_time = time.time() - decode_start_time
                self._record_timing("vae_decode", decode_time, chunk_id)
                
                # 后处理阶段
                postprocess_start_time = time.time()
                # video = video[:, :, 1:].permute(0, 2, 1, 3, 4)
                # video = video[:, :, 1:]
                # Ensure video is in float32 for compatibility with numpy conversion
                print(f"Video before float conversion - dtype: {video.dtype}, shape: {video.shape}")
                video = (video.float() + 1) / 2  # Normalize from [-1, 1] to [0, 1]
                print(f"Video after float conversion - dtype: {video.dtype}, shape: {video.shape}")
                postprocess_time = time.time() - postprocess_start_time
                self._record_timing("vae_postprocessing", postprocess_time, chunk_id)
                
                # Store result in buffer with lock
                buffer_start_time = time.time()
                task.update({"video": video})
                with self.result_lock:
                    self.result_buffer[chunk_id] = task
                buffer_time = time.time() - buffer_start_time
                self._record_timing("vae_buffer_storage", buffer_time, chunk_id)
                
                # 记录整个VAE任务的总时间
                total_vae_time = time.time() - vae_start_time
                self._record_timing("vae_total", total_vae_time, chunk_id)
                
                print(f"Video formatting completed for block {chunk_id}")
                # trigger vae event
                self.vae_events[self.current_clock - 1].set()
                self.vae_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in video formatting for block {chunk_id}: {e}")
                traceback.print_exc()
                self.vae_queue.task_done()

    def _send_chunk_frames(self, chunk_id, streaming_callback, session_id, total_frames_generated, total_blocks):
        if chunk_id < 0:
            return 0
        """Send frames for a completed block using frame-by-frame method"""
        print(f"Sending block {chunk_id} frames")
        
        # 记录发送帧的开始时间
        send_start_time = time.time()
        
        # Wait for result to be available
        wait_start_time = time.time()
        while chunk_id not in self.result_buffer:
            time.sleep(0.1)
        wait_time = time.time() - wait_start_time
        self._record_timing("send_wait_for_result", wait_time, chunk_id)
        
        block_data = self.result_buffer[chunk_id]
        video = block_data["video"]
        # print(f"video: {video.shape}")
        # output_path = f"generated_video_{chunk_id}.mp4"
        # video_np = (video.squeeze(0).permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        # import imageio
        # print(f"video np: {video_np.shape}")
        # imageio.mimsave(output_path, video_np, fps=16)
        # print(f"Video saved to: {output_path}")
        
        # if streaming_callback is None:
        #     return total_frames_generated + video.shape[1]
        
        # 直接使用逐帧发送方式，避免视频流同步问题
        print(f"Using frame-by-frame method for chunk {chunk_id}")
        frame_send_start_time = time.time()
        total_frames_generated = self._send_frames_fallback(
            video, chunk_id, streaming_callback, 
            session_id, total_frames_generated, total_blocks
        )
        frame_send_time = time.time() - frame_send_start_time
        self._record_timing("send_frames_processing", frame_send_time, chunk_id)
        
        # 记录整个发送过程的总时间
        total_send_time = time.time() - send_start_time
        self._record_timing("send_total", total_send_time, chunk_id)
        
        return total_frames_generated + video.shape[1]

    def _send_frames_fallback(self, video, chunk_id, streaming_callback, session_id, total_frames_generated, total_blocks):
        """Frame-by-frame sending method (following pipelined_inference.py pattern)"""
        print(f"Using frame-by-frame method for chunk {chunk_id}")
        # import pdb; pdb.set_trace()
        total_frames = (total_blocks * self.args.num_frame_per_block - 1) * 4 + 1
        # import pdb; pdb.set_trace()
        
        frame_processing_times = []
        for frame_idx in range(video.shape[1]):
            frame_start_time = time.time()
            
            frame_data = video[:, frame_idx]
            # Convert to base64 (same as pipelined_inference.py)
            # First convert bfloat16 to float32, then to numpy
            print(f"Frame {frame_idx} data type: {frame_data.dtype}, shape: {frame_data.shape}")
            
            # 图像转换时间
            convert_start_time = time.time()
            frame_np = (frame_data.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype('uint8')
            frame_pil = Image.fromarray(frame_np)
            # frame_pil.save(f"examples/frame_{frame_idx}.png")
            buffer = io.BytesIO()
            frame_pil.save(buffer, format='JPEG', quality=85)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            convert_time = time.time() - convert_start_time
            
            total_frames_generated += 1
            progress_overall = (total_frames_generated / total_frames) * 100 if total_frames else None
            
            # 回调发送时间
            callback_start_time = time.time()
            import pdb; pdb.set_trace()
            streaming_callback({
                "type": "video_frame",
                "session_id": session_id,
                "frame_data": frame_base64,
                "frame_number": total_frames_generated,
                "total_frames": total_frames,
                "chunk_number": chunk_id,
                "progress": progress_overall,
                "chunk_progress": ((frame_idx + 1) / video.shape[1]) * 100
            })
            callback_time = time.time() - callback_start_time
            
            frame_total_time = time.time() - frame_start_time
            frame_processing_times.append({
                'convert_time': convert_time,
                'callback_time': callback_time,
                'total_time': frame_total_time
            })
        
        # 记录帧处理的平均时间
        if frame_processing_times:
            avg_convert_time = np.mean([t['convert_time'] for t in frame_processing_times])
            avg_callback_time = np.mean([t['callback_time'] for t in frame_processing_times])
            avg_frame_time = np.mean([t['total_time'] for t in frame_processing_times])
            
            self._record_timing("send_frame_convert_avg", avg_convert_time, chunk_id)
            self._record_timing("send_frame_callback_avg", avg_callback_time, chunk_id)
            self._record_timing("send_frame_total_avg", avg_frame_time, chunk_id)
        
        # Chunk complete (same format as pipelined_inference.py)
        progress_overall_after_chunk = (total_frames_generated / total_frames) * 100 if total_frames else None
        streaming_callback({
            "type": "chunk_complete",
            "session_id": session_id,
            "chunk_number": chunk_id,
            "total_chunks": total_blocks,
            "frames_in_chunk": video.shape[1],
            "total_frames_generated": total_frames_generated,
            "progress": progress_overall_after_chunk,
            "message": f"Chunk {chunk_id} completed (causal pipelined)"
        })
        
        return total_frames_generated
