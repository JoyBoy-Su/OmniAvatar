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
from concurrent.futures import ThreadPoolExecutor
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
import soundfile as sf
from openai import OpenAI

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


class QwenOmniTalker:
    """Qwen-Omni语音对话处理器 - 支持多轮对话"""
    
    def __init__(self, api_key="sk-63ad221681734d339b8171797204f105", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.system_message = {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        }
        # 存储每个会话的对话历史 {session_id: [messages]}
        self.conversation_history = {}
        # 默认最大历史轮数（可根据需要调整）
        self.max_history_turns = 10
    
    def process_audio_conversation(self, audio_path, session_id=None, prompt="Analyze this audio and respond naturally."):
        """
        处理音频对话，返回回复的音频文件路径（支持多轮对话）

        Args:
            audio_path: 输入音频文件路径
            session_id: 会话ID，用于生成唯一的输出文件名和管理对话历史
            prompt: 文本提示词，默认为分析音频内容

        Returns:
            tuple: (reply_audio_path, reply_text) 回复音频路径和文本内容
        """
        try:
            # 使用默认session_id如果未提供
            if session_id is None:
                session_id = "default"
            
            # 初始化该会话的历史记录（如果不存在）
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            # 读取音频文件并编码为base64
            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            # 构建当前用户消息
            current_user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio_base64}", "format": "wav"}},
                ],
            }
            
            # 构建完整的消息列表：系统消息 + 历史消息 + 当前消息
            messages = [self.system_message] + self.conversation_history[session_id] + [current_user_message]
            
            # 调用Qwen-Omni API
            completion = self.client.chat.completions.create(
                model="qwen3-omni-flash",
                messages=messages,
                modalities=["text", "audio"],
                audio={
                    "voice": "Cherry",  # Cherry, Ethan, Serena, Chelsie is available
                    "format": "wav"
                },
                stream=True,
                stream_options={"include_usage": True}
            )
            
            # 收集响应
            text_parts = []
            audio_string = ""
            
            for chunk in completion:
                if chunk.choices:
                    if hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio:
                        try:
                            if "data" in chunk.choices[0].delta.audio:
                                audio_string += chunk.choices[0].delta.audio["data"]
                            elif "transcript" in chunk.choices[0].delta.audio:
                                text_parts.append(chunk.choices[0].delta.audio["transcript"])
                        except Exception as e:
                            print(f"Error processing audio chunk: {e}")
                    elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        text_parts.append(chunk.choices[0].delta.content)
                else:
                    if hasattr(chunk, 'usage') and chunk.usage:
                        print(f"Usage: {chunk.usage}")
            
            reply_text = "".join(text_parts)
            print(f"Qwen-Omni reply text: {reply_text}")
            
            # 保存对话历史：将用户消息和助手回复添加到历史记录
            # 用户消息（包含文本和音频数据）
            user_history_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio_base64}", "format": "wav"}},
                ]
            }
            
            # 助手回复消息
            assistant_history_message = {
                "role": "assistant",
                "content": reply_text
            }
            
            # 添加到历史记录
            self.conversation_history[session_id].append(user_history_message)
            self.conversation_history[session_id].append(assistant_history_message)
            
            # 限制历史记录长度（保留最近的 max_history_turns 轮对话）
            # 每轮对话包含2条消息（用户+助手），所以总共保留 max_history_turns * 2 条消息
            max_messages = self.max_history_turns * 2
            if len(self.conversation_history[session_id]) > max_messages:
                self.conversation_history[session_id] = self.conversation_history[session_id][-max_messages:]
            
            print(f"Session {session_id} history: {len(self.conversation_history[session_id])} messages")
            
            # 保存音频文件
            if audio_string:
                wav_bytes = base64.b64decode(audio_string)
                wav_array = np.frombuffer(wav_bytes, dtype=np.int16)
                
                # 生成唯一的输出文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_suffix = f"_{session_id}" if session_id else ""
                reply_audio_path = f"demo_out/qwen_omni_reply_{timestamp}{session_suffix}.wav"
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(reply_audio_path), exist_ok=True)
                
                # 保存音频文件
                sf.write(reply_audio_path, wav_array, samplerate=24000)
                print(f"Qwen-Omni reply audio saved to: {reply_audio_path}")
                
                return reply_audio_path, reply_text
            else:
                print("Warning: No audio data received from Qwen-Omni")
                return None, reply_text
                
        except Exception as e:
            print(f"Error in Qwen-Omni conversation: {e}")
            traceback.print_exc()
            return None, None
    
    def clear_session_history(self, session_id=None):
        """清除指定会话的对话历史"""
        if session_id is None:
            self.conversation_history.clear()
            print("All conversation history cleared")
        elif session_id in self.conversation_history:
            del self.conversation_history[session_id]
            print(f"Session {session_id} history cleared")
        else:
            print(f"Session {session_id} not found")


class PipelinedEvalPipeline(nn.Module):
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
        self.latents_buffer = {}
        # 多线程控制
        self.denoising_thread = None
        self.vae_thread = None
        self.result_lock = threading.Lock()
        self.latents_lock = threading.Lock()
        self.stop_workers = threading.Event()  # Signal to stop worker threads
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
            audio_path: str,
            id: str = None
        ):
        """
        Run pipelined causal inference
        
        Args:
            noise_blocks: List of noise tensors for each block
            text_prompts: Text prompts for generation
            image_path: Path to reference image
            audio_path: Path to audio file
            initial_latent: Initial latent for I2V
            id: Unique identifier for this generation session
            return_latents: Whether to return latents
        """
        # Stop old worker threads if they exist
        if self.denoising_thread is not None or self.vae_thread is not None:
            print("Stopping old worker threads...")
            self.stop_workers.set()
            if self.denoising_thread is not None and self.denoising_thread.is_alive():
                self.denoising_thread.join(timeout=2.0)
            if self.vae_thread is not None and self.vae_thread.is_alive():
                self.vae_thread.join(timeout=2.0)
            self.stop_workers.clear()
            print("Old worker threads stopped")

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
        current_start_frame_global = 0
        current_start_frame_local = 0
        if initial_latent is not None:
            print("INITIAL_LATENT is not None!!")
            # raise ValueError
        
        # Step 3: temporal denoising loop
        print(f"NUM BLOCKS is {num_blocks}")
        all_num_frames = [self.args.num_frame_per_block] * num_blocks
        for current_num_frames in all_num_frames:
            print(f"===================== Current clock: {self.current_clock} ======================")
            print(f"Processing frame {current_start_frame_global - num_input_frames} to {current_start_frame_global + current_num_frames - num_input_frames}.")
            noisy_input = noise[:, current_start_frame_global - num_input_frames:current_start_frame_global + current_num_frames - num_input_frames]
            y_input = conditional_dict["image"][:, :, current_start_frame_local - num_input_frames:current_start_frame_local + current_num_frames - num_input_frames]
            audio_input = conditional_dict["audio"][:,current_start_frame_global - num_input_frames:current_start_frame_global + current_num_frames - num_input_frames]
            block_conditional_dict = conditional_dict.copy()
            block_conditional_dict.update(image=y_input.clone(), audio=audio_input.clone())
            
            denoising_task = {
                "chunk_id": self.current_clock,
                "current_start_frame": current_start_frame_local,
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
            # Step 3.4: update the start and end frame indices
            current_start_frame_local += current_num_frames
            current_start_frame_global += current_num_frames
            # Step 3.5: Update img_lat every 3 blocks with latest generated latents
            # After denoising is complete, we can safely access output
            # Update after completing blocks 2, 5, 8, ... (i.e., when self.current_clock is 2, 5, 8, ...)
            # This ensures blocks 3, 6, 9, ... will use the updated img_lat
            # if reset, wait vae_events[self.current_clock]
            if (self.current_clock + 1) % 3 == 0 and self.current_clock >= 2:
                print(f"Updating img_lat after block {self.current_clock} with latest generated latents")
                # Extract the latest generated latents from output
                self.vae_events[self.current_clock].wait()
                # get last frame
                with self.result_lock:
                    last_block_frames = self.result_buffer[self.current_clock]["video"]
                    print(f"last_block_frames shape: {last_block_frames.shape}")
                last_frame = last_block_frames[:, -1:]  # (batch 1, frame 1, chanel 3, h 400, w 720)
                # save last frame as image
                for frame_idx in range(last_block_frames.shape[1]):
                    frame = last_block_frames[:, frame_idx]
                    frame_np = (frame.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype('uint8')
                    frame_pil = Image.fromarray(frame_np)
                    frame_pil.save(f"examples/chunk_id_{self.current_clock}_frame_{frame_idx}.png")
                import pdb; pdb.set_trace()
                # target shape (batch 1, channel 3, frame 1, h 400, w 720)
                last_frame = last_frame.permute(0, 2, 1, 3, 4)
                print(f"last_frame shape: {last_frame.shape}")
                
                # encode last frame to img_lat (following the logic in forward function)
                encode_start_time = time.time()
                
                # Ensure last_frame is in the correct range [-1, 1] for VAE encoding
                # last_frame should already be in [0, 1] or [-1, 1] from VAE decode
                # If it's in [0, 1], convert to [-1, 1]
                if last_frame.min() >= 0:
                    print(f"last_frame.min() >= 0, convert to [-1, 1]")
                    last_frame = last_frame * 2.0 - 1.0
                
                # VAE encode to get latent representation
                self.causal_pipe.vae.to("cuda:1")
                new_img_lat = self.causal_pipe.vae.encode(
                    videos=last_frame.to(dtype=self.dtype),
                    device="cuda:1"
                )  # Shape: (batch, 16, 1, h, w)
                print(f"Encoded new_img_lat shape: {new_img_lat.shape}")
                
                # Repeat to match num_frames dimension (same as original img_lat)
                num_frames_for_img_lat = img_lat.shape[2]  # Use original img_lat's num_frames
                new_img_lat = new_img_lat.repeat(1, 1, num_frames_for_img_lat, 1, 1)  # (batch, 16, num_frames, h, w)
                print(f"Repeated new_img_lat shape: {new_img_lat.shape}")
                
                # Create mask: first frame mask=0 (fixed), rest mask=1 (can be generated)
                msk = torch.zeros_like(new_img_lat)[:, :1]  # (batch, 1, num_frames, h, w)
                msk[:, :, 1:] = 1
                
                # Concatenate latents and mask to get final img_lat format
                img_lat = torch.cat([new_img_lat, msk], dim=1)  # (batch, 17, num_frames, h, w)
                print(f"Updated img_lat shape: {img_lat.shape}")
                
                # Update conditional_dict with new img_lat for all future frames
                total_num_frames = noise.shape[1]
                # if img_lat.shape[2] < total_num_frames:
                #     # Repeat img_lat to match total num_frames
                #     print(f"img_lat.shape[2] < total_num_frames, img_lat.shape[2]: {img_lat.shape[2]}, total_num_frames: {total_num_frames}")
                    # repeat_factor = total_num_frames // img_lat.shape[2] + 1
                    # img_lat_repeated = img_lat.repeat(1, 1, repeat_factor, 1, 1)
                    # img_lat_repeated = img_lat_repeated[:, :, :total_num_frames]
                # else:
                #     print(f"img_lat.shape[2] >= total_num_frames, img_lat.shape[2]: {img_lat.shape[2]}, total_num_frames: {total_num_frames}")  
                #     img_lat_repeated = img_lat[:, :, :total_num_frames]
                
                # img_lat = img_lat_repeated.clone()
                conditional_dict["image"] = img_lat.clone()
                print(f"img_lat shape: {img_lat.shape}")
                print(f"updated conditional_dict['image'] shape: {conditional_dict['image'].shape}")
                import pdb; pdb.set_trace()
                encode_time = time.time() - encode_start_time
                self._record_timing("img_lat_update_encode", encode_time, self.current_clock)
                # import pdb; pdb.set_trace()
                # reset
                self.causal_pipe.vae.clear_cache()
                self.causal_pipe.reset_caches(noise.device)
                current_start_frame_local = 0
                
            self.current_clock += 1
        
        # Wait for all decoding to complete
        self.vae_events[self.current_clock - 1].wait()
        
        # 打印性能统计报告
        self._print_timing_report()
        # 保存完整视频并合并音频
        self._save_complete_video_with_audio(num_blocks, audio_path, id)
  
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
            send_steps = {}
            
            for key, times in self.timing_stats.items():
                if key.startswith('forward_'):
                    forward_steps[key] = times
                elif 'denoising' in key or 'generator_forward' in key:
                    denoising_steps[key] = times
                elif 'vae' in key or 'decode' in key:
                    vae_steps[key] = times
                elif 'send' in key:
                    send_steps[key] = times
            
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

            if send_steps:
                print("\n📤 SEND PHASE TIMING:")
                print("-" * 50)
                total_send_time = 0
                for step, times in send_steps.items():
                    avg_time = np.mean(times)
                    total_time = np.sum(times)
                    total_send_time += total_time
                    print(f"  {step:<35}: {avg_time:>8.3f}s (total: {total_time:>8.3f}s, count: {len(times)})")
                print(f"  {'TOTAL SEND TIME':<35}: {total_send_time:>8.3f}s")
            
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
        id: str = None
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
            id: Unique identifier for this generation session
            
        Returns:
            Generated video tensor [batch_size, num_frames, channels, height, width]
        """
        # Calculate required number of frames from noise tensor
        batch_size, num_frames, num_channels, height, width = noise.shape

        # prepare image condition
        if image_path is not None:
            start_time = time.time()
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.causal_pipe.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = resize_pad(image, (h, w), select_size)
            print(f"image shape after resize: {image.shape}")
            
            # save original image (before normalization)
            # image shape should be (1, 3, h, w) in range [0, 1]
            image_to_save = image.squeeze(0).permute(1, 2, 0)  # (h, w, 3)
            image_np = (image_to_save.cpu().float().numpy() * 255).astype('uint8')
            image_pil = Image.fromarray(image_np)
            image_pil.save("examples/original_image.png")
            print(f"Saved original image with shape: {image_np.shape}")
            
            # Now normalize for model input
            image = image * 2.0 - 1.0  # Convert to [-1, 1]
            image = image[:, :, None]
            print(f"encode image shape: {image.shape}")
            # import pdb; pdb.set_trace()
            image_prep_time = time.time() - start_time
            self._record_timing("forward_image_preprocessing", image_prep_time)
            
            start_time = time.time()
            self.causal_pipe.vae.to("cuda:1")
            # Use num_frames from noise tensor instead of hardcoded 21
            # img_lat的作用机制：
            # 1. 作为条件信息：img_lat存储在conditional_dict["image"]中，传递给generator_forward作为图像条件
            # 2. 作为初始帧引导：在第一个block（current_start_frame==0）时，img_lat[:, :16, :1]用于初始化第一帧
            # 3. 格式：(batch, 17, num_frames, h, w)，其中前16个通道是VAE latent，最后1个通道是mask
            # 4. mask规则：第一帧mask=0（固定），后续帧mask=1（可生成）
            # 5. 在run_pipeline中，每隔3个block会用最新生成的latents更新img_lat，实现动态引导
            img_lat = self.causal_pipe.vae.encode(videos=image.to(dtype=self.dtype),device="cuda:1").repeat(1,1,num_frames,1,1)
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
        
            input_values = np.squeeze(
                    self.causal_pipe.wav_feature_extractor(audio, sampling_rate=16000).input_values # TODO: update sample rate to
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
        self.causal_pipe.vae.clear_cache()
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
            audio_path=audio_path,
            id=id
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
                # import pdb; pdb.set_trace()
                # Step 3.1: Spatial denoising loop
                # import pdb; pdb.set_trace()
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
                
                # 记录整个denoising任务的总时间
                total_denoising_time = time.time() - denoising_start_time
                self._record_timing("denoising_total", total_denoising_time, chunk_id)

                latents = output[:, current_start_frame:current_start_frame + current_num_frames]
                print(f"chunk_id: {chunk_id}, latents: {latents.shape}")
                task.update({
                    "latents": latents.clone()
                })
                self.vae_queue.put(task)
                with self.latents_lock:
                    self.latents_buffer[chunk_id] = latents.clone()
                print(f"Causal denoising inference completed for block {chunk_id}")
                # Trigger denoising event - use chunk_id instead of self.current_clock to avoid race condition
                self.denoising_events[chunk_id].set()
                
                # Create VAE event for this block
                self.vae_events.append(threading.Event())
                
                self.denoising_queue.task_done()

            except queue.Empty:
                # Check if we should stop
                if self.stop_workers.is_set():
                    print("Denoising worker received stop signal")
                    break
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
                video = self.causal_pipe.vae.decode(latents, device="cuda:1", tiled=False, tile_size=(30, 52), tile_stride=(15, 26)).permute(0, 2, 1, 3, 4)
                print(f"video shape: {video.shape}")
                decode_time = time.time() - decode_start_time
                self._record_timing("vae_decode", decode_time, chunk_id)
                
                # 后处理阶段
                postprocess_start_time = time.time()
                # video = video[:, :, 1:].permute(0, 2, 1, 3, 4)
                # video = video[:, :, 1:]
                # Ensure video is in float32 for compatibility with numpy conversion
                # video = video[:, 1:]
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
                # trigger vae event - use chunk_id instead of self.current_clock - 1 to avoid race condition
                self.vae_events[chunk_id].set()
                self.vae_queue.task_done()
            except queue.Empty:
                # Check if we should stop
                if self.stop_workers.is_set():
                    print("VAE worker received stop signal")
                    break
                pass
            except Exception as e:
                print(f"Error in video formatting for block {chunk_id}: {e}")
                traceback.print_exc()
                self.vae_queue.task_done()

    def _save_complete_video_with_audio(self, num_blocks: int, audio_path: Optional[str], id: Optional[str]):
        """
        保存完整视频并与音频合并
        
        Args:
            num_blocks: 总的块数
            audio_path: 音频文件路径（可选）
            id: 唯一标识符（用于文件命名）
        """
        print("\n" + "="*80)
        print("SAVING COMPLETE VIDEO WITH AUDIO")
        print("="*80)
        
        save_start_time = time.time()
        
        # 步骤1: 从result_buffer收集所有视频块
        print(f"收集 {num_blocks} 个视频块...")
        all_videos = []
        for chunk_id in range(num_blocks):
            if chunk_id not in self.result_buffer:
                print(f"警告: chunk {chunk_id} 不在 result_buffer 中")
                continue
            
            video_chunk = self.result_buffer[chunk_id]["video"]
            all_videos.append(video_chunk)
            print(f"  Chunk {chunk_id}: shape {video_chunk.shape}")
        
        if len(all_videos) == 0:
            print("错误: 没有找到任何视频块!")
            return
        
        # 步骤2: 拼接所有视频块
        print(f"拼接 {len(all_videos)} 个视频块...")
        complete_video = torch.cat(all_videos, dim=1)  # 在时间维度拼接
        print(f"完整视频 shape: {complete_video.shape}")
        
        # 步骤3: 转换为numpy数组
        print("将视频转换为 numpy 数组...")
        # Shape: (batch, time, channels, height, width) -> (time, height, width, channels)
        video_np = complete_video.squeeze(0).permute(0, 2, 3, 1).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        print(f"视频 numpy shape: {video_np.shape}")
        
        # 步骤4: 创建输出目录
        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        
        # 步骤5: 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_str = f"_{id}" if id else ""
        video_no_audio_path = os.path.join(output_dir, f"video_no_audio{id_str}_{timestamp}.mp4")
        video_with_audio_path = os.path.join(output_dir, f"video_with_audio{id_str}_{timestamp}.mp4")
        
        # 步骤6: 保存无音频视频
        print(f"保存无音频视频到: {video_no_audio_path}")
        fps = getattr(self.args, 'fps', 16)
        height, width = video_np.shape[1:3]
        
        # 使用cv2保存视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_no_audio_path, fourcc, fps, (width, height))
        
        for frame_idx in range(video_np.shape[0]):
            frame_bgr = cv2.cvtColor(video_np[frame_idx], cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        video_save_time = time.time() - save_start_time
        print(f"✓ 无音频视频保存成功 (用时 {video_save_time:.2f}s)")
        
        # 步骤7: 如果有音频路径，使用ffmpeg合并音频
        if audio_path is not None and os.path.exists(audio_path):
            print(f"\n使用 ffmpeg 合并音频: {audio_path}")
            merge_start_time = time.time()
            
            try:
                # 构建ffmpeg命令
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', video_no_audio_path,  # 输入视频
                    '-i', audio_path,            # 输入音频
                    '-map', '0:v',               # 选择第一个输入的视频流
                    '-map', '1:a',               # 选择第二个输入的音频流
                    '-c:v', 'libx264',           # 视频编码器
                    '-preset', 'medium',         # 编码预设
                    '-crf', '23',                # 质量参数
                    '-c:a', 'aac',               # 音频编码器
                    '-b:a', '192k',              # 音频比特率
                    '-shortest',                 # 以最短的流为准
                    '-y',                        # 覆盖已存在的文件
                    video_with_audio_path
                ]
                
                print(f"执行命令: {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分钟超时
                )
                
                if result.returncode == 0:
                    merge_time = time.time() - merge_start_time
                    print(f"✓ 音画合并视频保存成功: {video_with_audio_path}")
                    print(f"  音频合并用时: {merge_time:.2f}s")
                else:
                    print(f"✗ ffmpeg 错误 (返回码: {result.returncode})")
                    print(f"  stderr: {result.stderr}")
                    print(f"  无音频视频仍可用: {video_no_audio_path}")
                    
            except subprocess.TimeoutExpired:
                print("✗ ffmpeg 超时 (超过5分钟)")
                print(f"  无音频视频仍可用: {video_no_audio_path}")
            except FileNotFoundError:
                print("✗ 未找到 ffmpeg 命令，请确保已安装 ffmpeg")
                print(f"  无音频视频仍可用: {video_no_audio_path}")
            except Exception as e:
                print(f"✗ 音频合并异常: {e}")
                traceback.print_exc()
                print(f"  无音频视频仍可用: {video_no_audio_path}")
        else:
            if audio_path is None:
                print("\n未提供音频路径，跳过音频合并")
            else:
                print(f"\n音频文件不存在: {audio_path}，跳过音频合并")
        
        total_save_time = time.time() - save_start_time
        print(f"\n总保存时间: {total_save_time:.2f}s")
        print("="*80 + "\n")
    
    def process_multiturn_conversation(
        self,
        conversation_data: Dict[str, Any],
        image_path: str,
        id: Optional[str] = None
    ):
        """
        处理多轮对话，为每一轮生成视频（包含音频）
        
        Args:
            conversation_data: 包含conversation列表的字典，格式如：
                {
                    "test_001_multiturn": {
                        "conversation": [
                            {"role": "user", "content": "...", "audio_path": "..."},
                            {"role": "assistant", "content": "", "video_path": "..."},
                            ...
                        ]
                    }
                }
            image_path: 参考图像路径
            id: 唯一标识符（可选）
        
        Returns:
            Dict: 包含每轮生成结果的字典
        """
        print("\n" + "="*80)
        print("MULTI-TURN CONVERSATION PROCESSING")
        print("="*80)
        
        # 提取conversation列表
        # 支持两种格式：直接传conversation列表或嵌套在字典中
        if isinstance(conversation_data, dict):
            if "conversation" in conversation_data:
                conversation = conversation_data["conversation"]
                if id is None:
                    id = list(conversation_data.keys())[0] if len(conversation_data) == 1 else "default"
            else:
                # 假设是嵌套格式，取第一个key
                first_key = list(conversation_data.keys())[0]
                conversation = conversation_data[first_key]["conversation"]
                if id is None:
                    id = first_key
        else:
            conversation = conversation_data
            if id is None:
                id = "default"
        
        print(f"ID: {id}")
        print(f"Total turns in conversation: {len(conversation)}")
        print(f"Reference image: {image_path}")
        
        # 初始化Qwen-Omni talker
        qwen_talker = QwenOmniTalker()
        
        # 存储每轮的结果
        results = {
            "id": id,
            "turns": []
        }
        
        # 遍历对话，处理每个user-assistant对
        turn_number = 0
        for i in range(0, len(conversation), 2):
            if i + 1 >= len(conversation):
                print(f"Warning: Incomplete turn at index {i}, skipping")
                break
            
            user_turn = conversation[i]
            assistant_turn = conversation[i + 1]
            
            # 验证角色
            if user_turn.get("role") != "user" or assistant_turn.get("role") != "assistant":
                print(f"Warning: Invalid roles at index {i}, skipping")
                continue
            
            turn_number += 1
            print(f"\n{'='*60}")
            print(f"Processing Turn {turn_number}")
            print(f"{'='*60}")
            
            # 获取用户输入
            user_content = user_turn.get("content", "")
            user_audio_path = user_turn.get("audio_path")
            assistant_video_path = assistant_turn.get("video_path")
            
            if not user_audio_path:
                print(f"Warning: No audio_path in user turn {turn_number}, skipping")
                continue
            
            if not assistant_video_path:
                print(f"Warning: No video_path in assistant turn {turn_number}, skipping")
                continue
            
            print(f"User content: {user_content}")
            print(f"User audio: {user_audio_path}")
            print(f"Target video path: {assistant_video_path}")
            
            try:
                # Step 1: 使用Qwen-Omni处理音频，生成回复音频和文本
                print(f"\n[Turn {turn_number}] Step 1: Processing audio with Qwen-Omni...")
                reply_audio_path, reply_text = qwen_talker.process_audio_conversation(
                    audio_path=user_audio_path,
                    session_id=id,
                    prompt=user_content if user_content else "Please respond to this audio."
                )
                
                if reply_audio_path is None or reply_text is None:
                    print(f"Error: Failed to get reply from Qwen-Omni for turn {turn_number}")
                    results["turns"].append({
                        "turn": turn_number,
                        "status": "failed",
                        "error": "Qwen-Omni processing failed"
                    })
                    continue
                
                print(f"Reply audio saved: {reply_audio_path}")
                print(f"Reply text: {reply_text}")
                
                # Step 2: 使用reply_audio生成视频
                print(f"\n[Turn {turn_number}] Step 2: Generating video with audio...")
                temp_video_path = self._generate_video_for_turn(
                    image_path=image_path,
                    audio_path=reply_audio_path,
                    text_prompt=reply_text,
                    turn_number=turn_number,
                    id=id
                )
                
                if temp_video_path is None:
                    print(f"Error: Failed to generate video for turn {turn_number}")
                    results["turns"].append({
                        "turn": turn_number,
                        "status": "failed",
                        "error": "Video generation failed"
                    })
                    continue
                
                # Step 3: 移动视频到目标路径
                print(f"\n[Turn {turn_number}] Step 3: Moving video to target path...")
                os.makedirs(os.path.dirname(assistant_video_path), exist_ok=True)
                
                # 如果temp_video_path和assistant_video_path不同，则复制/移动文件
                if os.path.abspath(temp_video_path) != os.path.abspath(assistant_video_path):
                    import shutil
                    shutil.move(temp_video_path, assistant_video_path)
                    print(f"Video moved to: {assistant_video_path}")
                else:
                    print(f"Video already at target path: {assistant_video_path}")
                
                # 记录成功结果
                results["turns"].append({
                    "turn": turn_number,
                    "status": "success",
                    "user_audio": user_audio_path,
                    "reply_audio": reply_audio_path,
                    "reply_text": reply_text,
                    "video_path": assistant_video_path
                })
                
                print(f"\n✓ Turn {turn_number} completed successfully!")
                
            except Exception as e:
                print(f"\n✗ Error processing turn {turn_number}: {e}")
                traceback.print_exc()
                results["turns"].append({
                    "turn": turn_number,
                    "status": "failed",
                    "error": str(e)
                })
        
        # 打印总结
        print("\n" + "="*80)
        print("MULTI-TURN CONVERSATION SUMMARY")
        print("="*80)
        successful_turns = sum(1 for t in results["turns"] if t["status"] == "success")
        print(f"Total turns processed: {len(results['turns'])}")
        print(f"Successful: {successful_turns}")
        print(f"Failed: {len(results['turns']) - successful_turns}")
        print("="*80 + "\n")
        
        return results
    
    def _generate_video_for_turn(
        self,
        image_path: str,
        audio_path: str,
        text_prompt: str,
        turn_number: int,
        id: str
    ) -> Optional[str]:
        """
        为单个对话轮次生成视频（内部辅助方法）
        
        Args:
            image_path: 参考图像路径
            audio_path: 音频文件路径
            text_prompt: 文本提示
            turn_number: 当前轮次编号
            id: 唯一标识符
        
        Returns:
            str: 生成的视频路径（包含音频），如果失败则返回None
        """
        try:
            # 准备噪声输入
            batch_size = 1
            num_frames = self.args.num_frame_per_block * 3  # 可根据需要调整
            num_channels = 16
            height = self.args.latent_height
            width = self.args.latent_width
            
            noise = torch.randn(
                [batch_size, num_frames, num_channels, height, width],
                device=self.device,
                dtype=self.dtype
            )
            
            # 调用forward方法生成视频
            print(f"Generating video with {num_frames} frames...")
            self.forward(
                noise=noise,
                text_prompts=text_prompt,
                image_path=image_path,
                audio_path=audio_path,
                initial_latent=None,
                return_latents=False,
                id=f"{id}_turn_{turn_number}"
            )
            
            # 视频已通过_save_complete_video_with_audio保存
            # 查找最新生成的视频文件
            output_dir = "output_videos"
            id_str = f"_{id}_turn_{turn_number}"
            
            # 查找匹配的视频文件
            video_files = [
                f for f in os.listdir(output_dir)
                if f.startswith("video_with_audio") and id_str in f
            ]
            
            if video_files:
                # 返回最新的文件
                video_files.sort(reverse=True)
                video_path = os.path.join(output_dir, video_files[0])
                print(f"Generated video: {video_path}")
                return video_path
            else:
                # 如果没有找到with_audio版本，查找no_audio版本
                video_files_no_audio = [
                    f for f in os.listdir(output_dir)
                    if f.startswith("video_no_audio") and id_str in f
                ]
                if video_files_no_audio:
                    video_files_no_audio.sort(reverse=True)
                    video_path = os.path.join(output_dir, video_files_no_audio[0])
                    print(f"Generated video (no audio merged): {video_path}")
                    return video_path
                else:
                    print("Error: Could not find generated video file")
                    return None
            
        except Exception as e:
            print(f"Error generating video: {e}")
            traceback.print_exc()
            return None


def batch_process_multiturn_conversations_from_json(
    pipeline,
    json_file_path: str,
    image_path: str
) -> Dict[str, Any]:
    """
    批量处理JSON文件中的所有多轮对话数据
    
    参考pipelined_websocket_streaming_server.py中的QwenOmniTalker实现，
    为每条多轮对话数据生成对应的视频回复。
    
    Args:
        pipeline: PipelinedEvalPipeline实例
        json_file_path: JSON文件路径，包含多条多轮对话数据
        image_path: 参考图像路径（所有对话共用）
    
    Returns:
        Dict: 包含所有对话的处理结果
    
    JSON格式示例：
    {
        "conversation_001": {
            "conversation": [
                {"role": "user", "content": "...", "audio_path": "..."},
                {"role": "assistant", "content": "", "video_path": "..."},
                ...
            ]
        },
        "conversation_002": {
            "conversation": [...]
        },
        ...
    }
    """
    print("\n" + "="*80)
    print(f"BATCH PROCESSING MULTI-TURN CONVERSATIONS FROM: {json_file_path}")
    print("="*80 + "\n")
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_conversations = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        traceback.print_exc()
        return {"error": str(e), "conversations": []}
    
    # 验证图像路径
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return {"error": f"Image not found: {image_path}", "conversations": []}
    
    print(f"Image path: {image_path}")
    print(f"Total conversations found: {len(all_conversations)}\n")
    
    # 存储所有对话的处理结果
    batch_results = {
        "json_file": json_file_path,
        "image_path": image_path,
        "total_conversations": len(all_conversations),
        "conversations": []
    }
    
    # 遍历每条对话数据
    for conv_idx, (conv_id, conv_data) in enumerate(all_conversations.items(), 1):
        print(f"\n{'='*80}")
        print(f"Processing Conversation {conv_idx}/{len(all_conversations)}: {conv_id}")
        print(f"{'='*80}\n")
        
        try:
            # 调用process_multiturn_conversation处理单条对话
            conv_result = pipeline.process_multiturn_conversation(
                conversation_data={conv_id: conv_data},
                image_path=image_path,
                id=conv_id
            )
            
            # 记录结果
            batch_results["conversations"].append({
                "id": conv_id,
                "status": "success",
                "result": conv_result
            })
            
            print(f"\n✓ Conversation {conv_id} processed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error processing conversation {conv_id}: {e}")
            traceback.print_exc()
            
            batch_results["conversations"].append({
                "id": conv_id,
                "status": "failed",
                "error": str(e)
            })
    
    # 打印总结
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    successful = sum(1 for c in batch_results["conversations"] if c["status"] == "success")
    failed = len(batch_results["conversations"]) - successful
    print(f"Total conversations: {len(batch_results['conversations'])}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*80 + "\n")
    
    # 保存结果到JSON
    output_json_path = json_file_path.replace('.json', '_batch_results.json')
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        print(f"Batch results saved to: {output_json_path}\n")
    except Exception as e:
        print(f"Warning: Failed to save batch results: {e}\n")
    
    return batch_results


if __name__ == "__main__":
    """
    使用示例：
    
    python scripts/pipelined_eval_gen.py \\
        --json_path conversations.json \\
        --image_path reference.jpg \\
        --config configs/causal_inference.yaml
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process multi-turn conversations')
    parser.add_argument('--json_path', type=str, help='Path to JSON file containing conversations')
    parser.add_argument('--image_path', type=str, help='Path to reference image')
    parser.add_argument('--config', type=str, default='configs/causal_inference.yaml', help='Path to config file')
    
    cmd_args = parser.parse_args()
    
    if cmd_args.json_path and cmd_args.image_path:
        # 批量处理模式
        print("\n" + "="*80)
        print("BATCH PROCESSING MODE")
        print("="*80 + "\n")
        
        # 解析配置
        args = parse_args()
        
        # 初始化pipeline
        print("Initializing pipeline...")
        pipeline = PipelinedEvalPipeline(args)
        print("Pipeline initialized successfully!\n")
        
        # 批量处理所有对话
        results = batch_process_multiturn_conversations_from_json(
            pipeline=pipeline,
            json_file_path=cmd_args.json_path,
            image_path=cmd_args.image_path
        )
        
        print("\n" + "="*80)
        print("BATCH PROCESSING COMPLETE!")
        print("="*80 + "\n")
        
    else:
        # 显示使用说明
        print("\n" + "="*80)
        print("PIPELINED EVAL GEN - Multi-Turn Conversation Batch Processing")
        print("="*80 + "\n")
        print("This script supports batch processing of multi-turn conversations from JSON files.")
        print("\nUsage:")
        print("  python scripts/pipelined_eval_gen.py \\")
        print("    --json_path conversations.json \\")
        print("    --image_path reference.jpg \\")
        print("    --config configs/causal_inference.yaml")
        print("\nJSON file format:")
        print("""
{
  "conversation_001": {
    "conversation": [
      {"role": "user", "content": "Hello", "audio_path": "audio1.wav"},
      {"role": "assistant", "content": "", "video_path": "video1.mp4"},
      {"role": "user", "content": "How are you?", "audio_path": "audio2.wav"},
      {"role": "assistant", "content": "", "video_path": "video2.mp4"}
    ]
  },
  "conversation_002": {
    "conversation": [...]
  }
}
""")
        print("="*80 + "\n")
    
