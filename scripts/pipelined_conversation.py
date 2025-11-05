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
from openai import OpenAI

class QwenOmniTalker:
    """Qwen-Omni语音对话处理器"""
    
    def __init__(self, api_key="sk-63ad221681734d339b8171797204f105", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.system_message = {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        }
        
    def process_audio_conversation_stream(self, audio_path, session_id=None, prompt="Analyze this audio and respond naturally.", 
                                          fps=16, frames_per_block=12, sample_rate=24000):
        """
        流式处理音频对话，以block为单位yield音频数据

        Args:
            audio_path: 输入音频文件路径
            session_id: 会话ID，用于生成唯一的输出文件名
            prompt: 文本提示词，默认为分析音频内容
            fps: 视频帧率，默认16
            frames_per_block: 每个block的帧数，默认12
            sample_rate: 音频采样率，默认24000

        Yields:
            tuple: (block_audio_array, block_text, is_final)
                - block_audio_array: numpy数组，shape为(samples,)，dtype为int16
                - block_text: 该block对应的文本片段
                - is_final: 是否为最后一个block
        """
        try:
            # 计算每个block的采样点数
            block_duration = frames_per_block / fps  # 0.75秒
            samples_per_block = int(block_duration * sample_rate)  # 18000个采样点
            bytes_per_block = samples_per_block * 2  # 36000字节（int16）
            
            # 读取音频文件并编码为base64
            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            # 构建消息 - 使用input_audio格式发送音频输入
            messages = [
                self.system_message,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio_base64}", "format": "wav"}},
                    ],
                },
            ]
            
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
            
            # 流式收集和输出响应
            text_parts = []
            audio_buffer = b""  # 字节缓冲区
            block_idx = 0
            
            for chunk in completion:
                if chunk.choices:
                    if hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio:
                        try:
                            if "data" in chunk.choices[0].delta.audio:
                                # 累积音频数据
                                chunk_audio_base64 = chunk.choices[0].delta.audio["data"]
                                chunk_audio_bytes = base64.b64decode(chunk_audio_base64)
                                audio_buffer += chunk_audio_bytes
                                
                                # 当累积足够一个block时，yield出去
                                while len(audio_buffer) >= bytes_per_block:
                                    block_bytes = audio_buffer[:bytes_per_block]
                                    audio_buffer = audio_buffer[bytes_per_block:]
                                    
                                    block_audio_array = np.frombuffer(block_bytes, dtype=np.int16)
                                    block_text = "".join(text_parts) if text_parts else ""
                                    
                                    print(f"Yielding block {block_idx}: {len(block_audio_array)} samples, text_len={len(block_text)}")
                                    yield (block_audio_array, block_text, False)
                                    block_idx += 1
                                    text_parts = []  # 清空已输出的文本
                                    
                            elif "transcript" in chunk.choices[0].delta.audio:
                                text_parts.append(chunk.choices[0].delta.audio["transcript"])
                        except Exception as e:
                            print(f"Error processing audio chunk: {e}")
                    elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        text_parts.append(chunk.choices[0].delta.content)
                else:
                    if hasattr(chunk, 'usage') and chunk.usage:
                        print(f"Usage: {chunk.usage}")
            
            # 处理剩余的音频数据（最后一个不完整的block）
            if len(audio_buffer) > 0:
                # 填充到完整block（用静音填充）
                padding_size = bytes_per_block - len(audio_buffer)
                if padding_size > 0:
                    audio_buffer += b'\x00' * padding_size
                
                block_audio_array = np.frombuffer(audio_buffer[:bytes_per_block], dtype=np.int16)
                block_text = "".join(text_parts) if text_parts else ""
                
                print(f"Yielding final block {block_idx}: {len(block_audio_array)} samples (padded), text_len={len(block_text)}")
                yield (block_audio_array, block_text, True)
            else:
                # 如果没有剩余数据，发送空的final标记
                if block_idx > 0:
                    print(f"All blocks completed, total {block_idx} blocks")
                else:
                    print("Warning: No audio data received from Qwen-Omni")
                    yield (np.array([], dtype=np.int16), "".join(text_parts), True)
                
        except Exception as e:
            print(f"Error in Qwen-Omni conversation stream: {e}")
            traceback.print_exc()
            yield (None, None, True)

class BufferedGenerator:
    """
    Buffered Generator for audio conversation stream
    启动时立即开始在后台线程中生成数据，提供真正的流式体验
    """
    def __init__(self, generator_func, start_immediately=True):
        self.generator_func = generator_func
        self.queue = queue.Queue()  # 任务队列
        self.stop_event = threading.Event()
        self.producer_thread = None
        
        # 立即启动生产者线程，提前开始生成
        if start_immediately:
            self._start_producer()
        
    def _start_producer(self):
        """启动生产者线程"""
        if self.producer_thread is None or not self.producer_thread.is_alive():
            self.producer_thread = threading.Thread(target=self._producer, daemon=True)
            self.producer_thread.start()
            print("BufferedGenerator: Producer thread started")
        
    def _producer(self):
        """生产者线程：在后台生成数据并放入队列"""
        try:
            for item in self.generator_func():
                if self.stop_event.is_set():
                    print("BufferedGenerator: Stop event set, exiting producer")
                    break
                self.queue.put(item)
            self.queue.put(None)  # 结束标记
            print("BufferedGenerator: Producer finished")
        except Exception as e:
            print(f"BufferedGenerator: Producer error: {e}")
            traceback.print_exc()
            self.queue.put(None)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # 如果线程还没启动，启动它（兜底保护）
        if self.producer_thread is None or not self.producer_thread.is_alive():
            self._start_producer()
        
        # 从队列中获取数据（如果数据已生成则立即返回，否则阻塞等待）
        item = self.queue.get()
        if item is None:
            if self.producer_thread.is_alive():
                self.producer_thread.join(timeout=1.0)
            raise StopIteration
        return item

    def stop(self):
        self.stop_event.set()
        if self.producer_thread is not None and self.producer_thread.is_alive():
            self.producer_thread.join()

class PipelinedConversation(nn.Module):
    """
    流水线化的Conversation Pipeline实现
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
        
        # talker
        self.talker: QwenOmniTalker = QwenOmniTalker()
        
        # 流水线相关参数
        self.audio_gen_buffer = None
        self.accumulated_audio = np.array([], dtype=np.int16)
        self.accumulated_audio_lock = threading.Lock()
        self.denoising_queue = queue.Queue()  # denoising任务队列
        self.vae_queue = queue.Queue()  # decoding任务队列
        self.result_buffer = {}  # 保持字典结构，用于按chunk_id排序
        # 多线程控制
        self.denoising_thread = None
        self.vae_thread = None
        self.result_lock = threading.Lock()
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
    
    @torch.no_grad()
    def encode_audio_block(self, audio_block, sr=16000, block_idx=0, 
                          is_first_block=False, is_last_block=False,
                          fps=16, frames_per_block=12, h=1):
        """
        编码单个音频block，输出对应的音频特征
        
        复用 causal_inference_audio_stream.py 的逻辑：
        - 第0块：编码9帧，复制第0帧3次，输出12帧
        - 块1-6：编码12+2帧（左右各1帧overlap），discard后输出12帧
        
        Args:
            audio_block: numpy数组 [audio_samples]，int16格式，采样率sr
            sr: 音频采样率，默认24000（Qwen-Omni输出）
            block_idx: 当前块索引（0-6）
            is_first_block: 是否是第一个块
            is_last_block: 是否是最后一个块
            fps: 视频帧率，默认16
            frames_per_block: 每块输出的帧数，默认12
            h: overlap区域大小，默认1帧
            
        Returns:
            audio_emb: torch.Tensor, shape=[1, frames_per_block, 10752]
        """
        frame_duration = 1.0 / fps  # 每帧时长（秒）
        
        # 将音频从int16转换为float32（归一化到[-1, 1]）
        if audio_block.dtype == np.int16:
            audio = audio_block.astype(np.float32) / 32768.0
        else:
            audio = audio_block.astype(np.float32)
        
        # 如果采样率不是16000，需要重采样
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        print(f"encode_audio_block - Block {block_idx} (accumulated mode): audio length {len(audio)} samples")
        
        # 计算当前块对应的帧范围
        if is_first_block:
            # 第0块特殊处理：只编码帧0-8（9帧）
            start_frame = 0
            end_frame = 9
            actual_frames = 9
            
            # 添加上下文帧（第0块只有右侧上下文）
            start_frame_with_context = 0
            end_frame_with_context = end_frame + h
            target_seq_len = actual_frames + h  # 9 + 1 = 10
        else:
            # 后续块正常处理
            start_frame = 9 + (block_idx - 1) * frames_per_block
            end_frame = start_frame + frames_per_block
            actual_frames = frames_per_block
            
            # 添加上下文帧
            start_frame_with_context = max(0, start_frame - h)
            if is_last_block:
                end_frame_with_context = end_frame
                target_seq_len = actual_frames + h  # 12 + 1 = 13
            else:
                end_frame_with_context = end_frame + h
                target_seq_len = actual_frames + 2 * h  # 12 + 2 = 14
        
        # 计算对应的音频范围（采样点）
        start_sample = int(start_frame_with_context * frame_duration * sr)
        end_sample = int(end_frame_with_context * frame_duration * sr)
        
        # 对于累积音频，确保不超出音频长度
        audio_length = len(audio)
        start_sample = min(start_sample, audio_length)
        end_sample = min(end_sample, audio_length)
        
        # 提取音频片段
        audio_segment = audio[start_sample:end_sample]
        
        # 如果音频片段太短，进行填充
        min_audio_length = int(0.1 * sr)  # 至少0.1秒
        if len(audio_segment) < min_audio_length:
            audio_segment = np.pad(audio_segment, (0, min_audio_length - len(audio_segment)),
                                   mode='constant', constant_values=0)
            print(f"encode_audio_block - Block {block_idx}: padded audio segment to {len(audio_segment)} samples")
        # 编码音频片段
        input_values = np.squeeze(
            self.causal_pipe.wav_feature_extractor(audio_segment, sampling_rate=sr).input_values
        )
        input_values = torch.from_numpy(input_values).float().to(device=self.device)
        input_values = input_values.unsqueeze(0)
        
        # 使用audio_encoder编码
        with torch.no_grad():
            self.causal_pipe.audio_encoder.to(self.device)
            hidden_states = self.causal_pipe.audio_encoder(
                input_values, 
                seq_len=target_seq_len, 
                output_hidden_states=True
            )
            audio_embeddings = hidden_states.last_hidden_state
            for mid_hidden_states in hidden_states.hidden_states:
                audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
        
        # audio_embeddings shape: (1, target_seq_len, 10752)
        # Overlap-discard: 根据边界情况选择性丢弃
        if h > 0:
            if is_first_block:
                # 第0块：只丢弃右侧h帧
                audio_embeddings_trimmed = audio_embeddings[:, :-h, :]  # (1, 9, 10752)
            elif is_last_block:
                # 最后一块：只丢弃左侧h帧
                audio_embeddings_trimmed = audio_embeddings[:, h:, :]  # (1, 12, 10752)
            else:
                # 中间块：丢弃前后各h帧
                audio_embeddings_trimmed = audio_embeddings[:, h:-h, :]  # (1, 12, 10752)
        else:
            audio_embeddings_trimmed = audio_embeddings
        
        # 第0块特殊处理：将第0帧复制3次，使其输出12帧
        if is_first_block:
            # audio_embeddings_trimmed shape: (1, 9, 10752)
            # 复制第0帧3次: [feat_0, feat_0, feat_0, feat_0, feat_1, ..., feat_8]
            first_frame = audio_embeddings_trimmed[:, 0:1, :]  # (1, 1, 10752)
            repeated_first_frame = first_frame.repeat(1, 3, 1)  # (1, 3, 10752)
            # 拼接：3个重复 + 原始9帧 = 12帧
            audio_embeddings_trimmed = torch.cat([repeated_first_frame, audio_embeddings_trimmed], dim=1)
            print(f"encode_audio_block - Block {block_idx} (first): output shape {audio_embeddings_trimmed.shape}")
        else:
            print(f"encode_audio_block - Block {block_idx}: output shape {audio_embeddings_trimmed.shape}")
        
        # 转换为DiT所需的格式：[1, 10752, 12, 1, 1]
        # 注意：12帧已经能被patch_size=4整除，不需要额外复制
        audio_emb = audio_embeddings_trimmed.permute(0, 2, 1)[:, :, :, None, None]
        # shape: [1, 10752, 12, 1, 1]
        
        # 投影到DiT特征空间
        audio_emb = self.causal_pipe.generator.audio_proj(audio_emb.to(self.dtype))
        audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.causal_pipe.generator.audio_cond_projs], 0)
        
        print(f"encode_audio_block - Block {block_idx}: final audio_emb shape {audio_emb.shape}")
        
        return audio_emb
    
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
            audio_path: str,    # quesiton audio
            streaming_callback,
            session_id
        ):
        
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
        with self.accumulated_audio_lock:
            self.accumulated_audio = np.array([], dtype=np.int16)
            print(f"Reset accumulated audio, length: {len(self.accumulated_audio)}")
        
        # 清空性能统计
        with self.timing_lock:
            self.timing_stats.clear()
        
        # run async threads
        # audio gen
        def audio_gen_func():
            return self.talker.process_audio_conversation_stream(audio_path, session_id)
        self.audio_gen_buffer = BufferedGenerator(audio_gen_func)
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
        executor = ThreadPoolExecutor(max_workers=2)
        total_frames_generated = 0
        send_future = None
        
        # Process video blocks (limited by num_blocks)
        for video_block_idx in range(num_blocks):
            current_num_frames = all_num_frames[video_block_idx]
            print(f"===================== Current clock: {self.current_clock} ======================")
            print(f"Processing frame {current_start_frame - num_input_frames} to {current_start_frame + current_num_frames - num_input_frames}.")
            
            # get audio block
            try:
                audio_block, text, is_final = next(self.audio_gen_buffer)
                print(f"Received audio block {video_block_idx}: is_final={is_final}, samples={len(audio_block) if audio_block is not None else 0}")
                
                # update accumulated audio
                if audio_block is not None and len(audio_block) > 0:
                    with self.accumulated_audio_lock:
                        self.accumulated_audio = np.concatenate([self.accumulated_audio, audio_block])
                        print(f"Accumulated audio length: {len(self.accumulated_audio)} samples")
                
                audio_emb = self.encode_audio_block(audio_block, sr=24000, block_idx=self.current_clock, is_first_block=self.current_clock == 0, is_last_block=self.current_clock == num_blocks - 1, fps=16, frames_per_block=12, h=1)
            except StopIteration:
                print(f"Audio generation completed at block {video_block_idx}")
                break
            noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            y_input = conditional_dict["image"][:, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            audio_input = audio_emb.clone()
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
                "output": output.clone(),
                "audio_segment": audio_block
            }
            self.denoising_queue.put(denoising_task)
            # wait denoising clock and decoding clock - 1
            self.denoising_events.append(threading.Event())
            self.denoising_events[self.current_clock].wait()
            if self.current_clock >= 1:
                self.vae_events[self.current_clock - 1].wait()
            # send current chunk, TODO: update interface
            if send_future is not None:
                print("Waiting previous send task...")
                total_frames_generated = send_future.result()
                send_future = None
            send_future = executor.submit(
                self._send_chunk_frames,
                self.current_clock - 1,
                streaming_callback,
                session_id,
                total_frames_generated,
                num_blocks
            )
            # total_frames_generated = self._send_chunk_frames(self.current_clock - 1, streaming_callback, session_id, total_frames_generated, num_blocks)
            # import pdb; pdb.set_trace()
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            self.current_clock += 1
        # Collect remaining audio blocks (if any)
        print("Collecting remaining audio blocks...")
        remaining_audio_blocks = 0
        try:
            while True:
                audio_block, text, is_final = next(self.audio_gen_buffer)
                if audio_block is not None and len(audio_block) > 0:
                    with self.accumulated_audio_lock:
                        self.accumulated_audio = np.concatenate([self.accumulated_audio, audio_block])
                        remaining_audio_blocks += 1
                        print(f"Collected remaining audio block {remaining_audio_blocks}: {len(audio_block)} samples")
                if is_final:
                    print(f"Finished collecting {remaining_audio_blocks} remaining audio blocks")
                    break
        except StopIteration:
            print(f"All audio blocks collected (total remaining: {remaining_audio_blocks})")
        
        print(f"Total accumulated audio: {len(self.accumulated_audio)} samples")
        
        # TODO: wait decoding the last chunk?
        self.vae_events[self.current_clock - 1].wait()
        if send_future is not None:
            print("Waiting previous send task...")
            total_frames_generated = send_future.result()
            send_future = None
        # send the last chunk, TODO: update interface
        total_frames_generated = self._send_chunk_frames(self.current_clock - 1, streaming_callback, session_id, total_frames_generated, num_blocks)
        
        # 打印性能统计报告
        self._print_timing_report()
        # 保存完整视频并合并音频
        self._save_complete_video_with_audio(num_blocks, audio_path, session_id)
  
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
        audio_path: Optional[str] = None,   # question audio path
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        streaming_callback=None,  # 流式生成回调函数
        session_id=None
    ) -> torch.Tensor:
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
            image = image * 2.0 - 1.0
            image = image[:, :, None]
            image_prep_time = time.time() - start_time
            self._record_timing("forward_image_preprocessing", image_prep_time)
            
            start_time = time.time()
            self.causal_pipe.vae.to("cuda:1")
            # Use num_frames from noise tensor instead of hardcoded 21
            img_lat = self.causal_pipe.vae.encode(videos=image.to(dtype=self.dtype),device="cuda:1").repeat(1,1,num_frames,1,1)
            msk = torch.zeros_like(img_lat)[:,:1]
            msk[:, :, 1:] = 1
            img_lat = torch.cat([img_lat, msk], dim=1)
            image_encode_time = time.time() - start_time
            self._record_timing("forward_image_vae_encode", image_encode_time)
            print("img_lat:",img_lat.shape)

        # inference (prepare for run_pipeline)
        batch_size, num_frames, num_channels, height, width = noise.shape
        # frame block calculations
        assert num_frames % self.args.num_frame_per_block == 0
        num_blocks = num_frames // self.args.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        # text conditioning
        self.causal_pipe.text_encoder.to("cuda")
        self.causal_pipe.audio_encoder.to("cuda")
        self.causal_pipe.vae.clear_cache()
        conditional_dict = self.causal_pipe.encode_text_prompts(text_prompts, positive=True)
        conditional_dict["image"] = img_lat
        # TODO: add audio embeddings

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        # initialize KV caches
        self.causal_pipe.setup_caches(batch_size, noise.dtype, noise.device)
        print("Pipelined conversation init done")
        self.run_pipeline(
            noise, 
            batch_size,
            num_blocks,
            num_input_frames,
            initial_latent,
            conditional_dict,
            img_lat,
            output,
            audio_path,
            streaming_callback,
            session_id
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

                # Trigger denoising event - use chunk_id instead of self.current_clock to avoid race condition
                self.denoising_events[chunk_id].set()
                
                # Create VAE event for this block
                self.vae_events.append(threading.Event())
                
                # Add VAE decoding task (video is already decoded by causal_pipe.inference)
                # But we still need to process it for streaming
                # if current_start_frame > 0:
                #     task.update({
                #         "latents": output[:, current_start_frame - 1:current_start_frame + current_num_frames].clone()
                #     })
                # else:
                # if current_start_frame > 0:
                #     latents = output[:, current_start_frame-1:current_start_frame + current_num_frames]
                # else:
                latents = output[:, current_start_frame:current_start_frame + current_num_frames]
                print(f"chunk_id: {chunk_id}, latents: {latents.shape}")
                task.update({
                    "latents": latents.clone()
                })
                self.vae_queue.put(task)
                
                print(f"Causal denoising inference completed for block {chunk_id}")
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

    def _send_chunk_frames(self, chunk_id, streaming_callback, session_id, total_frames_generated, total_blocks):
        if chunk_id < 0:
            return 0
        """Send frames for a completed block using frame-by-frame method"""
        print(f"Sending block {chunk_id} frames")
        # import pdb; pdb.set_trace()
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
        
        # audio
        audio_segment = block_data["audio_segment"]
        
        # 直接使用逐帧发送方式，避免视频流同步问题
        print(f"Using frame-by-frame method for chunk {chunk_id}")
        frame_send_start_time = time.time()
        total_frames_generated = self._send_frames_fallback(
            video, audio_segment, chunk_id, streaming_callback, 
            session_id, total_frames_generated, total_blocks
        )
        frame_send_time = time.time() - frame_send_start_time
        self._record_timing("send_frames_processing", frame_send_time, chunk_id)
        
        # 记录整个发送过程的总时间
        total_send_time = time.time() - send_start_time
        self._record_timing("send_total", total_send_time, chunk_id)
        print(f"total_frames_generated: {total_frames_generated}")
        
        # return total_frames_generated + video.shape[1]
        return total_frames_generated

    def _send_frames_fallback(self, video, audio_segment, chunk_id, streaming_callback, session_id, total_frames_generated, total_blocks):
        """Frame-by-frame sending method (following pipelined_inference.py pattern)"""
        print(f"Using frame-by-frame method for chunk {chunk_id}")
        # import pdb; pdb.set_trace()
        total_frames = (total_blocks * self.args.num_frame_per_block - 1) * 4 + 1
        # import pdb; pdb.set_trace()
        frame_duration = 1.0 / self.args.fps
        audio_sample_rate = 24000  # 音频采样率
        samples_per_frame = int(audio_sample_rate / self.args.fps)  # 每帧对应的采样点数
        frame_processing_times = []
        for frame_idx in range(video.shape[1]):
            frame_start_time = time.time()
            
            frame_data = video[:, frame_idx]
            
            # 图像转换时间
            convert_start_time = time.time()
            frame_np = (frame_data.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype('uint8')
            frame_pil = Image.fromarray(frame_np)
            buffer = io.BytesIO()
            frame_pil.save(buffer, format='JPEG', quality=85)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            convert_time = time.time() - convert_start_time
            
            # audio clip - 使用采样点索引而不是时间
            audio_start_sample = int(frame_idx * samples_per_frame)
            audio_end_sample = int(audio_start_sample + samples_per_frame)
            
            # 边界检查：确保索引在有效范围内
            audio_start_sample = max(0, audio_start_sample)
            audio_end_sample = min(len(audio_segment), audio_end_sample)
            
            audio_start_time = frame_idx * frame_duration  # 保留时间信息用于回调
            audio_end_time = audio_start_time + frame_duration
            
            # 提取音频片段
            if audio_start_sample >= audio_end_sample:
                # 如果没有有效音频，创建静音
                current_frame_audio = np.zeros(samples_per_frame, dtype=np.int16)
                current_frame_audio_base64 = None
                print(f"Warning: No valid audio for frame {frame_idx}, using silence")
            else:
                current_frame_audio = audio_segment[audio_start_sample:audio_end_sample]
                
                # 如果音频是立体声，转换为单声道
                if len(current_frame_audio.shape) > 1:
                    current_frame_audio = np.mean(current_frame_audio, axis=1)
                
                # 转换为16位整数
                current_frame_audio = (current_frame_audio * 32767).astype(np.int16)
                
                # 写入临时文件并转换为base64
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, current_frame_audio, audio_sample_rate)
                    with open(temp_file.name, 'rb') as f:
                        current_frame_audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                    os.unlink(temp_file.name)
            
            total_frames_generated += 1
            progress_overall = (total_frames_generated / total_frames) * 100 if total_frames else None
            
            # 回调发送时间
            callback_start_time = time.time()
            # import pdb; pdb.set_trace()
            streaming_callback({
                "type": "video_frame",
                "session_id": session_id,
                "frame_data": frame_base64,
                "frame_number": total_frames_generated,
                "total_frames": total_frames,
                "chunk_number": chunk_id,
                "progress": progress_overall,
                "chunk_progress": ((frame_idx + 1) / video.shape[1]) * 100,
                "audio_segment": current_frame_audio_base64,
                "audio_start_time": audio_start_time,
                "audio_duration": frame_duration
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

    def _save_complete_video_with_audio(self, num_blocks: int, audio_path: Optional[str], session_id: Optional[str]):
        """
        保存完整视频并与音频合并
        
        Args:
            num_blocks: 总的块数
            audio_path: 音频文件路径（可选）
            session_id: 会话ID（用于文件命名）
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
        session_str = f"_{session_id}" if session_id else ""
        video_no_audio_path = os.path.join(output_dir, f"video_no_audio{session_str}_{timestamp}.mp4")
        video_with_audio_path = os.path.join(output_dir, f"video_with_audio{session_str}_{timestamp}.mp4")
        
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
