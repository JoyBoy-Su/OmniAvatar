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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from peft import LoraConfig, inject_adapter_in_model

from OmniAvatar.utils.args_config import parse_args
from scripts.inference import WanInferencePipeline, set_seed
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4, load_state_dict
from OmniAvatar.distributed.fsdp import shard_model
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms as TT
from transformers import Wav2Vec2FeatureExtractor
from PIL import Image
from OmniAvatar.pipelined_wan_video import PipelinedWanVideoPipeline
from OmniAvatar.models.model_manager import ModelManager

args = parse_args()

# 导入必要的函数
def match_size(image_size, h, w):
    ratio_ = 9999
    size_ = 9999
    select_size = None
    for image_s in image_size:
        ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
        size_tmp = abs(max(image_s) - max(w, h))
        if ratio_tmp < ratio_:
            ratio_ = ratio_tmp
            size_ = size_tmp
            select_size = image_s
        if ratio_ == ratio_tmp:
            if size_ == size_tmp:
                select_size = image_s
    return select_size

def resize_pad(image, ori_size, tgt_size):
    h, w = ori_size
    scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
    scale_h = int(h * scale_ratio)
    scale_w = int(w * scale_ratio)

    image = transforms.Resize(size=[scale_h, scale_w])(image)

    padding_h = tgt_size[0] - scale_h
    padding_w = tgt_size[1] - scale_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return image


class PipelinedWanInferencePipeline(nn.Module):
    """
    流水线化的WanInferencePipeline实现
    主要改进：
    1. prompt encoding提前到times循环外
    2. denoising inference和VAE decoding实现流水线并行
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
        # load model
        self.pipe = self.load_model()
        if args.i2v:
            chained_trainsforms = []
            chained_trainsforms.append(TT.ToTensor())
            self.transform = TT.Compose(chained_trainsforms)
        if args.use_audio:
            # load audio encoder
            from OmniAvatar.models.wav2vec import Wav2VecModel
            # what's the difference between wav_feature_extractor and audio_encoder
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
            self.audio_encoder = Wav2VecModel.from_pretrained(args.wav2vec_path, local_files_only=True).to(device=self.device)
            self.audio_encoder.feature_extractor._freeze_parameters()
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
    
    def load_model(self):
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        # parallel settings
        from xfuser.core.distributed import (initialize_model_parallel,
                                            init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=args.sp_size,
            ring_degree=1,
            ulysses_degree=args.sp_size,
        )
        torch.cuda.set_device(dist.get_rank())
        # load checkpoint
        ckpt_path = f'{args.exp_path}/pytorch_model.pt'
        assert os.path.exists(ckpt_path), f"pytorch_model.pt not found in {args.exp_path}"
        if args.train_architecture == 'lora':
            args.pretrained_lora_path = pretrained_lora_path = ckpt_path
        else:
            resume_path = ckpt_path
        
        self.step = 0

        # load pretrained models first
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models(
            [
                args.dit_path.split(","),   # load dit
                args.text_encoder_path,     # load text encoder
                args.vae_path               # load vae
            ],
            torch_dtype=self.dtype, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
            device='cpu',
        )   # model.manager: self.model, self.model_path, self.model_name

        # fetch models from model_manager: text_encoder, dit, vae, image_encoder
        pipe = PipelinedWanVideoPipeline.from_model_manager(model_manager, 
                                                torch_dtype=self.dtype, 
                                                device=f"cuda:{dist.get_rank()}", 
                                                use_usp=True if args.sp_size > 1 else False,
                                                infer=True)
        # load updated weights
        if args.train_architecture == "lora":
            print(f'Use LoRA: lora rank: {args.lora_rank}, lora alpha: {args.lora_alpha}')
            self.add_lora_to_model(
                    pipe.denoising_model(),
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_target_modules=args.lora_target_modules,
                    init_lora_weights=args.init_lora_weights,
                    pretrained_lora_path=pretrained_lora_path,
                )
        else:
            missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(load_state_dict(resume_path), strict=True)
            print(f"load from {resume_path}, {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
        pipe.requires_grad_(False)
        pipe.eval()
        if args.use_fsdp:
            shard_fn = partial(shard_model, device_id=self.device)
            pipe.dit = shard_fn(pipe.dit)
        # load modle to device
        for model_name in pipe.model_names:
            model = getattr(pipe, model_name)
            if model is not None:
                model.to(self.device)
                print(f"Move {model_name} to {self.device}")
        # import pdb; pdb.set_trace()
        return pipe
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    # TODO: wrap input params
    def run_pipeline(
            self,
            L,
            T,
            audio_embeddings,
            audio_prefix,
            prompt_emb_posi,
            prompt_emb_nega,
            image_emb,
            guidance_scale,
            audio_scale,
            num_steps,
            fixed_frame,
            first_fixed_frame,
            image,
            img_lat,
            img_lat_backup,
            times,
            streaming_callback,
            session_id,
            ori_audio_len
        ):
        # clear
        self.current_clock = 0
        self.denoising_queue.queue.clear()
        self.vae_queue.queue.clear()
        self.result_buffer.clear()
        self.denoising_events.clear()
        self.vae_events.clear()
        # prepare
        all_video_segments = []
        completed_chunks = 0
        total_frames_generated = 0
        # run async thread
        self.denoising_thread = threading.Thread(target=self._denoising_worker, daemon=True)
        self.vae_thread = threading.Thread(target=self._vae_worker, daemon=True)
        self.denoising_thread.start()
        self.vae_thread.start()
        # submit chunk1
        overlap = first_fixed_frame
        prefix_overlap = (3 + overlap) // 4
        print(f"Prefix overlap: {prefix_overlap}")
        # audio input, TODO: check
        audio_emb = {}
        audio_tensor = audio_embeddings[:min(L - overlap, audio_embeddings.shape[0])]
        audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
        audio_prefix = audio_tensor[-fixed_frame:]
        audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        audio_emb["audio_emb"] = audio_tensor
        # image input
        if image is not None and img_lat is None:
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
            assert img_lat.shape[2] == prefix_overlap
        img_lat = torch.cat([img_lat_backup, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))], dim=2)
        # denosing task
        self.denoising_events.append(threading.Event()) # add denoising event
        denoising_task = {
            "chunk_id": 0,
            "img_lat": img_lat.clone(), # 需要依赖上一个时刻的latents（目前是实现overlap逻辑，后面是需要kv cache）
            "prompt_emb_posi": prompt_emb_posi,
            "prompt_emb_nega": prompt_emb_nega,
            "image_emb": image_emb.copy(),
            "audio_emb": audio_emb.copy(),
            "guidance_scale": guidance_scale,
            "audio_scale": audio_scale,
            "num_steps": num_steps,
            "prefix_overlap": prefix_overlap,
            "overlap": overlap,
            "first_fixed_frame": first_fixed_frame,
            "fixed_frame": fixed_frame
        }
        self.denoising_queue.put(denoising_task)
        # wait
        self.denoising_events[self.current_clock].wait()
        # loop
        for i in range(1, times):
            self.current_clock += 1
            print(f"===================== Current clock: {self.current_clock} ======================")
            # TODO: read kv cache and update denoising input
            # next time clock
            self.denoising_events.append(threading.Event())
            overlap = fixed_frame
            prefix_overlap = (3 + overlap) // 4
            image_emb["y"][:, -1:, :prefix_overlap] = 0
            print(f"prefix overlap: {prefix_overlap}")
            # update audio embeddings
            audio_start = L - first_fixed_frame + (i - 1) * (L - overlap)
            audio_tensor = audio_embeddings[audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])]
            audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
            audio_prefix = audio_tensor[-fixed_frame:]
            audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            audio_emb["audio_emb"] = audio_tensor
            # update img lat
            img_lat = torch.cat([img_lat_backup, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))], dim=2)
            # denosing task
            denoising_task = {
                "chunk_id": i,
                "img_lat": img_lat.clone(),
                "prompt_emb_posi": prompt_emb_posi,
                "prompt_emb_nega": prompt_emb_nega,
                "image_emb": image_emb.copy(),
                "audio_emb": audio_emb.copy(),
                "guidance_scale": guidance_scale,
                "audio_scale": audio_scale,
                "num_steps": num_steps,
                "prefix_overlap": prefix_overlap,
                "overlap": overlap,
                "first_fixed_frame": first_fixed_frame,
                "fixed_frame": fixed_frame
            }
            self.denoising_queue.put(denoising_task)
            # wait denoising clock and decoding clock - 1
            self.denoising_events[self.current_clock].wait()
            self.vae_events[self.current_clock - 1].wait()
            # send chunk frames
            self._send_chunk_frames(i - 1, streaming_callback, session_id, total_frames_generated, ori_audio_len, times)
        # send last chunk
        self._send_chunk_frames(i - 1, streaming_callback, session_id, total_frames_generated, ori_audio_len, times)
        
    def forward(self, 
                prompt,             # text prompt
                image_path=None,    # reference image
                audio_path=None,    # reference audio
                seq_len=101,        # not used while audio_path is not None
                height=720, 
                width=720,
                overlap_frame=None,
                num_steps=None,
                negative_prompt=None,
                guidance_scale=None,
                audio_scale=None,
                streaming_callback=None,  # 流式生成回调函数
                streaming_mode=False,    # 是否启用流式模式
                session_id=None):       # 会话ID
        overlap_frame = overlap_frame if overlap_frame is not None else self.args.overlap_frame
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale

        # tai2v, preprocess image
        if image_path is not None:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
        else:
            image = None
            select_size = [height, width]
            
        # calculate video frames (pixel level), L = 57
        print(f"select_size: {select_size}")
        L = int(self.args.max_tokens * 16 * 16 * 4 / select_size[0] / select_size[1])
        L = L // 4 * 4 + 1 if L % 4 != 0 else L - 3  # video frames
        # calculate latent frames (latent level), T = 15
        T = (L + 3) // 4  # latent frames
        print(f"L = {L}, T = {T}")

        if self.args.i2v:
            if self.args.random_prefix_frames:
                fixed_frame = overlap_frame
                assert fixed_frame % 4 == 1
            else:
                fixed_frame = 1
            prefix_lat_frame = (3 + fixed_frame) // 4
            first_fixed_frame = 1
        else:
            fixed_frame = 0
            prefix_lat_frame = 0
            first_fixed_frame = 0
            
        print(f"fixed_frame = {fixed_frame}, prefix_lat_frame = {prefix_lat_frame}, first_fixed_frame = {first_fixed_frame}")
        
        # load audio, and get audio embeddings
        if audio_path is not None:
            # load audio wave
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
            # wav2vec (audio features)
            input_values = np.squeeze(self.wav_feature_extractor(audio, sampling_rate=16000).input_values)
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            ori_audio_len = audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
            input_values = input_values.unsqueeze(0)
            # padding audio
            if audio_len < L - first_fixed_frame:
                audio_len = audio_len + ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
            elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
                audio_len = audio_len + ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
            input_values = F.pad(input_values, (0, audio_len * int(self.args.sample_rate / self.args.fps) - input_values.shape[1]), mode='constant', value=0)
            # encode audio features
            with torch.no_grad():
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            seq_len = audio_len
            audio_embeddings = audio_embeddings.squeeze(0)
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
        else:
            audio_embeddings = None
            
        # loop
        times = (seq_len - L + first_fixed_frame) // (L - fixed_frame) + 1
        if times * (L - fixed_frame) + fixed_frame < seq_len:
            times += 1
        print(f"times = {times}")   # chunk的数量

        image_emb = {}
        img_lat = None
        # image emb: 包含了reference image的latent，用来在channel维度和latent concatenate（即每一帧都有reference image的features）
        # image lat: image的latent representation，(b, c, t, h, w) (1, 16, 1, H//8, W//8)
        if self.args.i2v:
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
            img_lat_backup = img_lat.clone()
            msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1])
            image_cat = img_lat.repeat(1, 1, T, 1, 1)
            msk[:, :, 1:] = 1
            image_emb["y"] = torch.cat([image_cat, msk], dim=1)
        print(f"Starting pipelined streaming generation with {times} chunks...")
        
        # 流式生成准备
        total_frames_generated = 0
        if 'ori_audio_len' not in locals():
            ori_audio_len = seq_len
        output_dir = f'demo_out/streaming_videos_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Video will be saved to: {output_dir}")

        # 发送开始事件
        streaming_callback({
            'type': 'start',
            'session_id': session_id,
            'total_chunks': times,
            'message': 'Pipelined video generation started',
            'output_dir': output_dir
        })

        # 第一阶段：提前进行prompt encoding（只执行一次）
        print("Phase 1: Prompt encoding (executed once)")
        start_time = time.time()
        # 提前进行prompt encoding
        prompt_emb_posi = self.pipe.encode_prompt(prompt)
        if guidance_scale != 1.0:
            prompt_emb_nega = self.pipe.encode_prompt(negative_prompt)
        else:
            prompt_emb_nega = None
        prompt_time = time.time() - start_time
        print(f"Prompt encoding completed in {prompt_time:.4f}s")
        # 发送prompt encoding完成事件
        streaming_callback({
            'type': 'prompt_encoding_complete',
            'session_id': session_id,
            'time_taken': prompt_time,
            'message': 'Prompt encoding completed'
        })

        # 第二阶段：流水线化的chunk生成
        print("Phase 2: Pipelined chunk generation")
        self.run_pipeline(
            L=L, T=T,
            audio_embeddings=audio_embeddings,
            audio_prefix=audio_prefix,
            prompt_emb_posi=prompt_emb_posi,
            prompt_emb_nega=prompt_emb_nega,
            image_emb=image_emb,
            guidance_scale=guidance_scale,
            audio_scale=audio_scale,
            num_steps=num_steps,
            fixed_frame=fixed_frame,
            first_fixed_frame=first_fixed_frame,
            image=image,
            img_lat=img_lat,
            img_lat_backup=img_lat_backup,
            times=times,
            streaming_callback=streaming_callback,
            session_id=session_id,
            ori_audio_len=ori_audio_len
        )
    
    def _denoising_worker(self):
        """
        Denoising worker
        从队列中获取任务并执行
        """
        while True:
            try:
                task = self.denoising_queue.get(timeout=0.1)
                chunk_id = task["chunk_id"]
                print(f"Processing denoising inference for chunk {chunk_id}")
                latents = self.pipe.denoising_inference(
                    lat=task['img_lat'],
                    prompt_emb_posi=task['prompt_emb_posi'],
                    prompt_emb_nega=task['prompt_emb_nega'],
                    fixed_frame=task['prefix_overlap'],
                    image_emb=task['image_emb'],
                    audio_emb=task['audio_emb'],
                    cfg_scale=task['guidance_scale'],
                    audio_cfg_scale=task['audio_scale'] if task['audio_scale'] is not None else task['guidance_scale'],
                    num_inference_steps=task['num_steps'],
                    tea_cache_l1_thresh=self.args.tea_cache_l1_thresh,
                    tea_cache_model_id="Wan2.1-T2V-14B"
                )
                # trigger vae event
                self.denoising_events[self.current_clock].set()
                # create vae events
                self.vae_events.append(threading.Event())
                # add a vae decoding task
                # TODO: wrapper the data format
                vae_task = {
                    "chunk_id": chunk_id,
                    "latents": latents,
                    "overlap": task["overlap"],
                    "first_fixed_frame": task["first_fixed_frame"],
                    "fixed_frame": task["fixed_frame"]
                }
                self.vae_queue.put(vae_task)
                print(f"Denoising inference completed for chunk {chunk_id}")
                self.denoising_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in denoising inference for chunk {chunk_id}: {e}")
                self.denoising_queue.task_done()
    
    def _vae_worker(self):
        """
        VAE decoding worker
        从队列中获取任务并执行
        """
        while True:
            try:
                task = self.vae_queue.get(timeout=0.1)
                chunk_id = task["chunk_id"]
                print(f"Processing vae decoding for chunk {chunk_id}")
                print(f"VAE decoding latents: {task['latents'].shape}")
                frames = self.pipe.vae_decoding(
                    latents=task["latents"],
                    tiled=False,
                    tile_size=(30, 52),
                    tile_stride=(15, 26)
                )
                # trigger vae event
                self.vae_events[self.current_clock - 1].set()
                # overlap, TODO: delete when switch to block_wise generation
                if chunk_id == 0:
                    current_chunk_frames = frames
                else:
                    current_chunk_frames = frames[:, task["overlap"]:]
                # assess result buffer with lock
                with self.result_lock:
                    self.result_buffer[chunk_id] = {"frames": current_chunk_frames}
                print(f"VAE decoding completed for chunk {chunk_id}")
                self.vae_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in VAE decoding for chunk {chunk_id}")
                self.vae_queue.task_done()

    def _send_chunk_frames(self, chunk_id, streaming_callback, session_id, total_frames_generated, total_frames, times):
        print(f"Sending chunk {chunk_id} frames")
        # get chunk from buffer
        while chunk_id not in self.result_buffer:
            print(f"self.result_buffer: {self.result_buffer}")
            time.sleep(0.1)
        # read chunk data
        chunk_data = self.result_buffer[chunk_id]
        current_chunk_frames = chunk_data["frames"]
        # send frames (loop)
        for frame_idx in range(current_chunk_frames.shape[1]):
            frame_data = current_chunk_frames[:, frame_idx]
            # convert to base64
            frame_np = (frame_data.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            frame_pil = Image.fromarray(frame_np)
            buffer = io.BytesIO()
            frame_pil.save(buffer, format='JPEG', quality=85)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            total_frames_generated += 1
            progress_overall = (total_frames_generated / total_frames) * 100 if total_frames else None
            streaming_callback({
                "type": "video_frame",
                "session_id": session_id,
                "frame_data": frame_base64,
                "frame_number": total_frames_generated,
                "total_frames": total_frames if total_frames else None,
                "chunk_number": chunk_id,
                "progress": progress_overall,
                "chunk_progress": ((frame_idx + 1) / current_chunk_frames.shape[1]) * 100
            })
        # chunk complete
        progress_overall_after_chunk = (total_frames_generated / total_frames) * 100 if total_frames else None
        streaming_callback({
            "type": "chunk_complete",
            "session_id": session_id,
            "chunk_number": chunk_id,
            "total_chunks": times,
            "frames_in_chunk": current_chunk_frames.shape[1],
            "total_frames_generated": total_frames_generated,
            "progress": progress_overall_after_chunk,
            "message": f"Chunk {chunk_id} completed (pipelined)"
        })
