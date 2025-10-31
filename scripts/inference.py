import subprocess
import os, sys
import io
import base64
from glob import glob
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
from OmniAvatar.utils.args_config import parse_args

from OmniAvatar.utils.io_utils import load_state_dict 
from peft import LoraConfig, inject_adapter_in_model
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
import torch.distributed as dist
import torchvision.transforms as TT
from transformers import Wav2Vec2FeatureExtractor
import torchvision.transforms as transforms
import torch.nn.functional as F
from OmniAvatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg
from OmniAvatar.distributed.fsdp import shard_model
from PIL import Image

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 设置当前GPU
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU

def read_from_file(p):
    with open(p, "r") as fin:
        for l in fin:
            yield l.strip()

def match_size(image_size, h, w):   # h = 1472, w = 1104, return select_size: [720, 720]
    ratio_ = 9999
    size_ = 9999
    select_size = None
    # image sizes: 
    # image_sizes_720: [[400, 720], 
    #          [720, 720], 
    #          [720, 400]]
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

class WanInferencePipeline(nn.Module):
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
        pipe = WanVideoPipeline.from_model_manager(model_manager, 
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
        # auto wrapped modules, like: linear, embedding, ...
        # **set device and dtype automatically**
        # pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required. 
        if args.use_fsdp:
            shard_fn = partial(shard_model, device_id=self.device)
            pipe.dit = shard_fn(pipe.dit)
        # load modle to device
        # import pdb; pdb.set_trace()
        # pipe.load_models_to_device(pipe.model_names)
        for model_name in pipe.model_names:
            model = getattr(pipe, model_name)
            if model is not None:
                model.to(self.device)
                print(f"Move {model_name} to {self.device}")
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
                streaming_callback=None,  # 新增：流式生成回调函数
                streaming_mode=False,    # 新增：是否启用流式模式
                session_id=None):       # 新增：会话ID
        overlap_frame = overlap_frame if overlap_frame is not None else self.args.overlap_frame # args.overlap_frame is 13
        num_steps = num_steps if num_steps is not None else self.args.num_steps # denoise steps
        # cfg (audio and text)
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale

        # tai2v
        if image_path is not None:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
        # ta2v
        else:
            image = None
            select_size = [height, width]
        # calculate video frames (pixel level), L = 57
        print(f"select_size: {select_size}")
        L = int(args.max_tokens * 16 * 16 * 4 / select_size[0] / select_size[1])    # 16, 16, 4是vae+patchify的下采样倍数
        L = L // 4 * 4 + 1 if L % 4 != 0 else L - 3  # video frames
        # calculate latent frames (latent level), T = 15
        T = (L + 3) // 4  # latent frames
        print(f"L = {L}, T = {T}")

        if self.args.i2v:
            if self.args.random_prefix_frames:
                fixed_frame = overlap_frame
                assert fixed_frame % 4 == 1
            else:
                fixed_frame = 1 # this branch, fixed frame is video level, prefix lat frame is latent level
            prefix_lat_frame = (3 + fixed_frame) // 4   # 1
            first_fixed_frame = 1   # 1
        else:
            fixed_frame = 0
            prefix_lat_frame = 0
            first_fixed_frame = 0
        print(f"fixed_frame = {fixed_frame}, prefix_lat_frame = {prefix_lat_frame}, first_fixed_frame = {first_fixed_frame}")
        # preprocess audio condition
        if audio_path is not None and args.use_audio:
            # load audio wave
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)  # shape: (SR * time,), e.g., (73728,)
            # wav2vec (audio features)
            input_values = np.squeeze(self.wav_feature_extractor(audio, sampling_rate=16000).input_values)  # also (73728,) ??
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            ori_audio_len = audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
            input_values = input_values.unsqueeze(0)    # (1, 73728)
            # padding audio (audio_len = 1 / k * L or audio_len = k * L) audio_len是L的整数倍或L是audio_len的整数倍
            if audio_len < L - first_fixed_frame:   # L - first_fixed_frame is the number of generated frames (pixel level)
                audio_len = audio_len + ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
            elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
                audio_len = audio_len + ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
            input_values = F.pad(input_values, (0, audio_len * int(self.args.sample_rate / self.args.fps) - input_values.shape[1]), mode='constant', value=0)   # (1, L')
            # encode audio features
            with torch.no_grad():
                # audio encoder:
                # step 1: model.feature_extractor(input_values), output (1, 512, 230), 512 is the pretrained feature extractor's output channels
                # step 2: linear_interpolation to audio_len (pixel level frame num)
                # step 3: feature projection: conv_out_channels(512) -> encoder_hidden_size (768)
                # step 4: encoder forward (1, audio_len, hidden_size)
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                # concat hidden states from each layer
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1) # (1, audio_len, hidden_size * layers)
            seq_len = audio_len
            audio_embeddings = audio_embeddings.squeeze(0)
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])   # ?
        else:
            audio_embeddings = None
        # loop
        times = (seq_len - L + first_fixed_frame) // (L - fixed_frame) + 1
        if times * (L - fixed_frame) + fixed_frame < seq_len:
            times += 1
        print(f"times = {times}")
        video = []
        image_emb = {}
        img_lat = None
        if args.i2v:
            # self.pipe.load_models_to_device(['vae'])
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)    # (b, c, t, h, w) (1, 16, 1, H//8, W//8)
            img_lat_backup = img_lat.clone()
            msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1]) # (b, 1, T, h, w), (1, 1, 15, H//8, W//8)
            image_cat = img_lat.repeat(1, 1, T, 1, 1)   # (b, c, T, h, w) (1, 16, 15, H//8, W//8)
            msk[:, :, 1:] = 1   # after the first frame
            image_emb["y"] = torch.cat([image_cat, msk], dim=1) # (b, c + 1, T, h, w) (1, 17, 15, H//8, W//8)
        
        # 流式生成模式
        if streaming_mode and streaming_callback is not None:
            print(f"Starting streaming generation with {times} chunks...")            
            # import pdb; pdb.set_trace()
            # 流式生成变量
            total_frames_generated = 0
            if session_id is None:
                session_id = f"session_{int(torch.rand(1).item() * 1000000)}"

            # 创建输出目录用于保存video
            import os
            from datetime import datetime
            output_dir = f'demo_out/streaming_videos_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.makedirs(output_dir, exist_ok=True)

            print(f"Video will be saved to: {output_dir}")

            # 发送开始事件
            streaming_callback({
                'type': 'start',
                'session_id': session_id,
                'total_chunks': times,
                'message': 'Video generation started',
                'output_dir': output_dir
            })

            # 配置：是否在每个chunk生成后立刻推送帧（实时更流畅）
            stream_frames_immediately = True

            # 第一阶段：生成各个chunk；如配置为实时，则边生成边推送
            print("Phase 1: Generating chunks" + (" and streaming frames immediately..." if stream_frames_immediately else " and collecting frames...") )
            all_video_segments = []

            for t in range(times):
                print(f"[{t+1}/{times}] Generating chunk {t+1}")

                # 发送chunk开始事件
                streaming_callback({
                    'type': 'chunk_start',
                    'session_id': session_id,
                    'chunk_number': t + 1,
                    'total_chunks': times,
                    'message': f'Generating chunk {t + 1}'
                })

                audio_emb = {}
                if t == 0:
                    overlap = first_fixed_frame # value = 1
                else:
                    overlap = fixed_frame   # value = 1 + 4*n (5)
                    image_emb["y"][:, -1:, :prefix_lat_frame] = 0 # image_emb["y"][:, -1:] 意味着mask, 第一次推理是mask只有1，往后都是mask overlap
                prefix_overlap = (3 + overlap) // 4 # value = 2
                print(f"prefix_overlap: {prefix_overlap}")
                if audio_embeddings is not None:
                    if t == 0:
                        audio_tensor = audio_embeddings[:min(L - overlap, audio_embeddings.shape[0])]
                    else:
                        audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                        audio_tensor = audio_embeddings[audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])]
                    audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                    audio_prefix = audio_tensor[-fixed_frame:]
                    audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                    audio_emb["audio_emb"] = audio_tensor
                else:
                    audio_prefix = None
                if image is not None and img_lat is None:
                    # self.pipe.load_models_to_device(['vae'])
                    img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
                    # print(f"img_lat: {img_lat.shape}")
                    assert img_lat.shape[2] == prefix_overlap
                # print("")
                img_lat = torch.cat([img_lat_backup, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))], dim=2)

                # 生成当前chunk
                frames, _, latents = self.pipe.log_video(
                    lat=img_lat,
                    prompt=prompt,
                    fixed_frame=prefix_overlap,
                    image_emb=image_emb,
                    audio_emb=audio_emb,
                    negative_prompt=negative_prompt,
                    cfg_scale=guidance_scale,
                    audio_cfg_scale=audio_scale if audio_scale is not None else guidance_scale,
                    num_inference_steps=num_steps,
                    tea_cache_l1_thresh=args.tea_cache_l1_thresh,
                    tea_cache_model_id="Wan2.1-T2V-14B",
                    return_latent=True
                )

                # 处理生成的frames
                # img_lat = None
                image = (frames[:, -fixed_frame:].clip(0, 1) * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()

                # 准备当前chunk的frames
                if t == 0:
                    current_chunk_frames = frames
                else:
                    current_chunk_frames = frames[:, overlap:]

                # 添加到video列表（用于后续保存或备用）
                all_video_segments.append(current_chunk_frames)

                # 推送当前chunk的帧（若启用即时推送）
                if stream_frames_immediately:
                    for frame_idx in range(current_chunk_frames.shape[1]):
                        frame_data = current_chunk_frames[:, frame_idx]  # (1, 3, H, W)

                        # 转换为base64格式用于传输
                        frame_np = (frame_data.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                        frame_pil = Image.fromarray(frame_np)
                        buffer = io.BytesIO()
                        frame_pil.save(buffer, format='JPEG', quality=85)
                        frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                        total_frames_generated += 1
                        progress_overall = (total_frames_generated / ori_audio_len) * 100 if 'ori_audio_len' in locals() and ori_audio_len else None

                        streaming_callback({
                            'type': 'video_frame',
                            'session_id': session_id,
                            'frame_data': frame_base64,
                            'frame_number': total_frames_generated,
                            'total_frames': ori_audio_len if 'ori_audio_len' in locals() and ori_audio_len else None,
                            'chunk_number': t + 1,
                            'progress': progress_overall,
                            'chunk_progress': ((frame_idx + 1) / current_chunk_frames.shape[1]) * 100
                        })

                else:
                    # 如未即时推送，仅累计帧数用于进度
                    total_frames_generated += current_chunk_frames.shape[1]

                # 发送chunk完成事件（当前chunk生成与(可选)推送完成）
                progress_overall_after_chunk = (total_frames_generated / ori_audio_len) * 100 if 'ori_audio_len' in locals() and ori_audio_len else None
                streaming_callback({
                    'type': 'chunk_complete',
                    'session_id': session_id,
                    'chunk_number': t + 1,
                    'total_chunks': times,
                    'frames_in_chunk': current_chunk_frames.shape[1],
                    'total_frames_generated': total_frames_generated,
                    'progress': progress_overall_after_chunk,
                    'message': f'Chunk {t + 1} completed' + (" (generation + streaming)" if stream_frames_immediately else " (generation)")
                })

                print(f"Chunk {t+1} completed. Frames: {current_chunk_frames.shape[1]}")

            # 第一阶段完成：保存完整的video到文件
            print("\nPhase 1 completed! Saving video to file...")

            # 拼接所有video segments
            complete_video = torch.cat(all_video_segments, dim=1)
            complete_video = complete_video[:, :ori_audio_len + 1]

            # 保存video文件
            try:
                from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
                video_save_path = os.path.join(output_dir, f"streaming_video_{session_id}")
                save_video_as_grid_and_mp4(
                    complete_video,
                    output_dir,
                    args.fps,
                    prompt=prompt,
                    prefix=f'streaming_video_{session_id}'
                )
                print(f"✅ Video saved successfully to: {output_dir}")

                # 发送video保存完成事件
                streaming_callback({
                    'type': 'video_saved',
                    'session_id': session_id,
                    'output_path': output_dir,
                    'message': f'Video saved to {output_dir}'
                })

            except Exception as e:
                print(f"❌ Error saving video: {e}")
                streaming_callback({
                    'type': 'video_save_error',
                    'session_id': session_id,
                    'error': str(e),
                    'message': 'Failed to save video'
                })

            # 如已即时推送，则无需第二阶段再重复发送
            if not stream_frames_immediately:
                print("\nPhase 2: Starting streaming transmission...")

                # 发送流式开始事件
                streaming_callback({
                    'type': 'streaming_start',
                    'session_id': session_id,
                    'total_frames': ori_audio_len,
                    'message': 'Starting streaming transmission'
                })

                # 重新遍历所有frames并发送
                frame_counter = 0
                for t in range(times):
                    print(f"[{t+1}/{times}] Streaming chunk {t+1}")

                    # 获取对应的chunk frames
                    chunk_frames = all_video_segments[t]

                    # 逐帧发送
                    for frame_idx in range(chunk_frames.shape[1]):
                        frame_data = chunk_frames[:, frame_idx]  # (1, 3, H, W)

                        # 转换为base64格式用于传输
                        frame_np = (frame_data.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                        frame_pil = Image.fromarray(frame_np)
                        buffer = io.BytesIO()
                        frame_pil.save(buffer, format='JPEG', quality=85)
                        frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                        # 调用流式回调函数，实时返回每一帧
                        frame_info = {
                            'type': 'video_frame',
                            'session_id': session_id,
                            'frame_data': frame_base64,
                            'frame_number': frame_counter + frame_idx + 1,
                            'total_frames': ori_audio_len,
                            'chunk_number': t + 1,
                            'progress': ((frame_counter + frame_idx + 1) / ori_audio_len) * 100,
                            'chunk_progress': ((t + 1) / times) * 100
                        }

                        streaming_callback(frame_info)

                    frame_counter += chunk_frames.shape[1]
                    print(f"Chunk {t+1} streaming completed. Frames sent: {chunk_frames.shape[1]}")

            # 流式模式完成，返回完整视频
            final_video = torch.cat(all_video_segments, dim=1)
            final_video = final_video[:, :ori_audio_len + 1]

            # 发送完成事件
            streaming_callback({
                'type': 'generation_complete',
                'session_id': session_id,
                'total_frames': total_frames_generated,
                'output_path': output_dir,
                'message': 'Video generation and streaming completed'
            })

            print(f"🎉 Streaming generation completed! Video saved to: {output_dir}")
            return final_video
            
        else:
            # 原有的非流式模式
            for t in range(times):
                print(f"[{t+1}/{times}]")
                audio_emb = {}
                if t == 0:
                    overlap = first_fixed_frame # value = 1
                else:
                    overlap = fixed_frame   # value = 1
                    image_emb["y"][:, -1:, :prefix_lat_frame] = 0 # image_emb["y"][:, -1:] 意味着mask, 第一次推理是mask只有1，往后都是mask overlap
                prefix_overlap = (3 + overlap) // 4
                if audio_embeddings is not None:
                    if t == 0:
                        audio_tensor = audio_embeddings[:min(L - overlap, audio_embeddings.shape[0])]
                    else:
                        audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                        audio_tensor = audio_embeddings[audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])]
                    audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                    audio_prefix = audio_tensor[-fixed_frame:]
                    audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                    audio_emb["audio_emb"] = audio_tensor
                else:
                    audio_prefix = None
                if image is not None and img_lat is None:
                    # self.pipe.load_models_to_device(['vae'])
                    img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
                    assert img_lat.shape[2] == prefix_overlap
                img_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))], dim=2)
                frames, _, latents = self.pipe.log_video(
                    lat=img_lat,
                    prompt=prompt,
                    fixed_frame=prefix_overlap,
                    image_emb=image_emb,
                    audio_emb=audio_emb,
                    negative_prompt=negative_prompt,
                    cfg_scale=guidance_scale,
                    audio_cfg_scale=audio_scale if audio_scale is not None else guidance_scale,
                    num_inference_steps=num_steps,
                    tea_cache_l1_thresh=args.tea_cache_l1_thresh,
                    tea_cache_model_id="Wan2.1-T2V-14B",
                    return_latent=True
                )
                img_lat = None
                image = (frames[:, -fixed_frame:].clip(0, 1) * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()
                if t == 0:
                    video.append(frames)
                else:
                    video.append(frames[:, overlap:])
            video = torch.cat(video, dim=1)
            video = video[:, :ori_audio_len + 1]
            return video


def main():
    args = parse_args()
    set_seed(args.seed)
    # laod data
    data_iter = read_from_file(args.input_file)
    exp_name = os.path.basename(args.exp_path)
    seq_len = args.seq_len
    date_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Text-to-video
    inferpipe = WanInferencePipeline(args)
    if args.sp_size > 1:
        date_name = inferpipe.pipe.sp_group.broadcast_object_list([date_name])
        date_name = date_name[0]
    output_dir = f'demo_out/{exp_name}/res_{os.path.splitext(os.path.basename(args.input_file))[0]}_'\
                f'seed{args.seed}_step{args.num_steps}_cfg{args.guidance_scale}_'\
                f'ovlp{args.overlap_frame}_{args.max_tokens}_{args.fps}_{date_name}'
    if args.tea_cache_l1_thresh > 0:
        output_dir = f'{output_dir}_tea{args.tea_cache_l1_thresh}'
    if args.audio_scale is not None:
        output_dir = f'{output_dir}_acfg{args.audio_scale}'
    if args.max_hw == 1280:
        output_dir = f'{output_dir}_720p'
    for idx, text in tqdm(enumerate(data_iter)):
        if len(text) == 0:
            continue
        input_list = text.split("@@")
        assert len(input_list)<=3
        if len(input_list) == 0:
            continue
        elif len(input_list) == 1:
            text, image_path, audio_path = input_list[0], None, None
        elif len(input_list) == 2:
            text, image_path, audio_path = input_list[0], input_list[1], None
        elif len(input_list) == 3:
            text, image_path, audio_path = input_list[0], input_list[1], input_list[2]
        audio_dir = output_dir + '/audio'
        os.makedirs(audio_dir, exist_ok=True)
        if args.silence_duration_s > 0:
            input_audio_path = os.path.join(audio_dir, f"audio_input_{idx:03d}.wav")
        else:
            input_audio_path = audio_path
        prompt_dir = output_dir + '/prompt'
        os.makedirs(prompt_dir, exist_ok=True)
        if dist.get_rank() == 0:
            if args.silence_duration_s > 0:
                add_silence_to_audio_ffmpeg(audio_path, input_audio_path, args.silence_duration_s)
        dist.barrier()
        video = inferpipe(
            prompt=text,
            image_path=image_path,
            audio_path=input_audio_path,
            seq_len=seq_len
        )
        tmp2_audio_path = os.path.join(audio_dir, f"audio_out_{idx:03d}.wav") # 因为第一帧是参考帧，因此需要往前1/25秒
        prompt_path = os.path.join(prompt_dir, f"prompt_{idx:03d}.txt") 
        
        if dist.get_rank() == 0:
            add_silence_to_audio_ffmpeg(audio_path, tmp2_audio_path, 1.0 / args.fps + args.silence_duration_s)
            save_video_as_grid_and_mp4(video, 
                                    output_dir, 
                                    args.fps, 
                                    prompt=text,
                                    prompt_path = prompt_path,
                                    audio_path=tmp2_audio_path if args.use_audio else None, 
                                    prefix=f'result_{idx:03d}')
        dist.barrier()

class NoPrint:
    def write(self, x):
        pass
    def flush(self):
        pass

if __name__ == '__main__':
    if not args.debug:
        if args.local_rank != 0: # 屏蔽除0外的输出
            sys.stdout = NoPrint()
    main()