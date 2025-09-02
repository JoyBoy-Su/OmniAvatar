import types
import os
from .models.model_manager import ModelManager
from .models.wan_video_dit import WanModel
from .models.wan_video_text_encoder import WanTextEncoder
from .models.wan_video_vae import WanVideoVAE
from .schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from .prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from .vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from .models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from .models.wan_video_dit import RMSNorm
from .models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
import time
import json
from collections import defaultdict

class PipelinedWanVideoPipeline(BasePipeline):
    """
    流水线化的WanVideoPipeline实现
    主要改进：
    1. 支持跳过prompt encoding
    2. 支持接收预编码的prompt embeddings
    3. 优化流水线处理流程
    4. cfg相关的多次前向传播并行实现
    """

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.sp_size = 1
        self.enable_timing = True
        self.current_session_timing = {}
        self.timing_stats = defaultdict(list)
        
    def run_dit_inference(rank, world_size, model, data_queue, result_queue):
        # init distributed environment
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        # load model to device
        model.to(f"cuda:{rank}")
        # get input data
        input_data = data_queue.get()
        

    def start_timing(self, stage_name):
        """开始计时某个阶段"""
        if self.enable_timing:
            self.current_session_timing[stage_name] = time.time()
    
    def end_timing(self, stage_name):
        """结束计时某个阶段并记录耗时"""
        if self.enable_timing and stage_name in self.current_session_timing:
            duration = time.time() - self.current_session_timing[stage_name]
            self.timing_stats[stage_name].append(duration)
            del self.current_session_timing[stage_name]
            return duration
        return 0.0

    def fetch_models(self, model_manager: ModelManager):
        # load text encode weight
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        # load dit weight
        self.dit = model_manager.fetch_model("wan_video_dit")
        # load vae weight
        self.vae = model_manager.fetch_model("wan_video_vae")
        # load image encoder weight
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False, infer=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = PipelinedWanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size, get_sp_group
            from OmniAvatar.distributed.xdit_context_parallel import usp_attn_forward
            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
            pipe.sp_group = get_sp_group()
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}

    @torch.no_grad()
    def denoising_inference(
        self,
        lat,            # (B, C, T, H, W)
        prompt_emb_posi=None,       # 新增：预编码的正向prompt embeddings
        prompt_emb_nega=None,       # 新增：预编码的负向prompt embeddings
        fixed_frame=0,  # Number of latent frames to keep fixed (for overlap mechanism)
        image_emb={},   # {"y": (B, C+1, T, H, W)}
        audio_emb={},   # {"audio_emb": (1, seq_len, hidden_size * layers)}
        cfg_scale=5.0,
        audio_cfg_scale=5.0,
        num_inference_steps=50,
        denoising_strength=1.0,
        sigma_shift=5.0,
        tea_cache_l1_thresh=None,
        tea_cache_model_id=""
    ):
        """只进行denoising inference，不进行VAE decoding，用于流水线并行"""
        # 开始计时
        self.start_timing("denoising_inference_only")
        
        # 阶段1: 初始化和准备
        self.start_timing("initialization")
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        latents = lat.clone()   # (B, C, T, H, W)
        latents = torch.randn_like(latents) # noisy latents, (B, C, T, H, W)
        init_time = self.end_timing("initialization")
        print(f"[Timing] Initialization: {init_time:.4f}s")
        
        # 阶段2: TeaCache初始化
        self.start_timing("tea_cache_init")
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None and tea_cache_l1_thresh > 0 else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None and tea_cache_l1_thresh > 0 else None}
        tea_cache_time = self.end_timing("tea_cache_init")
        print(f"[Timing] TeaCache initialization: {tea_cache_time:.4f}s")

        # 阶段3: 降噪推理
        self.start_timing("denoising_inference")
        denoising_times = []
        # 降噪循环
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            step_start_time = time.time()
            if fixed_frame > 0:
                latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]   # prefix clean latent
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **audio_emb, **tea_cache_posi)
            if cfg_scale != 1.0:
                audio_emb_uc = {}
                for key in audio_emb.keys():
                    audio_emb_uc[key] = torch.zeros_like(audio_emb[key])
                if audio_cfg_scale == cfg_scale:
                    noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, **audio_emb_uc, **tea_cache_nega)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    tea_cache_nega_audio = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None and tea_cache_l1_thresh > 0 else None}
                    audio_noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **audio_emb_uc, **tea_cache_nega_audio)
                    text_noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, **audio_emb_uc, **tea_cache_nega)
                    noise_pred = text_noise_pred_nega + cfg_scale * (audio_noise_pred_nega - text_noise_pred_nega) + audio_cfg_scale * (noise_pred_posi - audio_noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            # 记录每个step的耗时
            step_time = time.time() - step_start_time
            denoising_times.append(step_time)
        
        denoising_time = self.end_timing("denoising_inference")
        print(f"[Timing] Denoising inference: {denoising_time:.4f}s (avg step: {np.mean(denoising_times):.4f}s)")
            
        if fixed_frame > 0:
            latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]
        total_time = self.end_timing("denoising_inference_only")
        print(f"[Timing] Total denoising: {total_time:.4f}s")
        return latents

    @torch.no_grad()
    def vae_decoding(
        self,
        latents,        # (B, C, T, H, W) - denoised latents
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        self.start_timing("vae_decoding_standalone")
        # VAE解码
        frames = self.decode_video(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        # 后处理
        self.start_timing("post_processing")
        frames = (frames.permute(0, 2, 1, 3, 4).float() + 1) / 2
        post_time = self.end_timing("post_processing")
        vae_time = self.end_timing("vae_decoding_standalone")
        print(f"[Timing] VAE decoding standalone: {vae_time:.4f}s (post_processing: {post_time:.4f}s)")
        return frames

class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states
