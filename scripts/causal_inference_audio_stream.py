from tqdm import tqdm
from typing import List, Optional, Dict, Any
import os
import torch
import torch.nn as nn
from OmniAvatar.utils.args_config import parse_args
args = parse_args()
import math
import numpy as np
import librosa
import torchvision.transforms as TT
#from OmniAvatar.schedulers.flow_match import FlowDPMSolverMultistepScheduler
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.prompters import WanPrompter
# Import match_size function from scripts.inference
from scripts.inference import match_size,resize_pad
from OmniAvatar.schedulers.flow_match import FlowMatchScheduler
from transformers import Wav2Vec2FeatureExtractor
import subprocess



class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            model_manager: ModelManager,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        self.args = args
        self.device = device
        if args.dtype=='bf16':
            self.dtype = torch.bfloat16
        # Initialize models from model_manager
        self.generator = model_manager.fetch_model("causal_wan_video_dit") if generator is None else generator
        self.generator.eval()
        self.text_encoder = model_manager.fetch_model("wan_video_text_encoder") if text_encoder is None else text_encoder
        self.vae = model_manager.fetch_model("wan_video_vae") if vae is None else vae
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True,num_inference_steps=4)
        from OmniAvatar.models.wav2vec import Wav2VecModel
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                args.wav2vec_path
            )
        self.audio_encoder = Wav2VecModel.from_pretrained(args.wav2vec_path, local_files_only=True).to(device=self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.generator.to("cuda")
    
        # Initialize prompter for text encoding
        self.prompter = WanPrompter()
        if self.text_encoder is not None:
            self.prompter.fetch_models(self.text_encoder)
            # Initialize tokenizer (similar to wan_video.py:131)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(args.text_encoder_path), "google/umt5-xxl"))
        # Initialize Image transform
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        self.transform = TT.Compose(chained_trainsforms)
        
        # Scheduler configuration
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        self.denoising_step_list = torch.linspace(0, 1000, steps=5, dtype=torch.long)[1:].flip(0)
        if args.warp_denoising_step:
           
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[:-1].to(dtype=self.dtype,device=self.device)
            #self.denoising_step_list = timesteps[1000 - self.denoising_step_list].to(dtype=self.dtype,device=self.device)

        # Causal configuration
        self.num_transformer_blocks = getattr(args, 'num_transformer_blocks', 30)
        self.frame_seq_length = getattr(args, 'frame_seq_length', 1560)
        self.num_frame_per_block = getattr(args, 'num_frame_per_block', 3)
        self.independent_first_frame = getattr(args, 'independent_first_frame', False)
        self.local_attn_size = getattr(self.generator, 'local_attn_size', -1)
        
        # Cache initialization
        self.kv_cache1=None
        
        print(f"Causal KV inference with {self.num_frame_per_block} frames per block")
        self.generator.num_frame_per_block = self.num_frame_per_block
        self.generator.local_attn_size=self.local_attn_size
        self.generator.independent_first_frame=self.independent_first_frame

    def encode_audio_with_sliding_window(self, audio, sr=16000, num_target_frames=84,
                                          frames_per_step=12, context_frames=3, h=1):
        """
        使用滑动窗口和overlap-discard策略编码音频（流式友好）

        Args:
            audio: 输入音频数组 [audio_length]
            sr: 采样率，默认16000
            num_target_frames: 目标视频帧数，默认84（21块×4帧，能被12整除）
            frames_per_step: 每个块输出的帧数，默认12（对应16fps下12帧时长）
            context_frames: 左右上下文帧数，默认3（确保覆盖编码器感受野）
            h: 插值核半径，默认1（线性插值）

        Returns:
            audio_embeddings: 编码后的音频特征 [1, num_target_frames, 10752]
        """
        fps = args.fps  # 视频帧率
        frame_duration = 1.0 / fps  # 每帧时长（秒）
        audio_length = len(audio)

        # 计算需要多少块（块0特殊处理，只编码9帧；后续块正常12帧）
        # 块0: 12帧输出（帧0复制3次 + 帧1-8）
        # 块1-6: 每块12帧（帧9-20, 21-32, ..., 69-80）
        num_blocks = 7  # 固定7块，共84帧

        all_block_embeddings = []

        for block_idx in range(num_blocks):
            if block_idx == 0:
                # 第0块特殊处理：只编码帧0-8（9帧），但输出12帧
                start_frame = 0
                end_frame = 9  # 只到帧8
                actual_frames = 9  # 实际只编码9帧

                # 添加上下文帧（第0块只有右侧上下文，用于overlap-discard）
                start_frame_with_context = 0  # 不需要左侧上下文
                end_frame_with_context = end_frame + h  # 只需要h个右侧帧用于discard
            else:
                # 后续块正常处理
                # 块1对应帧9-20，块2对应帧21-32，...
                start_frame = 9 + (block_idx - 1) * frames_per_step  # 9, 21, 33, 45, 57, 69
                end_frame = start_frame + frames_per_step  # 21, 33, 45, 57, 69, 81
                actual_frames = frames_per_step

                # 判断是否是最后一块
                is_last = (block_idx == num_blocks - 1)

                # 添加上下文帧
                start_frame_with_context = max(0, start_frame - h)  # 左侧需要h帧用于discard
                if is_last:
                    end_frame_with_context = end_frame  # 最后一块不需要右侧上下文
                else:
                    end_frame_with_context = end_frame + h  # 中间块需要h个右侧帧用于discard

            # 计算对应的音频范围（采样点）
            start_sample = int(start_frame_with_context * frame_duration * sr)
            end_sample = int(end_frame_with_context * frame_duration * sr)
            start_sample = min(start_sample, audio_length)
            end_sample = min(end_sample, audio_length)

            # 提取音频片段
            audio_segment = audio[start_sample:end_sample]

            # 如果音频片段太短，进行填充
            min_audio_length = int(0.1 * sr)  # 至少0.1秒
            if len(audio_segment) < min_audio_length:
                audio_segment = np.pad(audio_segment, (0, min_audio_length - len(audio_segment)),
                                       mode='constant', constant_values=0)

            # 编码音频片段
            input_values = np.squeeze(
                self.wav_feature_extractor(audio_segment, sampling_rate=sr).input_values
            )
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            input_values = input_values.unsqueeze(0)

            # 根据边界情况确定目标序列长度和裁剪方式
            is_first_block = (block_idx == 0)
            is_last_block = (block_idx == num_blocks - 1)

            if is_first_block:
                # 第0块：没有左侧上下文，只需右侧h帧用于discard
                target_seq_len = actual_frames + h
            elif is_last_block:
                # 最后一块：没有右侧上下文，只需左侧h帧用于discard
                target_seq_len = actual_frames + h
            else:
                # 中间块：左右各需要h帧用于discard
                target_seq_len = actual_frames + 2 * h

            with torch.no_grad():
                self.audio_encoder.to(self.device)
                # 直接在audio_encoder中指定seq_len，让编码器内部完成插值
                hidden_states = self.audio_encoder(input_values, seq_len=target_seq_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)

            # audio_embeddings shape: (1, target_seq_len, 10752)
            # Overlap-discard: 根据边界情况选择性丢弃
            if h > 0:
                if is_first_block:
                    # 第0块：只丢弃右侧h帧
                    audio_embeddings_trimmed = audio_embeddings[:, :-h, :]  # (1, actual_frames, 10752)
                elif is_last_block:
                    # 最后一块：只丢弃左侧h帧
                    audio_embeddings_trimmed = audio_embeddings[:, h:, :]  # (1, actual_frames, 10752)
                else:
                    # 中间块：丢弃前后各h帧
                    audio_embeddings_trimmed = audio_embeddings[:, h:-h, :]  # (1, actual_frames, 10752)
            else:
                audio_embeddings_trimmed = audio_embeddings

            # 第0块特殊处理：将第0帧复制3次，使其输出12帧
            if block_idx == 0:
                # audio_embeddings_trimmed shape: (1, 9, 10752)
                # 复制第0帧3次: [feat_0, feat_0, feat_0, feat_0, feat_1, ..., feat_8]
                first_frame = audio_embeddings_trimmed[:, 0:1, :]  # (1, 1, 10752)
                repeated_first_frame = first_frame.repeat(1, 3, 1)  # (1, 3, 10752)
                # 拼接：3个重复 + 原始9帧 = 12帧
                audio_embeddings_trimmed = torch.cat([repeated_first_frame, audio_embeddings_trimmed], dim=1)
                print(f"Block {block_idx} (special): frames 0-11 (frame 0 repeated 3 times), "
                      f"audio samples {start_sample}-{end_sample}, "
                      f"output shape {audio_embeddings_trimmed.shape}")
            else:
                print(f"Block {block_idx}: frames {start_frame}-{end_frame-1}, "
                      f"audio samples {start_sample}-{end_sample}, "
                      f"output shape {audio_embeddings_trimmed.shape}")

            all_block_embeddings.append(audio_embeddings_trimmed)

        # 拼接所有块
        stacked_audio_embeddings = torch.cat(all_block_embeddings, dim=1)  # (1, num_target_frames, 10752)
        print(f"Final stacked audio embeddings shape: {stacked_audio_embeddings.shape}")

        return stacked_audio_embeddings

    def forward(
        self,
        noise: torch.Tensor,
        text_prompts: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
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
        # Initialize conditioning based on available inputs
        # For now, we use the same conditioning as inference method
        # This can be extended to handle image_path and audio_path when needed
        
        #prepare text_prompts
        
        text_prompts=text_prompts
        #prepare image_condition
        if image_path is not None:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
            self.vae.to(self.device)
            img_lat = self.vae.encode(videos=image.to(dtype=self.dtype),device=self.device).repeat(1,1,21,1,1)
            msk = torch.zeros_like(img_lat)[:,:1]
            msk[:, :, 1:] = 1
            img_lat = torch.cat([img_lat, msk], dim=1)
            print("img_lat:",img_lat.shape)
            
        #prepare audio_condition
        if audio_path is not None:
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)

            # Trim audio to 5 seconds
            max_duration = 5.0  # 5 seconds
            max_samples = int(max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                print(f"Audio trimmed to {max_duration} seconds")

            # Use sliding window encoding with overlap-discard strategy
            # This simulates streaming audio generation with 12-frame steps
            # Block 0 repeats frame 0 three times, so we get 84 frames directly
            # h=3 means we interpolate with larger margin and discard 3 frames on each side
            stacked_audio_embeddings = self.encode_audio_with_sliding_window(
                audio=audio,
                sr=sr,
                num_target_frames=84,
                frames_per_step=12,
                context_frames=8,
                h=8
            )  # Returns: (1, 84, 10752)
            print(f"Stacked audio_embeddings shape: {stacked_audio_embeddings.shape}")

            # Process the stacked embeddings (no manual duplication needed)
            audio_emb = stacked_audio_embeddings.permute(0, 2, 1)[:, :, :, None, None]  # (1, 10752, 84, 1, 1)
            
            # Group audio_emb into 21 chunks, each containing 4 frames along dimension 2
            batch_size, feature_dim, total_frames, h, w = audio_emb.shape  # (1, 10752, 84, 1, 1)
            chunk_size = 4
            num_chunks = total_frames // chunk_size  # 84 // 4 = 21
            
            # Reshape to group every 4 frames: (1, 10752, 21, 4, 1, 1)
            audio_emb_chunked = audio_emb[:, :, :num_chunks*chunk_size].view(
                batch_size, feature_dim, num_chunks, chunk_size, h, w
            )
            print(f"Audio chunked shape: {audio_emb_chunked.shape}")  # (1, 10752, 21, 4, 1, 1)
            
            # Process each chunk through the projection layers
            audio_emb_processed_chunks = []
            for i in range(num_chunks):
                chunk = audio_emb_chunked[:, :, i]  # (1, 10752, 4, 1, 1)
                # Apply audio projection
                chunk_proj = self.generator.audio_proj(chunk.to(self.dtype))
                chunk_final = torch.concat([audio_cond_proj(chunk_proj) for audio_cond_proj in self.generator.audio_cond_projs], 0)
                audio_emb_processed_chunks.append(chunk_final)
                #print(f"Chunk {i} processed shape: {chunk_final.shape}")
            
            # Stack all processed chunks: (21, projected_dim, 4, 1, 1)
            audio_emb = torch.stack(audio_emb_processed_chunks, dim=0)
            print("Final chunked audio_shape:", audio_emb.shape)
                    
        else:
            print("Detect No audio input!!")
            audio_embeddings = None
          
        return self.inference(
            noise=noise,
            text_prompts=text_prompts,
            img_lat=img_lat,
            audio_embed=audio_emb,
            initial_latent=initial_latent,
            return_latents=return_latents
        )

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: str,
        img_lat:torch.Tensor,
        audio_embed:torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
      
    ) -> torch.Tensor:
        """
        Perform causal inference.
        
        Args:
            noise: Input noise tensor [batch_size, num_output_frames, channels, height, width]
            text_prompts: List of text prompts
            initial_latent: Initial latent for I2V [batch_size, num_input_frames, channels, height, width]
            return_latents: Whether to return latents
            start_frame_index: Starting frame index for long video generation
            guidance_scale: CFG scale for text conditioning
            
        Returns:
            Generated video tensor [batch_size, num_frames, channels, height, width]
        """
        with torch.no_grad():
            batch_size, num_frames, num_channels, height, width = noise.shape
        
        
            # Frame block calculations
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
            
            num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
            num_output_frames = num_frames + num_input_frames
            
            # Text conditioning
            self.text_encoder.to("cuda")
            conditional_dict = self._encode_text_prompts(text_prompts, positive=True)
            conditional_dict['image']=img_lat
            conditional_dict['audio']=audio_embed
            
            output = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=noise.device,
                dtype=noise.dtype
            )
            
            # Step 1: Initialize KV caches
            self._setup_caches(batch_size, noise.dtype, noise.device)
            
            # Step 2: Cache context feature
            current_start_frame = 0
            if initial_latent is not None:
                print("INITIAL_LATENT is not None!!")
                timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
                #independent_first_frame=false
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

                for _ in range(num_input_blocks):
                    current_ref_latents = \
                        initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                    output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                    self._generator_forward(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    current_start_frame += self.num_frame_per_block
           
            # Step 3: Temporal denoising loop
            all_num_frames = [self.num_frame_per_block] * num_blocks
            
            for current_num_frames in all_num_frames:
                print(f"Processing frame {current_start_frame - num_input_frames} to {current_start_frame + current_num_frames - num_input_frames}.")
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                y_input=conditional_dict['image'][:,:,current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                #audio_input=conditional_dict['audio'][:,current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                audio_input=conditional_dict['audio'][current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames].squeeze(-2).permute(1, 0, 2, 3, 4)
              
                block_conditional_dict = conditional_dict.copy()
                block_conditional_dict.update(image=y_input.clone(), audio=audio_input.clone())
                
              
                # Step 3.1: Spatial denoising loop
                for index, current_timestep in enumerate(self.denoising_step_list):
                    #print(current_timestep)
                    if current_start_frame==0:
                        noisy_input[:, :1] = img_lat[:, :16, :1].permute(0,2,1,3,4)
                
                    timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                 
                    v, denoised_pred = self._generator_forward(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=block_conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    
                    if index < len(self.denoising_step_list) - 1:
                        
                        next_timestep = self.denoising_step_list[index + 1]

                        #noisy_input = self.scheduler.step(v, current_timestep,noisy_input)
                        
                        
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                        
                        
                # Step 3.2: record the model's output
                if current_start_frame==0:
                        denoised_pred[:, :1] = img_lat[:, :16, :1].permute(0,2,1,3,4)
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                # Step 3.3: rerun with timestep zero to update KV cache using clean context
                context_timestep = torch.ones_like(timestep) * 0
                self._generator_forward(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=block_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                # Step 3.4: update the start and end frame indices
                current_start_frame += current_num_frames
            
            # Decode to video
            output=output.permute(0,2,1,3,4)
            self.vae.to(output.device)
            video = self.vae.decode(output,device=self.device)
            video = (video * 0.5 + 0.5).clamp(0, 1)
        
        if return_latents:
            return video, output
        else:
            return video

    def _encode_text_prompts(self, text_prompts: str, positive: bool = True) -> Dict[str, torch.Tensor]:
        """Encode text prompts using prompter (similar to wan_video.py)."""
        prompt_emb = self.prompter.encode_prompt(text_prompts, positive=positive, device=self.device)
        return {"prompt_embeds": prompt_emb}
    
    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Convert flow matching's prediction to x0 prediction."""
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)
    
    def _generator_forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        kv_cache=None,
        crossattn_cache=None,
        current_start=None,
        cache_start=None
    ):
        """Wrapper function that calls generator and converts flow_pred to x0_pred."""
        prompt_embeds = conditional_dict["prompt_embeds"]
        y=conditional_dict['image'].to(device=noisy_image_or_video.device,dtype=noisy_image_or_video.dtype)
        audio_emb=conditional_dict['audio'].to(device=noisy_image_or_video.device,dtype=noisy_image_or_video.dtype)
        # Call the model to get flow prediction
        flow_pred = self.generator(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            timestep=timestep,
            context=prompt_embeds,
            y=y,
            audio_emb=audio_emb,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start
        ).permute(0, 2, 1, 3, 4)
        
        # Convert flow prediction to x0 prediction
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])
        
        return flow_pred, pred_x0


    def _setup_caches(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize or reset KV and cross-attention caches."""
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size, dtype, device)
            self._initialize_crossattn_cache(batch_size, dtype, device)
        else:
            self._reset_caches(device)

    def _reset_caches(self, device: torch.device):
        """Reset existing caches for new inference."""
        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache[block_index]["is_init"] = False
        # reset kv cache
        for block_index in range(len(self.kv_cache1)):
            self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device)
            self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device)

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize KV cache for causal attention."""
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12*128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12*128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize cross-attention cache."""
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12*128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12*128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

  

    @classmethod
    def from_model_manager(
        cls,
        model_manager: ModelManager,
        args,
        device,
        **kwargs
    ):
        """Create pipeline from ModelManager (similar to WanVideoPipeline interface)."""
        return cls(
            args=args,
            device=device,
            model_manager=model_manager,
            **kwargs
        )


def load_models(args):
    """Load DiT, VAE, and text encoder models using ModelManager."""
    # Set dtype
    if args.dtype == 'bf16':
        dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        dtype = torch.float16
    else:   
        dtype = torch.float32
    
    print(f"Loading models with dtype: {dtype}")
    # Initialize model manager
    model_manager = ModelManager(device="cpu", infer=True)
    
    # Load models following the inference.py pattern
    model_manager.load_models(
        [
            args.dit_path.split(","),
            args.text_encoder_path,
            args.vae_path
        ],
        torch_dtype=dtype,
        device='cpu',
    )
    
    # Fetch models using the fetch_model method
    dit_model = model_manager.fetch_model("causal_wan_video_dit")
    vae_model = model_manager.fetch_model("wan_video_vae") 
    text_encoder_model = model_manager.fetch_model("wan_video_text_encoder")
    
    return model_manager, dtype


def main():
    """Main function to load models and test the causal inference pipeline."""
    torch.set_grad_enabled(False)
    # Set device based on rank (for distributed inference compatibility)
    device = torch.device(f"cuda:{getattr(args, 'rank', 0)}")
    # Load models
    model_manager, dtype = load_models(args)
    # Create causal inference pipeline
    pipeline = CausalInferencePipeline.from_model_manager(
        model_manager=model_manager,
        args=args,
        device=device
    )
    print("Causal inference pipeline initialized successfully!")
    
    print("Preparing dummy noise……")
    noise = torch.randn(
            [1, 21, 16, 50, 90], device=device, dtype=torch.bfloat16
        )
   
   
    # Prepare text prompts
    text_prompts =getattr(args, 'prompt', "A realistic video of a man speaking directly to the camera on a sofa, with dynamic and rhythmic hand gestures that complement his speech. His hands are clearly visible, independent, and unobstructed. His facial expressions are expressive and full of emotion, enhancing the delivery. The camera remains steady, capturing sharp, clear movements and a focused, engaging presence")
    #text_prompts =getattr(args, 'prompt',"a man talking.")
    #text_prompts =getattr(args, 'prompt',"a woman talking.")
    #text_prompts="A man with a bald head and short beard, wearing a dark suit over a light blue shirt, addresses the camera directly. His expressions shift subtly as he speaks, conveying thoughtful engagement. Framed closely against a neutral background, the high-definition footage emphasizes his clear communication and calm demeanor."
    print(f"Noise tensor shape: {noise.shape}")
    print(f"Text prompts: {text_prompts}")
    
    # Perform causal inference
    print("Starting causal inference...")
    return_latents=False

    audio_path="/inspire/hdd/global_user/liupengfei-24025/ethan/repos/self-forcing-gair/hallo3_data/audio/0a81b5f45b88b89593bd89dcaf600d87.wav"
    video=pipeline(noise=noise,
        text_prompts=text_prompts,
        image_path="/inspire/hdd/global_user/liupengfei-24025/ethan/repos/self-forcing-gair/hallo3_data/first_frames_qwen_high_quality/0a81b5f45b88b89593bd89dcaf600d87.png",
        audio_path=audio_path,
        initial_latent=None,
        return_latents=return_latents)
    '''
    video = pipeline.inference(
        noise=noise,
        text_prompts=text_prompts,
        initial_latent=None,
        return_latents=return_latents
    )
    '''
    print(f"Generated video shape: {video.shape}")
    
    # Save generated video
    import imageio
    output_path = "generated_video_audio_stream.mp4"
    video_np = (video.squeeze(0).permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
    print(video_np.shape)#（81,400,720,3）
    imageio.mimsave(
    "tmp.mp4",
    video_np,                       # (T,H,W,3) uint8 0-255
    fps=args.fps,
    codec="libx264",
    macro_block_size=None,          # 避免对齐引起的缩放
    ffmpeg_params=[
        "-crf", "18",               # 18更清晰；可改 20/22/24 找平衡
        "-preset", "veryfast",      # 编码速度/效率权衡：ultrafast..placebo
        "-pix_fmt", "yuv420p"       # 兼容性最好
    ]
)
    
    
    cmd = [
    "ffmpeg", "-y",
    "-i", "tmp.mp4",            # 无声视频
    "-i", audio_path,           # 原始音频（16 kHz mono）
    "-map", "0:v:0", "-map", "1:a:0",
    "-c:v", "copy",             # 不重编码视频
    "-c:a", "aac",              # AAC-LC
    "-ar", "48000",             # 上采样到 48 kHz（避免每帧比特上限告警）
    "-ac", "1",                 # 单声道（需要立体声可改 2）
    "-b:a", "96k",              # 常用语音码率；128k 也可
    "-movflags", "+faststart",  # 网页首开更快（可选）
    "-shortest",
    output_path
]
    subprocess.run(cmd, check=True)

    print(f"Video saved to: {output_path}")
    
    print("Causal inference completed successfully!")
        
   
    
    
if __name__ == "__main__":
    main()
