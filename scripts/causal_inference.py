from tqdm import tqdm
from typing import List, Optional, Dict, Any
import os
import torch
import torch.nn as nn
from OmniAvatar.utils.args_config import parse_args
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
            
            input_values = np.squeeze(
                    self.wav_feature_extractor(audio, sampling_rate=16000).input_values
                )
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            ori_audio_len = audio_len = 81
            input_values = input_values.unsqueeze(0)
            
            with torch.no_grad():
                self.audio_encoder.to(self.device)
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
                audio_emb = audio_embeddings.permute(0, 2, 1)[:, :, :, None, None]
                audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2) # 1, 768, 44, 1, 1
                audio_emb = self.generator.audio_proj(audio_emb.to(self.dtype))
                audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.generator.audio_cond_projs], 0)
                print("audio_shape:",audio_emb.shape)
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
                import pdb; pdb.set_trace()
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
                audio_input=conditional_dict['audio'][:,current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
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
    print(f"model_manager model name: {model_manager.model_name}")
    # Fetch models using the fetch_model method
    dit_model = model_manager.fetch_model("causal_wan_video_dit")
    vae_model = model_manager.fetch_model("wan_video_vae")
    text_encoder_model = model_manager.fetch_model("wan_video_text_encoder")
    return model_manager, dtype

def main():
    """Main function to load models and test the causal inference pipeline."""
    torch.set_grad_enabled(False)
    args = parse_args()
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
    noise = torch.randn([1, 21, 16, 50, 90], device=device, dtype=torch.bfloat16)
    # Prepare text prompts
    text_prompts =getattr(args, 'prompt', "A realistic video of a man speaking directly to the camera on a sofa, with dynamic and rhythmic hand gestures that complement his speech. His hands are clearly visible, independent, and unobstructed. His facial expressions are expressive and full of emotion, enhancing the delivery. The camera remains steady, capturing sharp, clear movements and a focused, engaging presence")
    print(f"Noise tensor shape: {noise.shape}")
    print(f"Text prompts: {text_prompts}")
    
    # Perform causal inference
    print("Starting causal inference...")
    return_latents=False

    
    video=pipeline(noise=noise,
        text_prompts=text_prompts,
        image_path="/data2/jdsu/projects/OmniAvatar/examples/images/0000.jpeg",
        audio_path="/data2/jdsu/projects/OmniAvatar/examples/audios/0000.MP3",
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
    output_path = "generated_video_debug_fm_style.mp4"
    video_np = (video.squeeze(0).permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
    imageio.mimsave(output_path, video_np, fps=getattr(args, 'fps', 16))
    print(f"Video saved to: {output_path}")
    
    print("Causal inference completed successfully!")

if __name__ == "__main__":
    main()
