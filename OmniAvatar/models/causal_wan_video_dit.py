from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from einops import rearrange
from typing import Tuple, Optional
from argparse import Namespace
import time

from .audio_pack import AudioPack
from .wan_video_dit import (
    RMSNorm,
    rope_apply,
    AttentionModule,
    CrossAttention,
    GateModule,
    modulate,
    precompute_freqs_cis_3d,
    MLP,
    sinusoidal_embedding_1d,
    flash_attention
)
from ..utils.args_config import parse_args # 有空可以把这个改了。
args = parse_args()

# import pdb; pdb.set_trace()

# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)

class CacheCrossAttention(CrossAttention):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__(dim, num_heads, eps, has_image_input)  # 复用父类的层与属性

    def forward(self, x: torch.Tensor, y: torch.Tensor,crossattn_cache):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
            
        q = self.norm_q(self.q(x))
        
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(ctx))
                v = self.v(ctx)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        
        else:
        
            k = self.norm_k(self.k(ctx))
            v = self.v(ctx)
            
        x = self.attn(q, k, v)
        
        if self.has_image_input: #We do need to care about k_img. Because even i2v will not go through this code branch
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
            
        return self.o(x)





class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6,local_attn_size=-1,sink_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
    
        self.attn = AttentionModule(self.num_heads)
        
        
        self.local_attn_size=local_attn_size
        self.sink_size=sink_size
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        
    def forward(self, x, freqs,block_mask=None,grid_sizes=None,kv_cache=None,current_start=0,cache_start=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        
        if kv_cache is None: #We do not consider teacher forcing here
            roped_query = rope_apply(q, freqs,self.num_heads).type_as(v)
            roped_key = rope_apply(k, freqs,self.num_heads).type_as(v)
            
            B, T, D = q.shape
            q = q.view(B, T, self.num_heads, self.head_dim)   
            roped_query=roped_query.view(B, T, self.num_heads, self.head_dim) 
            k = k.view(B, T, self.num_heads, self.head_dim)      
            roped_key=roped_key.view(B,T,self.num_heads,self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)       
            
            
            #pad至128的倍数
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                    torch.zeros([q.shape[0], padded_length, q.shape[2],q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2],k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2],v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )
            x=x[:, :, :x.shape[2]-padded_length].transpose(2, 1).reshape(B, T, -1) #这里与源代码不同防止pad=0变成空切片
        else:
            self_attn_start = time.time()
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            process_end = time.time()
            print(f"[TIMING] self attn forward process: {process_end - self_attn_start:.6f}s")

            ##Freqs here are causal.
            roped_query = rope_apply(q, freqs,self.num_heads).type_as(v)
            roped_key = rope_apply(k, freqs,self.num_heads).type_as(v)
            rope_end = time.time()
            print(f"[TIMING] self attn forward rope apply: {rope_end - process_end:.6f}s")
            
            
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            update_kv_cache_start = time.time()
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                update_kv_cache_end = time.time()
                print(f"[TIMING] self attn forward update kv cache (if): {update_kv_cache_end - update_kv_cache_start:.6f}s")
            else:
                # Assign new keys/values directly up to current_end
                # local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"]
                # local_end_index = local_end_index.item()
                # import pdb; pdb.set_trace()
                end1 = time.time()
                print(f"[TIMING] self attn forward update kv cache (else) time1: {end1 - update_kv_cache_start:.6f}s")
                local_start_index = local_end_index - num_new_tokens
                # local_start_index = local_start_index.item()
                end2 = time.time()
                print(f"[TIMING] self attn forward update kv cache (else) time2: {end2 - end1:.6f}s")
    
                # print(f"kv_cache_v: {kv_cache['v'].shape}, {kv_cache['v'].device}")
                # print(f"v: {v.shape}, {v.device}")
                kv_cache["v"][:, local_start_index:local_end_index] = v
                end3 = time.time()
                print(f"[TIMING] self attn forward update kv cache (else) time3: {end3 - end2:.6f}s")
                # print(f"kv_cache_k: {kv_cache['k'].shape}, {kv_cache['k'].device}")
                # print(f"roped_key: {roped_key.shape}, {roped_key.device}")
                # print(f"local_start_index: {local_start_index}, local_end_index: {local_end_index}")
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                end4 = time.time()
                print(f"[TIMING] self attn forward update kv cache (else) time4: {end4 - end3:.6f}s")
                update_kv_cache_end = time.time()
                print(f"[TIMING] self attn forward update kv cache (else): {update_kv_cache_end - update_kv_cache_start:.6f}s")
            attn_start = time.time()
            x = self.attn(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            attn_end = time.time()
            print(f"[TIMING] self attn forward attn: {attn_end - attn_start:.6f}s")
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index[0])
            final_kv_cache_end = time.time()
            print(f"[TIMING] self attn forward final kv cache: {final_kv_cache_end - attn_end:.6f}s")
        
        #应该不需要flatten
        x = self.o(x)
        output_end = time.time()
        print(f"[TIMING] self attn forward output layer: {output_end - final_kv_cache_end:.6f}s")
        return x

    
class CausalDiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6,local_attn_size=-1,sink_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.local_attn_size=local_attn_size
        
        self.self_attn = CausalSelfAttention(dim, num_heads, eps,local_attn_size=local_attn_size,sink_size=sink_size)
        self.cross_attn = CacheCrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs,block_mask=None,grid_sizes=None,kv_cache=None,crossattn_cache=None,current_start=0,cache_start=None):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
         #Note: 这里 t_mod 拓展成（b,f,6,dim） 
        
        forward_start = time.time()
        num_frames,frame_seqlen=t_mod.shape[1],x.shape[1]//t_mod.shape[1]
         
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.unsqueeze(1).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=2)
        
        input_x = modulate(self.norm1(x).unflatten(dim=1,sizes=(num_frames,frame_seqlen)), shift_msa, scale_msa).flatten(1,2)
        process_end = time.time()
        # print(f"[TIMING] block forward process: {process_end - forward_start:.6f}s")
        y=self.self_attn(input_x, freqs,grid_sizes=grid_sizes,block_mask=block_mask,kv_cache=kv_cache,current_start=current_start,cache_start=cache_start)
        self_attn_end = time.time()
        print(f"[TIMING] block forward self attention: {self_attn_end - process_end:.6f}s")
        #x = self.gate(x, gate_msa,y.unflatten(dim=1,sizes=(num_frames,frame_seqlen)))
        x=x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(1, 2)
        unflatten_end = time.time()
        # print(f"[TIMING] block forward unflatten: {unflatten_end - self_attn_end:.6f}s")
        
        x = x + self.cross_attn(self.norm3(x), context,crossattn_cache)
        cross_attn_end = time.time()
        # print(f"[TIMING] block forward cross attention: {cross_attn_end - unflatten_end:.6f}s")
        
        input_x = modulate(self.norm2(x).unflatten(dim=1,sizes=(num_frames,frame_seqlen)), shift_mlp, scale_mlp).flatten(1,2)
        #x = self.gate(x, gate_mlp, self.ffn(input_x))
        x = x+ (self.ffn(input_x).unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * gate_mlp).flatten(1, 2)
        
        return x    
    

class CausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        #T_mod (B,F,1,C)
        num_frames, frame_seqlen = t_mod.shape[1], x.shape[1] // t_mod.shape[1]
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device).unsqueeze(1) + t_mod).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + scale) + shift))
        return x
    


class CausalWanModel(torch.nn.Module): #note:目前训练分支传参等还需优化。 #inference的freqs优化
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        audio_hidden_size: int=32,
        local_attn_size=-1,  # CausalWanModel新增
        sink_size=0  # CausalWanModel新增
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
            # nn.LayerNorm(dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            CausalDiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,local_attn_size=local_attn_size,sink_size=sink_size)
            for _ in range(num_layers)
        ])
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280
            
        self.block_mask=None
        
        # 添加频率缓存机制
        self.freq_cache = {}
        self.max_cache_size = 100  # 限制缓存大小
        
        # 预计算常用的频率组合以进一步优化
        self._precompute_common_freqs()
        
        self.num_frame_per_block=1
        
        self.independent_first_frame=False
        # import pdb; pdb.set_trace()
        if 'use_audio' in args:
            self.use_audio = args.use_audio
        else:
            self.use_audio = False
        if self.use_audio:
            audio_input_dim = 10752
            audio_out_dim = dim
            self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True)
            self.audio_cond_projs = nn.ModuleList()
            for d in range(num_layers // 2 - 1):
                l = nn.Linear(audio_hidden_size, audio_out_dim)
                self.audio_cond_projs.append(l)      
                
    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
        
    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)
    
    def _forward_train(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                clean_x: Optional[torch.Tensor]=None,
                use_gradient_checkpointing: bool = False,
                audio_emb: Optional[torch.Tensor] = None,
                use_gradient_checkpointing_offload: bool = False,
                tea_cache = None,
                **kwargs,
                ):
        
        # ===== TIMING: Start of forward_train =====
        start_time = time.time()
        print(f"[TIMING] _forward_train started at {start_time:.6f}")
        
        device = self.patch_embedding.weight.device
        
        # ===== TIMING: Block mask preparation =====
        mask_start = time.time()
        
        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                pass # Note: We do need teacher forcing now.
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
        
        mask_end = time.time()
        print(f"[TIMING] Block mask preparation: {mask_end - mask_start:.6f}s")
        
        # ===== TIMING: Patch embedding =====
        patch_start = time.time()
        
        lat_h, lat_w = x.shape[-2], x.shape[-1]
        x = torch.cat([x, y], dim=1)
        x = self.patch_embedding(x)
        x, (f, h, w) = self.patchify(x)
        grid_sizes = torch.tensor([f, h, w]).unsqueeze(0) 
        
        patch_end = time.time()
        print(f"[TIMING] Patch embedding: {patch_end - patch_start:.6f}s")
        
        # ===== TIMING: Time embeddings =====
        time_emb_start = time.time()
        
        #time embeddings
        #timestep.shape (B,F)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=timestep.shape)
        #assert t.dtype == torch.float32 and t_mod.dtype == torch.float32
        
        #context
        context = self.text_embedding(context)
        
        time_emb_end = time.time()
        print(f"[TIMING] Time & text embeddings: {time_emb_end - time_emb_start:.6f}s")
        
        # ===== TIMING: Audio processing =====
        audio_start = time.time()
        
        if audio_emb != None and self.use_audio: # TODO  cache
            audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
            audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2) # 1, 768, 44, 1, 1
            audio_emb = self.audio_proj(audio_emb)

            audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.audio_cond_projs], 0)
        
        audio_end = time.time()
        print(f"[TIMING] Audio processing: {audio_end - audio_start:.6f}s")
        
        # ===== TIMING: Frequency preparation =====
        freq_start = time.time()
        
        # arguments
        kwargs=dict(
            grid_sizes=grid_sizes,
            block_mask=self.block_mask
        )
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        freq_end = time.time()
        print(f"[TIMING] Frequency preparation: {freq_end - freq_start:.6f}s")
        
        # ===== TIMING: Main transformer blocks =====
        blocks_start = time.time()
        
        def create_custom_forward(module):
            def custom_forward(*inputs,**kwargs):
                return module(*inputs,**kwargs)
            return custom_forward
        
        if tea_cache is not None:
            tea_cache_update = tea_cache.check(self, x, t_mod)
        else:
            tea_cache_update = False
        ori_x_len = x.shape[1]
        if tea_cache_update:
            x = tea_cache.update(x)
        else:
            if args.sp_size > 1:
                # Context Parallel
                sp_size = get_sequence_parallel_world_size()
                pad_size = 0
                if ori_x_len % sp_size != 0:
                    pad_size = sp_size - ori_x_len % sp_size
                    x = torch.cat([x, torch.zeros_like(x[:, -1:]).repeat(1, pad_size, 1)], 1)
                x = torch.chunk(x, sp_size, dim=1)[get_sequence_parallel_rank()]

            audio_emb = audio_emb.reshape(x.shape[0], audio_emb.shape[0] // x.shape[0], -1, *audio_emb.shape[2:])
            
            
            for layer_i, block in enumerate(self.blocks):
                # audio cond
                if self.use_audio:
                    au_idx = None
                    if (layer_i <= len(self.blocks) // 2 and layer_i > 1): # < len(self.blocks) - 1:
                        au_idx = layer_i - 2
                        audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, lat_h // 2, lat_w // 2, 1) # 1, 11, 45, 25, 128
                        audio_cond_tmp = self.patchify(audio_emb_tmp.permute(0, 4, 1, 2, 3))[0]
                        if args.sp_size > 1:
                            if pad_size > 0:    
                                audio_cond_tmp = torch.cat([audio_cond_tmp, torch.zeros_like(audio_cond_tmp[:, -1:]).repeat(1, pad_size, 1)], 1)
                            audio_cond_tmp = torch.chunk(audio_cond_tmp, sp_size, dim=1)[get_sequence_parallel_rank()]
                        x = audio_cond_tmp + x

                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x, context, t_mod, freqs,**kwargs,
                                use_reentrant=False,
                            )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,**kwargs,
                            use_reentrant=False,
                        )
                else:
                    x = block(x, context, t_mod, freqs,**kwargs)
                    
            blocks_end = time.time()
            print(f"[TIMING] All transformer blocks: {blocks_end - blocks_start:.6f}s")
            
            # ===== TIMING: Cache operations =====
            cache_start = time.time()
            
            if tea_cache is not None:
                x_cache = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
                x_cache = x_cache[:, :ori_x_len]
                tea_cache.store(x_cache)

        cache_end = time.time()
        print(f"[TIMING] Cache operations: {cache_end - cache_start:.6f}s")
        
        # ===== TIMING: Head and unpatchify =====
        head_start = time.time()

        x = self.head(x, t.unflatten(dim=0,sizes=timestep.shape).unsqueeze(2))
        if args.sp_size > 1:
            # Context Parallel
            x = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
            x = x[:, :ori_x_len]
        x=x.reshape(x.shape[0],-1,x.shape[-1])
        x = self.unpatchify(x, (f, h, w))
        
        head_end = time.time()
        print(f"[TIMING] Head and unpatchify: {head_end - head_start:.6f}s")
        
        # ===== TIMING: Total forward_train time =====
        total_end = time.time()
        print(f"[TIMING] _forward_train TOTAL: {total_end - start_time:.6f}s")
        print("=" * 60)
        
        return x
    
    def _forward_inference(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                clean_x: Optional[torch.Tensor]=None,
                use_gradient_checkpointing: bool = False,
                audio_emb: Optional[torch.Tensor] = None,
                use_gradient_checkpointing_offload: bool = False,
                tea_cache = None,
                kv_cache: dict = None,
                crossattn_cache: dict = None,
                current_start: int = 0,
                cache_start: int = 0,
                **kwargs,
                ):
    
        # ===== TIMING: Start of forward_inference =====
        start_time = time.time()
        print(f"[TIMING] _forward_inference started at {start_time:.6f}")
        
        device = self.patch_embedding.weight.device
        
        # ===== TIMING: Patch embedding =====
        patch_start = time.time()
        
        lat_h, lat_w = x.shape[-2], x.shape[-1]
        x = torch.cat([x, y], dim=1)
        x = self.patch_embedding(x)
        x, (f, h, w) = self.patchify(x)
        
        grid_sizes = torch.tensor([f, h, w]).unsqueeze(0)
        
        patch_end = time.time()
        # print(f"[TIMING] Patch embedding: {patch_end - patch_start:.6f}s")
        
        # ===== TIMING: Time embeddings =====
        time_emb_start = time.time()
        
        #time embeddings
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=timestep.shape)
        #context
        context = self.text_embedding(context)

        time_emb_end = time.time()
        # print(f"[TIMING] Time & text embeddings: {time_emb_end - time_emb_start:.6f}s")

        ##To trunk the audio_emb into each block,we will embedding audio_emd in pipeline
        '''
        if audio_emb != None and self.use_audio: # TODO  cache
            audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
            audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2) # 1, 768, 44, 1, 1
            audio_emb = self.audio_proj(audio_emb)
            audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.audio_cond_projs], 0)
        '''
        
        # ===== TIMING: Frequency preparation =====
        freq_start = time.time()
        
        # arguments
        kwargs=dict(
            grid_sizes=grid_sizes,
            block_mask=self.block_mask
        )
        
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        
        ##causal rope
        frame_seqlen=math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        
        # 使用缓存的频率计算
        freqs = self._get_cached_freqs(f, h, w, current_start_frame, x.device)
        
        freq_end = time.time()
        print(f"[TIMING] Frequency preparation: {freq_end - freq_start:.6f}s")
        
        # ===== TIMING: KV Cache operations =====
        kv_cache_start = time.time()
        
        if tea_cache is not None:
            tea_cache_update = tea_cache.check(self, x, t_mod)
        else:
            tea_cache_update = False
        ori_x_len = x.shape[1]
        if tea_cache_update:
            x = tea_cache.update(x)
        else:
            if args.sp_size > 1:
                # Context Parallel
                sp_size = get_sequence_parallel_world_size()
                pad_size = 0
                if ori_x_len % sp_size != 0:
                    pad_size = sp_size - ori_x_len % sp_size
                    x = torch.cat([x, torch.zeros_like(x[:, -1:]).repeat(1, pad_size, 1)], 1)
                x = torch.chunk(x, sp_size, dim=1)[get_sequence_parallel_rank()]
            kv_cache_end = time.time()
            # print(f"[TIMING] KV Cache setup: {kv_cache_end - kv_cache_start:.6f}s")
            
            # ===== TIMING: Main transformer blocks =====
            blocks_start = time.time()
            
            audio_emb = audio_emb.reshape(x.shape[0], audio_emb.shape[0] // x.shape[0], -1, *audio_emb.shape[2:])
            for layer_i, block in enumerate(self.blocks):
                layer_start = time.time()
                # audio cond
                if self.use_audio:
                    au_idx = None
                    if (layer_i <= len(self.blocks) // 2 and layer_i > 1): # < len(self.blocks) - 1:
                        au_idx = layer_i - 2
                        audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, lat_h // 2, lat_w // 2, 1) # 1, 11, 45, 25, 128    
                        audio_cond_tmp = self.patchify(audio_emb_tmp.permute(0, 4, 1, 2, 3))[0]
                        if args.sp_size > 1:
                            if pad_size > 0:    
                                audio_cond_tmp = torch.cat([audio_cond_tmp, torch.zeros_like(audio_cond_tmp[:, -1:]).repeat(1, pad_size, 1)], 1)
                            audio_cond_tmp = torch.chunk(audio_cond_tmp, sp_size, dim=1)[get_sequence_parallel_rank()]
                       
                        x = audio_cond_tmp + x
                layer_audio_end = time.time()
                print(f"[TIMING] Layer {layer_i} audio process: {layer_audio_end - layer_start:.6f}s")
                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            kwargs.update(
                                {
                                    "kv_cache": kv_cache[layer_i],
                                    "current_start": current_start,
                                    "cache_start": cache_start
                                }
                            )
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x, context, t_mod, freqs,**kwargs,
                                use_reentrant=False,
                            )
                    else:
                        kwargs.update(
                            {
                                "kv_cache": kv_cache[layer_i],
                                "current_start": current_start,
                                "crossattn_cache": crossattn_cache[layer_i],
                                "cache_start": cache_start
                            }
                        )
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,**kwargs,
                            use_reentrant=False,
                        )
                else:
                    layer_block_start = time.time()
                    kwargs.update(
                        {
                            "kv_cache": kv_cache[layer_i],
                            "current_start": current_start,
                            "crossattn_cache": crossattn_cache[layer_i],
                            "cache_start": cache_start
                        }
                    )
                    x = block(x, context, t_mod, freqs,**kwargs)
                    layer_block_end = time.time()
                    print(f"[TIMING] Layer {layer_i} (with KV cache): {layer_block_end - layer_block_start:.6f}s")
                    
            blocks_end = time.time()
            print(f"[TIMING] All transformer blocks (with KV cache): {blocks_end - blocks_start:.6f}s")
            
            # ===== TIMING: Cache operations =====
            cache_start = time.time()
            
            if tea_cache is not None:
                x_cache = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
                x_cache = x_cache[:, :ori_x_len]
                tea_cache.store(x_cache)

        cache_end = time.time()
        # print(f"[TIMING] Cache operations: {cache_end - cache_start:.6f}s")
        
        # ===== TIMING: Head and unpatchify =====
        head_start = time.time()

        x = self.head(x, t.unflatten(dim=0,sizes=timestep.shape).unsqueeze(2))
        if args.sp_size > 1:
            # Context Parallel
            x = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
            x = x[:, :ori_x_len]
        x=x.reshape(x.shape[0],-1,x.shape[-1])
        x = self.unpatchify(x, (f, h, w))
        
        head_end = time.time()
        # print(f"[TIMING] Head and unpatchify: {head_end - head_start:.6f}s")
        
        # ===== TIMING: Total forward_inference time =====
        total_end = time.time()
        print(f"[TIMING] _forward_inference TOTAL: {total_end - start_time:.6f}s")
        print("=" * 60)
        
        return x
    
    def _get_cached_freqs(self, f, h, w, current_start_frame, device):
        """获取缓存的频率张量，避免重复计算"""
        cache_key = (f, h, w, current_start_frame, str(device))
        
        # 首先检查设备特定的缓存
        if cache_key in self.freq_cache:
            return self.freq_cache[cache_key]
        
        # 检查是否有预计算的CPU版本
        cpu_cache_key = (f, h, w, current_start_frame, 'cpu')
        if cpu_cache_key in self.freq_cache:
            # 将预计算的CPU张量转移到目标设备
            freqs = self.freq_cache[cpu_cache_key].to(device)
            self.freq_cache[cache_key] = freqs
            return freqs
        
        # 如果没有缓存，进行计算
        # 使用torch.no_grad()减少内存开销
        with torch.no_grad():
            freqs = torch.cat([
                self.freqs[0][current_start_frame:current_start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(device)
        
        # 缓存管理：如果缓存太大，清理最旧的条目
        if len(self.freq_cache) >= self.max_cache_size:
            # 简单的FIFO策略，删除第一个条目
            oldest_key = next(iter(self.freq_cache))
            del self.freq_cache[oldest_key]
        
        self.freq_cache[cache_key] = freqs
        return freqs
    
    def _precompute_common_freqs(self):
        """预计算常用的频率组合"""
        # 常见的视频尺寸组合
        common_configs = [
            (1, 32, 32),   # 小尺寸
            (1, 45, 25),   # 中等尺寸  
            (4, 32, 32),   # 多帧小尺寸
            (4, 45, 25),   # 多帧中等尺寸
        ]
        
        # 预计算前几帧的频率
        for f, h, w in common_configs:
            for start_frame in range(min(10, len(self.freqs[0]))):  # 预计算前10帧
                try:
                    if start_frame + f <= len(self.freqs[0]):
                        # 使用CPU进行预计算，避免GPU内存问题
                        freqs = torch.cat([
                            self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
                        ], dim=-1).reshape(f * h * w, 1, -1)
                        
                        # 存储到缓存中（CPU版本，使用时再转移到GPU）
                        cache_key = (f, h, w, start_frame, 'cpu')
                        self.freq_cache[cache_key] = freqs
                except Exception:
                    # 如果预计算失败，跳过这个配置
                    continue
    
    #crate mask.
    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        #Currently, we do not need teacherforcing/
        raise NotImplementedError()

        return None

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        #Currently, we do not need it
        raise NotImplementedError()

        return None
    
    @staticmethod
    def state_dict_converter():
        return CausalWanModelStateDictConverter()
    
    
class CausalWanModelStateDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        #Do not need other config now
         
        print("Loading Trained CausalModel.")
        config = {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 33,
            "dim": 1536,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 12,
            "num_layers": 30,
            "eps": 1e-6,
        }
            
        if hasattr(args, "model_config"):
            model_config = args.model_config
            if model_config is not None:
                config.update(model_config)        
        return state_dict, config
