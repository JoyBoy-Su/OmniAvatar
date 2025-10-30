# 音视频联合生成 (Audio-Video Joint Generation) 项目

这是一个基于Diffusion Transformer (DiT)的音视频联合生成研究项目。本项目实现了从文本/音频/视频条件生成高质量、同步的音视频内容。

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [推理使用](#推理使用)
- [项目结构](#项目结构)
- [技术细节](#技术细节)
- [常见问题](#常见问题)

---

## 🎯 项目概述

### 核心特性

- **统一DiT架构**: 使用单一的Diffusion Transformer处理音视频联合生成
- **双解码器设计**: 独立的音频和视频解码器，保证各模态生成质量
- **时序对齐模块**: 确保生成的音频和视频在时序上完美同步
- **大规模训练**: 支持50M+样本、分布式训练、混合精度等
- **多种条件输入**: 支持文本、音频、视频作为条件输入

### 技术栈

```
核心框架:   PyTorch 2.1+, Hugging Face Diffusers
分布式:     DeepSpeed, FSDP, Accelerate
优化:       FlashAttention-2, xFormers, torch.compile
数据:       WebDataset, ffmpeg, librosa
监控:       Weights & Biases, TensorBoard
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/your-org/audio-video-generation.git
cd audio-video-generation

# 创建虚拟环境
conda create -n avgen python=3.10
conda activate avgen

# 安装依赖
pip install -r requirements.txt

# 安装额外依赖
pip install flash-attn --no-build-isolation
pip install git+https://github.com/facebookresearch/xformers.git
```

### 2. 下载预训练模型

```bash
# 下载预训练的编码器和VAE
python scripts/download_pretrained.py

# 或手动下载
# CLIP: openai/clip-vit-large-patch14
# Wav2Vec2: facebook/wav2vec2-large-robust
# SD-VAE: stabilityai/sd-vae-ft-mse
# HiFi-GAN: nvidia/bigvgan_v2_22khz_80band
```

### 3. 准备示例数据

```bash
# 下载示例数据集 (1K samples)
bash scripts/download_demo_data.sh

# 或使用自己的数据
python scripts/data_preprocessing.py \
    --input-dir /path/to/your/videos \
    --output-dir /data/processed \
    --mode all
```

### 4. 训练模型

```bash
# 单机8卡训练
bash scripts/train_multi_gpu.sh

# 或使用Python直接训练
accelerate launch --multi_gpu --num_processes 8 \
    training/train.py \
    --config configs/model_config.yaml \
    --stage stage2
```

### 5. 推理生成

```python
from pipelines import AudioVideoGenerationPipeline

# 加载模型
pipeline = AudioVideoGenerationPipeline.from_pretrained(
    "checkpoints/best_model",
    torch_dtype=torch.bfloat16
)
pipeline = pipeline.to("cuda")

# 文本到音视频生成
prompt = "A person playing piano in a concert hall"
output = pipeline(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    output_path="output.mp4"
)
```

---

## 📊 数据准备

### 数据爬取

我们提供了大规模音视频数据爬取工具，支持YouTube、Bilibili等平台。

```bash
# 1. 配置爬取任务
# 编辑 crawler_config.json，添加搜索关键词和频道

# 2. 启动Redis (用于任务队列)
docker run -d -p 6379:6379 redis:latest

# 3. 收集URL
python scripts/data_crawler.py \
    --mode collect \
    --config crawler_config.json

# 4. 启动多个爬取worker
for i in {1..50}; do
    python scripts/data_crawler.py --mode worker --num-workers 1 &
done

# 5. 分发下载任务
python scripts/data_crawler.py --mode download

# 6. 查看统计
python scripts/data_crawler.py --mode stats
```

### 预期数据规模

| 阶段 | 数据量 | 时长 | 存储 |
|------|--------|------|------|
| Stage 1 (单模态) | 20M | 55K小时 | 300TB |
| Stage 2 (联合) | 30M | 83K小时 | 450TB |
| Stage 3 (微调) | 2M | 8K小时 | 30TB |
| **总计** | **52M** | **146K小时** | **780TB** |

### 数据预处理

```bash
# 质量过滤 + 格式转换 + 打包成WebDataset
python scripts/data_preprocessing.py \
    --input-dir /data/raw \
    --output-dir /data/processed \
    --num-workers 32 \
    --samples-per-shard 1000 \
    --mode all
```

预处理包括:
- ✅ 分辨率检查 (≥480p)
- ✅ 帧率检查 (≥24fps)
- ✅ 模糊度检测 (Laplacian方差)
- ✅ 场景切换检测 (避免快速剪辑)
- ✅ 音频信噪比 (SNR >10dB)
- ✅ 静音比例 (<50%)
- ✅ 削波检测
- ✅ 格式统一 (512x512@8fps, 16kHz mono)

---

## 🏋️ 模型训练

### 训练配置

所有训练配置在 `configs/model_config.yaml` 中定义:

```yaml
# 核心模型参数
dit:
  hidden_size: 2048
  depth: 24
  num_heads: 16
  
# 训练配置
training:
  learning_rate: 1e-4
  train_batch_size: 16
  gradient_accumulation_steps: 4
  mixed_precision: "bf16"
  
# 分布式配置
distributed:
  strategy: "deepspeed"
  zero_stage: 2
```

### 三阶段训练

#### Stage 1: 单模态预训练 (2-4周)

```bash
bash scripts/train_multi_gpu.sh
# 修改脚本中的 TRAINING_STAGE="stage1"
```

- 数据: 10M音频 + 10M视频 (分别训练或并行)
- 目标: 建立音频/视频生成基础能力
- 学习率: 1e-4

#### Stage 2: 联合训练 (5-8周)

```bash
bash scripts/train_multi_gpu.sh
# TRAINING_STAGE="stage2"
```

- 数据: 30M音视频配对数据
- 目标: 学习音视频对齐和联合生成
- 学习率: 5e-5
- 关键: alignment loss权重逐步增加

#### Stage 3: 高质量微调 (1-2周)

```bash
bash scripts/train_multi_gpu.sh
# TRAINING_STAGE="stage3"
```

- 数据: 2M高质量配对数据
- 目标: 提升生成质量和细节
- 学习率: 1e-5
- 技巧: 使用人工筛选的高质量数据

### 分布式训练

#### 单节点多GPU (推荐用于开发和小规模实验)

```bash
# 使用Accelerate
accelerate launch --multi_gpu --num_processes 8 training/train.py

# 使用DeepSpeed
deepspeed --num_gpus=8 training/train.py --deepspeed configs/deepspeed_config.json
```

#### 多节点分布式 (大规模训练)

**节点0 (Master):**
```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export NODE_RANK=0
export WORLD_SIZE=4  # 4 nodes

deepspeed --num_gpus=8 --num_nodes=4 --node_rank=0 \
    training/train.py --deepspeed configs/deepspeed_config.json
```

**节点1-3 (Workers):**
```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export NODE_RANK=1  # 2, 3 for other nodes
export WORLD_SIZE=4

deepspeed --num_gpus=8 --num_nodes=4 --node_rank=$NODE_RANK \
    training/train.py --deepspeed configs/deepspeed_config.json
```

### 监控训练

```bash
# 使用Weights & Biases
export WANDB_API_KEY=your_api_key
# 自动记录到 W&B

# 或使用TensorBoard
tensorboard --logdir outputs/logs/
```

关键指标:
- `loss/diffusion`: 扩散模型主损失
- `loss/alignment`: 音视频对齐损失
- `metrics/fad`: 音频质量 (Fréchet Audio Distance)
- `metrics/fvd`: 视频质量 (Fréchet Video Distance)
- `metrics/ava`: 音视频对齐度

---

## 🎨 推理使用

### Python API

```python
import torch
from pipelines import AudioVideoGenerationPipeline

# 加载模型
pipeline = AudioVideoGenerationPipeline.from_pretrained(
    "checkpoints/stage2_best",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)
pipeline.to("cuda")

# 启用优化
pipeline.enable_xformers_memory_efficient_attention()
pipeline.enable_vae_slicing()

# 文本到音视频
output = pipeline(
    prompt="A violinist playing classical music in a grand hall",
    negative_prompt="low quality, blurry, distorted",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    num_frames=16,
    fps=8,
    audio_duration=10.0,
    seed=42
)

# 保存结果
output.save("output.mp4")
```

### 命令行工具

```bash
python scripts/generate.py \
    --checkpoint checkpoints/stage2_best \
    --prompt "A person playing piano" \
    --output output.mp4 \
    --steps 50 \
    --guidance-scale 7.5 \
    --seed 42
```

### 批量生成

```bash
# 从文件读取多个prompt
python scripts/batch_generate.py \
    --checkpoint checkpoints/stage2_best \
    --prompts-file prompts.txt \
    --output-dir outputs/generated \
    --batch-size 4
```

### 条件生成

```python
# 音频引导的视频生成
output = pipeline(
    audio_path="reference_audio.wav",
    prompt="A person making the sound",
    mode="audio_to_video"
)

# 视频引导的音频生成
output = pipeline(
    video_path="reference_video.mp4",
    prompt="Realistic sound for this video",
    mode="video_to_audio"
)

# 音视频编辑
output = pipeline(
    audio_path="original_audio.wav",
    video_path="original_video.mp4",
    prompt="Make it more dramatic",
    mode="edit"
)
```

---

## 📁 项目结构

```
audio-video-generation/
├── configs/                          # 配置文件
│   ├── model_config.yaml            # 模型配置
│   ├── training_config.yaml         # 训练配置
│   └── deepspeed_config.json        # DeepSpeed配置
│
├── models/                           # 模型实现
│   ├── dit_backbone.py              # DiT核心架构
│   ├── audio_decoder.py             # 音频解码器
│   ├── video_decoder.py             # 视频解码器
│   ├── encoders.py                  # 各类编码器
│   └── temporal_alignment.py        # 时序对齐模块
│
├── pipelines/                        # 推理Pipeline
│   ├── av_generation_pipeline.py    # 生成Pipeline
│   └── training_pipeline.py         # 训练Pipeline
│
├── training/                         # 训练相关
│   ├── train.py                     # 训练主脚本
│   ├── losses.py                    # 损失函数
│   ├── schedulers.py                # 学习率调度
│   └── callbacks.py                 # 训练回调
│
├── data/                             # 数据处理
│   ├── av_dataset.py                # 数据集定义
│   ├── preprocessing.py             # 预处理
│   └── augmentation.py              # 数据增强
│
├── evaluation/                       # 评估工具
│   ├── evaluate.py                  # 评估主脚本
│   ├── metrics.py                   # 评估指标
│   └── human_eval.py                # 人工评估工具
│
├── scripts/                          # 实用脚本
│   ├── data_crawler.py              # 数据爬取
│   ├── data_preprocessing.py        # 数据预处理
│   ├── train_multi_gpu.sh           # 训练启动脚本
│   ├── generate.py                  # 生成脚本
│   ├── export_model.py              # 模型导出
│   └── download_pretrained.py       # 下载预训练模型
│
├── utils/                            # 工具函数
│   ├── audio_utils.py               # 音频工具
│   ├── video_utils.py               # 视频工具
│   ├── logging_utils.py             # 日志工具
│   └── visualization.py             # 可视化工具
│
├── tests/                            # 测试
│   ├── test_models.py
│   ├── test_data.py
│   └── test_pipeline.py
│
├── Audio_Video_Joint_Generation_Proposal.md  # 详细技术方案
├── AUDIO_VIDEO_GENERATION_README.md          # 本文档
├── requirements.txt                           # Python依赖
├── crawler_config.json                        # 爬虫配置
└── README.md                                  # 项目主README
```

---

## 🔬 技术细节

### 模型架构

```
输入条件 (Text/Audio/Video)
    ↓
[编码器层]
├── Text: CLIP-ViT-L (1024d)
├── Audio: Wav2Vec2 (1024d)
└── Video: VideoMAE (1024d)
    ↓
[跨模态融合]
    ↓
[DiT Backbone] (2048d × 24 layers)
├── Patch Embedding
├── Position Embedding
├── Timestep Embedding
├── DiT Blocks (Self-Attention + MLP)
└── AdaLN (Adaptive Layer Norm)
    ↓
[时序对齐模块]
├── Audio Temporal Conv
├── Video Temporal Conv
└── Cross-Modal Attention
    ↓
    ┌───────┴───────┐
    ↓               ↓
[音频解码器]    [视频解码器]
8层Trans-        6层3D-CNN
former           Decoder
    ↓               ↓
Mel-Spec        Video Latent
80-bins         4×64×64
    ↓               ↓
[Vocoder]       [VAE Decoder]
HiFi-GAN        SD-VAE
    ↓               ↓
Waveform        Video Frames
16kHz           512×512@8fps
```

### 损失函数

总损失由以下部分组成:

```python
L_total = λ_diff * L_diffusion          # 主扩散损失
        + λ_align * L_alignment         # 音视频对齐损失
        + λ_perceptual * L_perceptual   # 感知质量损失
        + λ_temporal * L_temporal       # 时序一致性损失
```

**权重设置**:
- Stage 1: `λ_diff=1.0`, 其他为0
- Stage 2: `λ_diff=1.0, λ_align=0.1, λ_perceptual=0.05`
- Stage 3: `λ_diff=1.0, λ_align=0.5, λ_perceptual=0.1, λ_temporal=0.02`

### 关键技术

1. **AdaLN (Adaptive Layer Norm)**: 通过timestep调制normalization参数
2. **Flash Attention**: 高效的attention计算 (2-4x加速)
3. **Gradient Checkpointing**: 以计算换内存 (支持更大batch size)
4. **Mixed Precision (BF16)**: 加速训练，减少内存
5. **DeepSpeed ZeRO-2**: 分片优化器状态和梯度

---

## 💰 资源需求评估

### 算力需求

| 配置 | GPU型号 | 数量 | 训练时间 | 成本估算 |
|------|---------|------|----------|----------|
| **推荐** | A100-80GB | 128 | 8-12周 | $250K-350K |
| 经济 | A100-40GB | 256 | 16-24周 | $150K-250K |
| 最小 | A6000-48GB | 512 | 24-32周 | $100K-200K |

### 存储需求

- **原始数据**: 800TB (50M samples)
- **处理后数据**: 500TB (WebDataset格式)
- **检查点**: 50TB (每5K步保存一次)
- **临时缓存**: 100TB
- **总计**: ~1.5PB

### 网络带宽

- 数据爬取: 100Gbps × 2个月
- 训练集群: InfiniBand 200Gbps
- 数据传输到训练集群: 1PB @ 10Gbps = 10天

---

## 📈 预期性能指标

### 定量指标

| 指标 | 目标值 | SOTA对比 |
|------|--------|----------|
| FAD (Audio) | < 2.0 | 2.5 (AudioLDM) |
| FVD (Video) | < 500 | 550 (CogVideo) |
| CLAP Score | > 0.30 | 0.28 |
| CLIP Score | > 0.28 | 0.26 |
| AVA Score | > 0.70 | 0.65 |
| Temporal Consistency | LPIPS < 0.15 | 0.18 |

### 生成速度

- **训练**: 50 samples/sec (128 A100s)
- **推理** (A100):
  - 10秒音视频: ~8秒 (50 steps)
  - 批量生成 (batch=8): ~15秒
- **优化后** (TensorRT + INT8):
  - 单样本: ~3秒
  - 批量: ~6秒

---

## ❓ 常见问题

### Q1: 最小GPU要求是什么?

**A**: 推理至少需要16GB显存 (如RTX 4090, A100-40GB)。训练至少需要8×A100-40GB。

### Q2: 能否使用开源数据集而不爬取?

**A**: 可以。推荐数据集:
- VGGSound (200K)
- AudioSet (2M)
- Kinetics-700 (650K)
- 总计约3M样本

但规模较小，建议结合爬取达到20M+。

### Q3: 训练到Stage 2需要多久?

**A**: 
- 128 A100-80GB: 约6-8周
- 64 A100-80GB: 约10-12周
- 32 A100-80GB: 约16-20周

### Q4: 如何减少训练成本?

**A**: 
1. 使用Spot/Preemptible实例 (省50-70%)
2. 先用小模型验证 (1B参数)
3. 减少数据量到10M (省50%时间)
4. 使用DeepSpeed ZeRO-3 + Offload

### Q5: 模型可以商用吗?

**A**: 取决于:
- 训练数据的版权 (YouTube有限制)
- 预训练模型的License (CLIP, Wav2Vec2等)
- 建议使用开源数据集 + 授权内容

### Q6: 如何提升生成质量?

**A**:
1. 增加训练数据量和质量
2. 延长训练时间
3. 使用更大的模型 (5B-7B)
4. 加强对齐损失权重
5. 使用更好的vocoder (如BigVGAN)

### Q7: 支持实时生成吗?

**A**: 当前不支持。实时生成需要:
- 轻量级模型 (< 500M)
- 蒸馏技术
- 特殊的流式生成架构
- 可作为后续研究方向

---

## 🤝 贡献指南

我们欢迎社区贡献!

1. Fork本仓库
2. 创建feature分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -am 'Add some feature'`
4. 推送到分支: `git push origin feature/your-feature`
5. 提交Pull Request

代码规范:
- 遵循PEP 8
- 添加单元测试
- 更新文档
- 通过CI检查

---

## 📄 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 📧 联系方式

- 邮箱: research@example.com
- GitHub Issues: https://github.com/your-org/audio-video-generation/issues
- 论坛: https://discuss.example.com

---

## 🙏 致谢

本项目基于以下优秀工作:
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [DiT](https://github.com/facebookresearch/DiT)
- [Open-Sora](https://github.com/hpcaitech/Open-Sora)
- [AudioLDM](https://github.com/haoheliu/AudioLDM)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

特别感谢所有开源社区的贡献者!

---

## 📚 相关论文

如果本项目对您的研究有帮助，请考虑引用:

```bibtex
@article{avgen2025,
  title={Unified Diffusion Transformers for Joint Audio-Video Generation},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

---

**更新日期**: 2025-10-28  
**版本**: v1.0  
**维护者**: Research Team

