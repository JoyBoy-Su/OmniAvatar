#!/usr/bin/env python3
"""
音视频数据预处理脚本
包括质量过滤、格式转换、特征提取、数据打包
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
import librosa
import soundfile as sf
import torch
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ========================
# Configuration
# ========================

@dataclass
class PreprocessConfig:
    """预处理配置"""
    # Input/Output
    input_dir: str = "/data/audio_video_raw"
    output_dir: str = "/data/audio_video_processed"
    metadata_db: str = "metadata.jsonl"
    
    # Quality Filtering
    min_resolution: int = 480
    max_resolution: int = 1920
    min_fps: int = 15
    max_fps: int = 60
    min_duration: float = 3.0
    max_duration: float = 60.0
    min_audio_snr: float = 10.0  # dB
    max_silence_ratio: float = 0.5
    max_blur_score: float = 100.0  # Laplacian variance
    
    # Target Specifications
    target_video_resolution: int = 512
    target_video_fps: int = 8
    target_audio_sample_rate: int = 16000
    target_audio_channels: int = 1
    
    # Processing
    num_workers: int = 8
    batch_size: int = 100
    
    # WebDataset
    samples_per_shard: int = 1000
    shard_name_pattern: str = "shard-%06d.tar"

# ========================
# Logger
# ========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================
# Quality Filters
# ========================

class VideoQualityFilter:
    """视频质量检测"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
    
    def check_resolution(self, video_path: Path) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """检查分辨率"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            min_dim = min(width, height)
            max_dim = max(width, height)
            
            if min_dim < self.config.min_resolution or max_dim > self.config.max_resolution:
                return False, (width, height)
            
            return True, (width, height)
        except Exception as e:
            logger.error(f"Error checking resolution: {e}")
            return False, None
    
    def check_fps(self, video_path: Path) -> Tuple[bool, Optional[float]]:
        """检查帧率"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps < self.config.min_fps or fps > self.config.max_fps:
                return False, fps
            
            return True, fps
        except Exception as e:
            logger.error(f"Error checking FPS: {e}")
            return False, None
    
    def check_blur(self, video_path: Path, num_samples: int = 10) -> Tuple[bool, Optional[float]]:
        """检查模糊度（使用Laplacian方差）"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return False, None
            
            # Sample frames
            frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            blur_scores = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    blur_scores.append(laplacian_var)
            
            cap.release()
            
            if not blur_scores:
                return False, None
            
            avg_blur = np.mean(blur_scores)
            
            # Higher variance = sharper image
            if avg_blur < self.config.max_blur_score:
                return False, avg_blur
            
            return True, avg_blur
            
        except Exception as e:
            logger.error(f"Error checking blur: {e}")
            return False, None
    
    def check_scene_cuts(self, video_path: Path, threshold: float = 30.0) -> Tuple[bool, int]:
        """检查场景切换次数（避免快速剪辑）"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            prev_frame = None
            scene_cuts = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = diff.mean()
                    
                    if mean_diff > threshold:
                        scene_cuts += 1
                
                prev_frame = gray
                
                # Sample every 10 frames for efficiency
                for _ in range(9):
                    ret = cap.grab()
                    if not ret:
                        break
                    frame_count += 1
            
            cap.release()
            
            if frame_count == 0:
                return False, 0
            
            # Calculate cuts per second
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            cuts_per_second = scene_cuts / duration if duration > 0 else 0
            
            # Reject if too many cuts (> 2 per second)
            if cuts_per_second > 2.0:
                return False, scene_cuts
            
            return True, scene_cuts
            
        except Exception as e:
            logger.error(f"Error checking scene cuts: {e}")
            return False, 0
    
    def filter_video(self, video_path: Path) -> Tuple[bool, Dict]:
        """综合视频质量检查"""
        results = {}
        
        # Check resolution
        res_ok, resolution = self.check_resolution(video_path)
        results['resolution'] = resolution
        results['resolution_ok'] = res_ok
        
        if not res_ok:
            return False, results
        
        # Check FPS
        fps_ok, fps = self.check_fps(video_path)
        results['fps'] = fps
        results['fps_ok'] = fps_ok
        
        if not fps_ok:
            return False, results
        
        # Check blur
        blur_ok, blur_score = self.check_blur(video_path)
        results['blur_score'] = blur_score
        results['blur_ok'] = blur_ok
        
        if not blur_ok:
            return False, results
        
        # Check scene cuts
        cuts_ok, scene_cuts = self.check_scene_cuts(video_path)
        results['scene_cuts'] = scene_cuts
        results['cuts_ok'] = cuts_ok
        
        return cuts_ok, results


class AudioQualityFilter:
    """音频质量检测"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
    
    def check_sample_rate(self, audio_path: Path) -> Tuple[bool, Optional[int]]:
        """检查采样率"""
        try:
            info = sf.info(audio_path)
            sr = info.samplerate
            
            # Accept sample rates >= 8kHz
            if sr < 8000:
                return False, sr
            
            return True, sr
        except Exception as e:
            logger.error(f"Error checking sample rate: {e}")
            return False, None
    
    def check_silence_ratio(self, audio_path: Path) -> Tuple[bool, Optional[float]]:
        """检查静音比例"""
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Detect silence
            intervals = librosa.effects.split(audio, top_db=20)
            
            if len(intervals) == 0:
                # All silence
                return False, 1.0
            
            # Calculate non-silence duration
            non_silence_duration = sum([end - start for start, end in intervals])
            total_duration = len(audio)
            
            silence_ratio = 1.0 - (non_silence_duration / total_duration)
            
            if silence_ratio > self.config.max_silence_ratio:
                return False, silence_ratio
            
            return True, silence_ratio
            
        except Exception as e:
            logger.error(f"Error checking silence: {e}")
            return False, None
    
    def check_snr(self, audio_path: Path) -> Tuple[bool, Optional[float]]:
        """检查信噪比"""
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Estimate noise floor (bottom 10% of energy)
            energy = audio ** 2
            sorted_energy = np.sort(energy)
            noise_floor = np.mean(sorted_energy[:len(sorted_energy) // 10])
            
            # Estimate signal (top 50% of energy)
            signal_energy = np.mean(sorted_energy[len(sorted_energy) // 2:])
            
            if noise_floor == 0:
                snr = float('inf')
            else:
                snr = 10 * np.log10(signal_energy / noise_floor)
            
            if snr < self.config.min_audio_snr:
                return False, snr
            
            return True, snr
            
        except Exception as e:
            logger.error(f"Error checking SNR: {e}")
            return False, None
    
    def check_clipping(self, audio_path: Path) -> Tuple[bool, Optional[float]]:
        """检查削波失真"""
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Count samples near maximum amplitude
            clipping_threshold = 0.99
            clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
            clipping_ratio = clipped_samples / len(audio)
            
            # Reject if > 1% clipping
            if clipping_ratio > 0.01:
                return False, clipping_ratio
            
            return True, clipping_ratio
            
        except Exception as e:
            logger.error(f"Error checking clipping: {e}")
            return False, None
    
    def filter_audio(self, audio_path: Path) -> Tuple[bool, Dict]:
        """综合音频质量检查"""
        results = {}
        
        # Check sample rate
        sr_ok, sample_rate = self.check_sample_rate(audio_path)
        results['sample_rate'] = sample_rate
        results['sample_rate_ok'] = sr_ok
        
        if not sr_ok:
            return False, results
        
        # Check silence
        silence_ok, silence_ratio = self.check_silence_ratio(audio_path)
        results['silence_ratio'] = silence_ratio
        results['silence_ok'] = silence_ok
        
        if not silence_ok:
            return False, results
        
        # Check SNR
        snr_ok, snr = self.check_snr(audio_path)
        results['snr'] = snr
        results['snr_ok'] = snr_ok
        
        if not snr_ok:
            return False, results
        
        # Check clipping
        clip_ok, clip_ratio = self.check_clipping(audio_path)
        results['clipping_ratio'] = clip_ratio
        results['clipping_ok'] = clip_ok
        
        return clip_ok, results

# ========================
# Format Conversion
# ========================

class AVConverter:
    """音视频格式转换"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_video(self, input_path: Path, output_path: Path) -> bool:
        """转换视频到目标格式"""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-vf', f'scale={self.config.target_video_resolution}:-2',
                '-r', str(self.config.target_video_fps),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-an',  # No audio
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Video conversion error: {e.stderr.decode()}")
            return False
    
    def convert_audio(self, input_path: Path, output_path: Path) -> bool:
        """转换音频到目标格式"""
        try:
            # Load and resample
            audio, sr = librosa.load(
                input_path,
                sr=self.config.target_audio_sample_rate,
                mono=(self.config.target_audio_channels == 1)
            )
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Save
            sf.write(output_path, audio, self.config.target_audio_sample_rate)
            return True
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return False

# ========================
# Processing Pipeline
# ========================

class DataProcessor:
    """数据处理主流程"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.video_filter = VideoQualityFilter(config)
        self.audio_filter = AudioQualityFilter(config)
        self.converter = AVConverter(config)
        
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.output_dir / config.metadata_db
        
    def process_single_sample(self, sample_id: str) -> Optional[Dict]:
        """处理单个样本"""
        video_path = self.input_dir / f"{sample_id}.mp4"
        audio_path = self.input_dir / f"{sample_id}.wav"
        metadata_path = self.input_dir / f"{sample_id}.json"
        
        # Check files exist
        if not all([video_path.exists(), audio_path.exists(), metadata_path.exists()]):
            logger.warning(f"Missing files for {sample_id}")
            return None
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Quality filtering
        video_ok, video_results = self.video_filter.filter_video(video_path)
        audio_ok, audio_results = self.audio_filter.filter_audio(audio_path)
        
        if not video_ok or not audio_ok:
            logger.info(f"Sample {sample_id} filtered out")
            logger.debug(f"  Video: {video_results}")
            logger.debug(f"  Audio: {audio_results}")
            return None
        
        # Convert to target format
        output_video = self.output_dir / f"{sample_id}.mp4"
        output_audio = self.output_dir / f"{sample_id}.wav"
        
        video_conv_ok = self.converter.convert_video(video_path, output_video)
        audio_conv_ok = self.converter.convert_audio(audio_path, output_audio)
        
        if not video_conv_ok or not audio_conv_ok:
            logger.warning(f"Conversion failed for {sample_id}")
            return None
        
        # Update metadata
        metadata.update({
            'sample_id': sample_id,
            'video_quality': video_results,
            'audio_quality': audio_results,
            'processed_video_path': str(output_video),
            'processed_audio_path': str(output_audio),
            'status': 'processed'
        })
        
        return metadata
    
    def process_batch(self, sample_ids: List[str]):
        """批量处理样本"""
        results = []
        
        for sample_id in tqdm(sample_ids, desc="Processing samples"):
            result = self.process_single_sample(sample_id)
            if result:
                results.append(result)
        
        # Save metadata
        with open(self.metadata_file, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        return results
    
    def process_all(self):
        """处理所有样本"""
        # Find all samples
        json_files = list(self.input_dir.glob("*.json"))
        sample_ids = [f.stem for f in json_files]
        
        logger.info(f"Found {len(sample_ids)} samples to process")
        
        # Process in batches
        for i in range(0, len(sample_ids), self.config.batch_size):
            batch = sample_ids[i:i + self.config.batch_size]
            logger.info(f"Processing batch {i // self.config.batch_size + 1}")
            self.process_batch(batch)

# ========================
# WebDataset Creation
# ========================

def create_webdataset_shards(
    metadata_file: Path,
    output_dir: Path,
    samples_per_shard: int = 1000
):
    """创建WebDataset格式的数据分片"""
    import tarfile
    
    logger.info("Creating WebDataset shards...")
    
    # Read metadata
    samples = []
    with open(metadata_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    logger.info(f"Total samples: {len(samples)}")
    
    # Create shards
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    num_shards = (len(samples) + samples_per_shard - 1) // samples_per_shard
    
    for shard_idx in tqdm(range(num_shards), desc="Creating shards"):
        shard_path = shard_dir / f"shard-{shard_idx:06d}.tar"
        
        start_idx = shard_idx * samples_per_shard
        end_idx = min((shard_idx + 1) * samples_per_shard, len(samples))
        shard_samples = samples[start_idx:end_idx]
        
        with tarfile.open(shard_path, 'w') as tar:
            for local_idx, sample in enumerate(shard_samples):
                sample_id = f"{shard_idx:06d}_{local_idx:04d}"
                
                # Add video
                video_path = Path(sample['processed_video_path'])
                if video_path.exists():
                    tar.add(video_path, arcname=f"{sample_id}.mp4")
                
                # Add audio
                audio_path = Path(sample['processed_audio_path'])
                if audio_path.exists():
                    tar.add(audio_path, arcname=f"{sample_id}.wav")
                
                # Add metadata
                meta_str = json.dumps(sample)
                meta_bytes = meta_str.encode('utf-8')
                
                import io
                meta_buffer = io.BytesIO(meta_bytes)
                tarinfo = tarfile.TarInfo(name=f"{sample_id}.json")
                tarinfo.size = len(meta_bytes)
                tar.addfile(tarinfo, meta_buffer)
    
    logger.info(f"Created {num_shards} shards in {shard_dir}")

# ========================
# Main
# ========================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio-Video Data Preprocessing')
    parser.add_argument('--mode', choices=['process', 'create_shards', 'all'],
                        default='all', help='Processing mode')
    parser.add_argument('--input-dir', type=str, default='/data/audio_video_raw')
    parser.add_argument('--output-dir', type=str, default='/data/audio_video_processed')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--samples-per-shard', type=int, default=1000)
    
    args = parser.parse_args()
    
    config = PreprocessConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        samples_per_shard=args.samples_per_shard
    )
    
    if args.mode in ['process', 'all']:
        logger.info("Starting data processing...")
        processor = DataProcessor(config)
        processor.process_all()
        logger.info("Data processing completed")
    
    if args.mode in ['create_shards', 'all']:
        logger.info("Creating WebDataset shards...")
        metadata_file = Path(config.output_dir) / config.metadata_db
        create_webdataset_shards(
            metadata_file,
            Path(config.output_dir),
            config.samples_per_shard
        )
        logger.info("Shard creation completed")

if __name__ == '__main__':
    main()

