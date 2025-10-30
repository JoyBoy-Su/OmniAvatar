#!/usr/bin/env python3
"""
大规模音视频数据爬取脚本
支持YouTube, Bilibili, TikTok等多平台
使用分布式任务队列实现高效爬取
"""

import os
import sys
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import yt_dlp
from celery import Celery
from redis import Redis
import subprocess

# ========================
# Configuration
# ========================

@dataclass
class CrawlerConfig:
    """爬虫配置"""
    redis_url: str = "redis://localhost:6379/0"
    output_dir: str = "/data/audio_video_raw"
    max_duration: int = 60  # seconds
    min_duration: int = 3
    target_resolution: int = 720
    audio_format: str = "wav"
    video_format: str = "mp4"
    num_workers: int = 50
    download_rate_limit: str = "10M"  # Per worker
    retry_times: int = 3

# ========================
# Celery App Setup
# ========================

config = CrawlerConfig()
app = Celery('av_crawler', broker=config.redis_url, backend=config.redis_url)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ========================
# Logger Setup
# ========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================
# Redis Connection
# ========================

redis_client = Redis.from_url(config.redis_url, decode_responses=True)

# ========================
# URL Collection
# ========================

class URLCollector:
    """从各个平台收集视频URL"""
    
    def __init__(self):
        self.redis = redis_client
        
    def collect_youtube_urls(
        self, 
        queries: List[str], 
        max_results: int = 10000
    ) -> int:
        """从YouTube搜索并收集URL"""
        collected = 0
        
        for query in queries:
            logger.info(f"Collecting YouTube URLs for query: {query}")
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'force_generic_extractor': False,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Search query
                    search_url = f"ytsearch{max_results}:{query}"
                    result = ydl.extract_info(search_url, download=False)
                    
                    if 'entries' in result:
                        for entry in result['entries']:
                            url = entry.get('url') or entry.get('webpage_url')
                            if url:
                                self.add_url_to_queue(url, 'youtube', query)
                                collected += 1
                                
            except Exception as e:
                logger.error(f"Error collecting URLs for {query}: {e}")
                
        logger.info(f"Collected {collected} URLs from YouTube")
        return collected
    
    def collect_from_channels(
        self, 
        channel_urls: List[str],
        max_videos_per_channel: int = 1000
    ) -> int:
        """从YouTube频道收集视频"""
        collected = 0
        
        for channel_url in channel_urls:
            logger.info(f"Collecting from channel: {channel_url}")
            
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'playlistend': max_videos_per_channel,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(channel_url, download=False)
                    
                    if 'entries' in result:
                        for entry in result['entries']:
                            url = entry.get('url') or entry.get('webpage_url')
                            if url:
                                self.add_url_to_queue(url, 'youtube', 'channel')
                                collected += 1
                                
            except Exception as e:
                logger.error(f"Error collecting from channel {channel_url}: {e}")
        
        logger.info(f"Collected {collected} URLs from channels")
        return collected
    
    def add_url_to_queue(self, url: str, platform: str, category: str):
        """添加URL到Redis队列"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        # Check if already processed
        if self.redis.exists(f"processed:{url_hash}"):
            return False
        
        # Check if already in queue
        if self.redis.exists(f"queued:{url_hash}"):
            return False
        
        # Add to queue
        self.redis.lpush('url_queue', json.dumps({
            'url': url,
            'url_hash': url_hash,
            'platform': platform,
            'category': category,
        }))
        
        # Mark as queued
        self.redis.setex(f"queued:{url_hash}", 86400, 1)  # 24 hours
        
        return True

# ========================
# Download Tasks
# ========================

@app.task(bind=True, max_retries=3)
def download_video_task(self, url_data: str):
    """Celery任务：下载单个视频"""
    url_info = json.loads(url_data)
    url = url_info['url']
    url_hash = url_info['url_hash']
    platform = url_info['platform']
    
    logger.info(f"Downloading: {url}")
    
    try:
        downloader = VideoDownloader(config)
        success, metadata = downloader.download(url, url_hash, platform)
        
        if success:
            # Mark as processed
            redis_client.setex(f"processed:{url_hash}", 86400 * 7, 1)
            redis_client.delete(f"queued:{url_hash}")
            
            # Save metadata
            downloader.save_metadata(url_hash, metadata)
            
            logger.info(f"Successfully downloaded: {url_hash}")
            return {'status': 'success', 'url_hash': url_hash}
        else:
            logger.warning(f"Failed to download: {url}")
            return {'status': 'failed', 'url': url}
            
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        # Retry
        raise self.retry(exc=e, countdown=60)

# ========================
# Video Downloader
# ========================

class VideoDownloader:
    """视频下载器"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download(
        self, 
        url: str, 
        url_hash: str, 
        platform: str
    ) -> tuple[bool, Optional[Dict]]:
        """下载视频和音频"""
        
        video_path = self.output_dir / f"{url_hash}.{self.config.video_format}"
        audio_path = self.output_dir / f"{url_hash}.{self.config.audio_format}"
        
        # yt-dlp options
        ydl_opts = {
            'format': f'bestvideo[height<={self.config.target_resolution}]+bestaudio/best',
            'outtmpl': str(self.output_dir / f'{url_hash}.%(ext)s'),
            'ratelimit': self.config.download_rate_limit,
            'retries': self.config.retry_times,
            'fragment_retries': self.config.retry_times,
            'quiet': False,
            'no_warnings': False,
            'postprocessors': [
                {
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': self.config.video_format,
                },
            ],
        }
        
        try:
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Check duration
                duration = info.get('duration', 0)
                if duration < self.config.min_duration or duration > self.config.max_duration:
                    logger.warning(f"Duration {duration}s out of range, skipping")
                    self.cleanup_files(url_hash)
                    return False, None
                
                # Extract audio separately
                self.extract_audio(video_path, audio_path)
                
                # Prepare metadata
                metadata = {
                    'url_hash': url_hash,
                    'url': url,
                    'platform': platform,
                    'title': info.get('title', ''),
                    'duration': duration,
                    'resolution': f"{info.get('width', 0)}x{info.get('height', 0)}",
                    'fps': info.get('fps', 0),
                    'upload_date': info.get('upload_date', ''),
                    'uploader': info.get('uploader', ''),
                    'categories': info.get('categories', []),
                    'tags': info.get('tags', []),
                    'description': info.get('description', '')[:500],  # Truncate
                    'video_path': str(video_path),
                    'audio_path': str(audio_path),
                }
                
                return True, metadata
                
        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
            self.cleanup_files(url_hash)
            return False, None
    
    def extract_audio(self, video_path: Path, audio_path: Path):
        """从视频中提取音频"""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le' if self.config.audio_format == 'wav' else 'copy',
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            str(audio_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Extracted audio: {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
    
    def save_metadata(self, url_hash: str, metadata: Dict):
        """保存元数据到JSON文件"""
        metadata_path = self.output_dir / f"{url_hash}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def cleanup_files(self, url_hash: str):
        """清理下载失败的文件"""
        patterns = [
            f"{url_hash}.{self.config.video_format}",
            f"{url_hash}.{self.config.audio_format}",
            f"{url_hash}.json",
            f"{url_hash}.*"
        ]
        
        for pattern in patterns:
            for file in self.output_dir.glob(pattern):
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting {file}: {e}")

# ========================
# Worker Management
# ========================

class CrawlerManager:
    """爬虫管理器"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.collector = URLCollector()
        
    def start_workers(self, num_workers: int):
        """启动Celery workers"""
        logger.info(f"Starting {num_workers} workers...")
        
        cmd = [
            'celery',
            '-A', 'data_crawler',
            'worker',
            '--concurrency', str(num_workers),
            '--loglevel', 'info',
            '--max-tasks-per-child', '100',  # Restart worker after 100 tasks
        ]
        
        subprocess.Popen(cmd)
    
    def collect_urls_from_config(self, config_file: str):
        """从配置文件收集URL"""
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        # YouTube searches
        if 'youtube_queries' in data:
            self.collector.collect_youtube_urls(
                data['youtube_queries'],
                max_results=data.get('max_results_per_query', 1000)
            )
        
        # YouTube channels
        if 'youtube_channels' in data:
            self.collector.collect_from_channels(
                data['youtube_channels'],
                max_videos_per_channel=data.get('max_videos_per_channel', 500)
            )
    
    def dispatch_download_tasks(self):
        """分发下载任务"""
        logger.info("Dispatching download tasks...")
        
        while True:
            # Get URL from queue
            url_data = redis_client.rpop('url_queue')
            if not url_data:
                break
            
            # Submit to Celery
            download_video_task.delay(url_data)
    
    def get_statistics(self) -> Dict:
        """获取爬取统计信息"""
        queue_size = redis_client.llen('url_queue')
        processed = len(redis_client.keys('processed:*'))
        queued = len(redis_client.keys('queued:*'))
        
        return {
            'queue_size': queue_size,
            'processed': processed,
            'queued': queued,
            'total': queue_size + processed,
        }

# ========================
# CLI Interface
# ========================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio-Video Data Crawler')
    parser.add_argument('--mode', choices=['collect', 'download', 'worker', 'stats'],
                        required=True, help='Operation mode')
    parser.add_argument('--config', type=str, default='crawler_config.json',
                        help='Config file for URL collection')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of worker processes')
    parser.add_argument('--output-dir', type=str, default='/data/audio_video_raw',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Update config
    config.output_dir = args.output_dir
    config.num_workers = args.num_workers
    
    manager = CrawlerManager(config)
    
    if args.mode == 'collect':
        # Collect URLs from config
        logger.info("Starting URL collection...")
        manager.collect_urls_from_config(args.config)
        logger.info("URL collection completed")
        
    elif args.mode == 'download':
        # Dispatch download tasks
        logger.info("Dispatching download tasks...")
        manager.dispatch_download_tasks()
        logger.info("All tasks dispatched")
        
    elif args.mode == 'worker':
        # Start workers
        logger.info(f"Starting {args.num_workers} workers...")
        manager.start_workers(args.num_workers)
        
    elif args.mode == 'stats':
        # Show statistics
        stats = manager.get_statistics()
        print("=" * 50)
        print("Crawler Statistics")
        print("=" * 50)
        print(f"URLs in queue:    {stats['queue_size']}")
        print(f"URLs processed:   {stats['processed']}")
        print(f"URLs queued:      {stats['queued']}")
        print(f"Total:            {stats['total']}")
        print("=" * 50)

if __name__ == '__main__':
    main()

