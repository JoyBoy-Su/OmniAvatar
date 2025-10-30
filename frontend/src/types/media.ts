/**
 * 音视频相关的类型定义
 */

// ==================== 音频相关 ====================

export interface AudioSegment {
  data: ArrayBuffer;
  startTime: number;
  duration: number;
  frameNumber: number;
}

export interface AudioRecorderState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  blob: Blob | null;
}

export interface AudioPlayerState {
  isPlaying: boolean;
  isPaused: boolean;
  volume: number;
  isMuted: boolean;
  currentTime: number;
  duration: number;
}

// ==================== 视频相关 ====================

export interface VideoFrame {
  frameNumber: number;
  totalFrames: number;
  chunkNumber: number;
  imageData: string; // base64
  timestamp: number;
}

export interface VideoPlayerState {
  isPlaying: boolean;
  isPaused: boolean;
  currentFrame: number;
  totalFrames: number;
  fps: number;
  canvasReady: boolean;
}

export interface VideoMetadata {
  width: number;
  height: number;
  fps: number;
  totalFrames: number;
  duration: number;
}

// ==================== 同步机制 ====================

export interface SyncState {
  videoTime: number; // 视频当前时间（秒）
  audioTime: number; // 音频当前时间（秒）
  drift: number; // 音视频漂移（毫秒）
  isSynced: boolean; // 是否同步
}

export interface SyncConfig {
  maxDrift: number; // 最大允许漂移（毫秒）
  syncInterval: number; // 同步检查间隔（毫秒）
  bufferSize: number; // 缓冲区大小（帧数）
}

// ==================== 媒体文件信息 ====================

export interface MediaFile {
  file: File;
  name: string;
  size: number;
  type: string;
  preview?: string; // 预览URL（图片）
  duration?: number; // 持续时间（音频/视频）
}

// ==================== 上传状态 ====================

export enum UploadStatus {
  IDLE = 'idle',
  UPLOADING = 'uploading',
  COMPLETED = 'completed',
  ERROR = 'error',
}

export interface UploadProgress {
  status: UploadStatus;
  progress: number; // 0-100
  error?: string;
}

