/**
 * 应用程序状态和配置类型
 */

import { MediaFile } from './media';
import { ConnectionStatus, GenerationStatus } from './socket';

// ==================== 应用状态 ====================

export interface AppState {
  // 连接状态
  connectionStatus: ConnectionStatus;
  
  // 会话信息
  sessionId: string | null;
  
  // 生成状态
  generationStatus: GenerationStatus;
  generationProgress: number;
  
  // 媒体文件
  audioFile: MediaFile | null;
  imageFile: MediaFile | null;
  
  // 错误信息
  error: string | null;
  
  // UI 状态
  showAudioControls: boolean;
  showVideoControls: boolean;
}

// ==================== 通知消息 ====================

export enum MessageType {
  INFO = 'info',
  SUCCESS = 'success',
  WARNING = 'warning',
  ERROR = 'error',
}

export interface Message {
  id: string;
  type: MessageType;
  content: string;
  duration?: number; // 显示时长（毫秒），undefined 表示不自动关闭
}

// ==================== 应用配置 ====================

export interface AppConfig {
  serverUrl: string;
  socketPath: string;
  maxAudioSize: number; // 最大音频文件大小（字节）
  maxImageSize: number; // 最大图片文件大小（字节）
  supportedAudioFormats: string[];
  supportedImageFormats: string[];
  defaultFps: number;
  reconnectAttempts: number;
  reconnectDelay: number;
}

// ==================== 统计信息 ====================

export interface GenerationStats {
  startTime: number;
  endTime: number | null;
  duration: number | null; // 毫秒
  framesGenerated: number;
  totalFrames: number;
  averageFps: number;
  audioSegments: number;
}

