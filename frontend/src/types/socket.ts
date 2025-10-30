/**
 * Socket.IO 事件类型定义
 * 定义了与后端 WebSocket 通信的所有事件和数据结构
 */

// ==================== 会话相关 ====================

export interface SessionData {
  session_id: string;
}

export interface GenerationStartedData {
  session_id: string;
  message: string;
  audio_duration?: number;
  estimated_frames?: number;
}

// ==================== 视频帧相关 ====================

export interface VideoFrameData {
  session_id: string;
  frame_number: number;
  total_frames: number;
  chunk_number: number;
  total_chunks: number;
  progress: number;
  frame_data: string; // base64 encoded image
  audio_segment?: string; // base64 encoded audio
  audio_start_time?: number;
  audio_duration?: number;
}

export interface VideoChunkData {
  session_id: string;
  chunk_number: number;
  total_chunks: number;
  frames_in_chunk: number;
  progress: number;
  video_data: string; // base64 encoded video chunk
  audio_segment?: string;
}

export interface ChunkStartData {
  session_id: string;
  chunk_number: number;
  total_chunks: number;
}

export interface ChunkCompleteData {
  session_id: string;
  chunk_number: number;
  total_chunks: number;
  frames_in_chunk: number;
  total_frames_generated: number;
  progress: number;
}

// ==================== 进度和状态 ====================

export interface GenerationProgressData {
  session_id: string;
  progress: number;
  frames_generated?: number;
  total_frames?: number;
  status?: string;
}

export interface GenerationCompleteData {
  session_id: string;
  message: string;
  total_frames?: number;
  output_path?: string;
  duration?: number;
}

export interface GenerationErrorData {
  session_id: string;
  error: string;
  message?: string;
}

// ==================== 请求数据 ====================

export interface GenerateStreamingRequest {
  audio_base64: string;
  audio_mime_type: string;
  image_base64: string;
  prompt?: string;
}

// ==================== Socket 事件映射 ====================

export interface ServerToClientEvents {
  connected: (data: { message: string }) => void;
  session_joined: (data: SessionData) => void;
  session_left: (data: SessionData) => void;
  generation_started: (data: GenerationStartedData) => void;
  video_frame: (data: VideoFrameData) => void;
  video_chunk: (data: VideoChunkData) => void;
  chunk_start: (data: ChunkStartData) => void;
  chunk_complete: (data: ChunkCompleteData) => void;
  generation_progress: (data: GenerationProgressData) => void;
  generation_complete: (data: GenerationCompleteData) => void;
  generation_error: (data: GenerationErrorData) => void;
  streaming_start: (data: SessionData) => void;
  video_saved: (data: { output_path: string; session_id: string }) => void;
  video_save_error: (data: { error: string; session_id: string }) => void;
  error: (data: { message: string }) => void;
}

export interface ClientToServerEvents {
  join_session: (data: SessionData) => void;
  leave_session: (data: SessionData) => void;
  generate_streaming_base64: (data: GenerateStreamingRequest) => void;
}

// ==================== 连接状态 ====================

export enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  ERROR = 'error',
}

// ==================== 生成状态 ====================

export enum GenerationStatus {
  IDLE = 'idle',
  PREPARING = 'preparing',
  GENERATING = 'generating',
  COMPLETED = 'completed',
  ERROR = 'error',
}

