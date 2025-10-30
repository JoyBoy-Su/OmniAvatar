/**
 * WebSocket 服务层
 * 封装 Socket.IO 客户端连接和事件处理
 * 提供类型安全的 API
 */

import type {
    ClientToServerEvents,
    GenerateStreamingRequest,
    ServerToClientEvents
} from '@/types'
import { io, Socket } from 'socket.io-client'

type TypedSocket = Socket<ServerToClientEvents, ClientToServerEvents>

export class SocketService {
  private socket: TypedSocket | null = null
  private serverUrl: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000

  // 事件监听器存储
  private eventListeners = new Map<string, Set<Function>>()

  constructor(serverUrl: string = window.location.origin) {
    this.serverUrl = serverUrl
  }

  /**
   * 连接到 WebSocket 服务器
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // 使用 polling 传输以避免大数据帧问题
        this.socket = io(this.serverUrl, {
          transports: ['polling'],
          upgrade: false,
          reconnection: true,
          reconnectionAttempts: this.maxReconnectAttempts,
          reconnectionDelay: this.reconnectDelay,
        }) as TypedSocket

        // 设置连接事件监听
        this.socket.on('connect', () => {
          console.log('[SocketService] Connected to server', this.socket?.id)
          this.reconnectAttempts = 0
          this.emit('connectionChange', { status: 'connected', socketId: this.socket?.id })
          resolve()
        })

        this.socket.on('disconnect', (reason) => {
          console.log('[SocketService] Disconnected:', reason)
          this.emit('connectionChange', { status: 'disconnected', reason })
        })

        this.socket.on('connect_error', (error) => {
          console.error('[SocketService] Connection error:', error)
          this.reconnectAttempts++
          this.emit('connectionChange', { 
            status: 'error', 
            error: error.message,
            attempt: this.reconnectAttempts 
          })
          
          if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            reject(new Error('Failed to connect after multiple attempts'))
          }
        })

        // 设置服务器事件监听
        this.setupServerEventListeners()
      } catch (error) {
        console.error('[SocketService] Failed to create socket:', error)
        reject(error)
      }
    })
  }

  /**
   * 断开连接
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
    this.eventListeners.clear()
  }

  /**
   * 检查连接状态
   */
  isConnected(): boolean {
    return this.socket?.connected ?? false
  }

  /**
   * 获取 Socket ID
   */
  getSocketId(): string | undefined {
    return this.socket?.id
  }

  /**
   * 加入会话房间
   */
  joinSession(sessionId: string): void {
    if (!this.socket) {
      throw new Error('Socket not connected')
    }
    this.socket.emit('join_session', { session_id: sessionId })
  }

  /**
   * 离开会话房间
   */
  leaveSession(sessionId: string): void {
    if (!this.socket) {
      throw new Error('Socket not connected')
    }
    this.socket.emit('leave_session', { session_id: sessionId })
  }

  /**
   * 发送生成请求
   */
  generateStreaming(request: GenerateStreamingRequest): void {
    if (!this.socket) {
      throw new Error('Socket not connected')
    }
    console.log('[SocketService] Sending generation request')
    this.socket.emit('generate_streaming_base64', request)
  }

  /**
   * 设置服务器事件监听器
   */
  private setupServerEventListeners(): void {
    if (!this.socket) return

    // 连接确认
    this.socket.on('connected', (data) => {
      console.log('[SocketService] Server confirmed connection:', data)
      this.emit('connected', data)
    })

    // 会话相关
    this.socket.on('session_joined', (data) => {
      console.log('[SocketService] Session joined:', data.session_id)
      this.emit('sessionJoined', data)
    })

    this.socket.on('session_left', (data) => {
      console.log('[SocketService] Session left:', data.session_id)
      this.emit('sessionLeft', data)
    })

    // 生成相关
    this.socket.on('generation_started', (data) => {
      console.log('[SocketService] Generation started:', data.session_id)
      this.emit('generationStarted', data)
    })

    this.socket.on('video_frame', (data) => {
      // 打印日志用于调试
      console.log('[SocketService] Received video_frame:', data.frame_number, '/', data.total_frames)
      this.emit('videoFrame', data)
    })

    this.socket.on('video_chunk', (data) => {
      console.log('[SocketService] Video chunk received:', data.chunk_number)
      this.emit('videoChunk', data)
    })

    this.socket.on('chunk_start', (data) => {
      console.log('[SocketService] Chunk start:', data.chunk_number)
      this.emit('chunkStart', data)
    })

    this.socket.on('chunk_complete', (data) => {
      console.log('[SocketService] Chunk complete:', data.chunk_number)
      this.emit('chunkComplete', data)
    })

    this.socket.on('generation_progress', (data) => {
      console.log('[SocketService] Progress:', data.progress + '%')
      this.emit('generationProgress', data)
    })

    this.socket.on('generation_complete', (data) => {
      console.log('[SocketService] Generation complete:', data.session_id)
      this.emit('generationComplete', data)
    })

    this.socket.on('generation_error', (data) => {
      console.error('[SocketService] Generation error:', data.error)
      this.emit('generationError', data)
    })

    this.socket.on('streaming_start', (data) => {
      console.log('[SocketService] Streaming start:', data.session_id)
      this.emit('streamingStart', data)
    })

    this.socket.on('video_saved', (data) => {
      console.log('[SocketService] Video saved:', data.output_path)
      this.emit('videoSaved', data)
    })

    this.socket.on('video_save_error', (data) => {
      console.error('[SocketService] Video save error:', data.error)
      this.emit('videoSaveError', data)
    })

    this.socket.on('error', (data) => {
      console.error('[SocketService] Server error:', data.message)
      this.emit('error', data)
    })
  }

  /**
   * 订阅事件
   */
  on(eventName: string, listener: Function): void {
    if (!this.eventListeners.has(eventName)) {
      this.eventListeners.set(eventName, new Set())
    }
    this.eventListeners.get(eventName)!.add(listener)
  }

  /**
   * 取消订阅事件
   */
  off(eventName: string, listener: Function): void {
    const listeners = this.eventListeners.get(eventName)
    if (listeners) {
      listeners.delete(listener)
    }
  }

  /**
   * 触发事件
   */
  private emit(eventName: string, data?: any): void {
    const listeners = this.eventListeners.get(eventName)
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(data)
        } catch (error) {
          console.error(`[SocketService] Error in event listener for ${eventName}:`, error)
        }
      })
    }
  }

  /**
   * 清除所有事件监听器
   */
  removeAllListeners(): void {
    this.eventListeners.clear()
  }
}

// 导出单例实例
export const socketService = new SocketService()

