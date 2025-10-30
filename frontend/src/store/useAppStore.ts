/**
 * 应用状态管理 Store
 * 使用 Zustand 进行轻量级状态管理
 */

import type {
    ConnectionStatus,
    GenerationStats,
    GenerationStatus,
    MediaFile,
    Message,
    MessageType,
} from '@/types'
import { generateId } from '@/utils/helpers'
import { create } from 'zustand'

interface AppState {
  // 连接状态
  connectionStatus: ConnectionStatus
  socketId: string | null
  
  // 会话
  sessionId: string | null
  
  // 生成状态
  generationStatus: GenerationStatus
  generationProgress: number
  generationStats: GenerationStats | null
  
  // 媒体文件
  audioFile: MediaFile | null
  imageFile: MediaFile | null
  
  // UI 状态
  showAudioControls: boolean
  showVideoControls: boolean
  
  // 消息通知
  messages: Message[]
  
  // 错误
  error: string | null
}

interface AppActions {
  // 连接相关
  setConnectionStatus: (status: ConnectionStatus, socketId?: string) => void
  
  // 会话相关
  setSessionId: (sessionId: string | null) => void
  
  // 生成相关
  setGenerationStatus: (status: GenerationStatus) => void
  setGenerationProgress: (progress: number) => void
  startGeneration: () => void
  completeGeneration: () => void
  resetGeneration: () => void
  updateGenerationStats: (updates: Partial<GenerationStats>) => void
  
  // 媒体文件
  setAudioFile: (file: MediaFile | null) => void
  setImageFile: (file: MediaFile | null) => void
  clearMediaFiles: () => void
  
  // UI 状态
  setShowAudioControls: (show: boolean) => void
  setShowVideoControls: (show: boolean) => void
  
  // 消息通知
  addMessage: (type: MessageType, content: string, duration?: number) => void
  removeMessage: (id: string) => void
  clearMessages: () => void
  
  // 错误处理
  setError: (error: string | null) => void
  clearError: () => void
  
  // 重置
  reset: () => void
}

const initialState: AppState = {
  connectionStatus: 'disconnected' as ConnectionStatus,
  socketId: null,
  sessionId: null,
  generationStatus: 'idle' as GenerationStatus,
  generationProgress: 0,
  generationStats: null,
  audioFile: null,
  imageFile: null,
  showAudioControls: false,
  showVideoControls: false,
  messages: [],
  error: null,
}

export const useAppStore = create<AppState & AppActions>((set, get) => ({
  ...initialState,

  // 连接相关
  setConnectionStatus: (status, socketId) => 
    set({ connectionStatus: status, socketId: socketId || null }),

  // 会话相关
  setSessionId: (sessionId) => set({ sessionId }),

  // 生成相关
  setGenerationStatus: (status) => set({ generationStatus: status }),
  
  setGenerationProgress: (progress) => set({ generationProgress: progress }),
  
  startGeneration: () => {
    set({
      generationStatus: 'preparing' as GenerationStatus,
      generationProgress: 0,
      generationStats: {
        startTime: Date.now(),
        endTime: null,
        duration: null,
        framesGenerated: 0,
        totalFrames: 0,
        averageFps: 0,
        audioSegments: 0,
      },
      error: null,
    })
  },
  
  completeGeneration: () => {
    const stats = get().generationStats
    if (stats) {
      const endTime = Date.now()
      const duration = endTime - stats.startTime
      const averageFps = stats.totalFrames > 0 
        ? (stats.framesGenerated / (duration / 1000)) 
        : 0
      
      set({
        generationStatus: 'completed' as GenerationStatus,
        generationProgress: 100,
        generationStats: {
          ...stats,
          endTime,
          duration,
          averageFps,
        },
      })
    } else {
      set({
        generationStatus: 'completed' as GenerationStatus,
        generationProgress: 100,
      })
    }
  },
  
  resetGeneration: () => {
    set({
      generationStatus: 'idle' as GenerationStatus,
      generationProgress: 0,
      generationStats: null,
      sessionId: null,
    })
  },
  
  updateGenerationStats: (updates) => {
    const currentStats = get().generationStats
    if (currentStats) {
      set({
        generationStats: { ...currentStats, ...updates },
      })
    }
  },

  // 媒体文件
  setAudioFile: (file) => set({ audioFile: file }),
  
  setImageFile: (file) => set({ imageFile: file }),
  
  clearMediaFiles: () => set({ audioFile: null, imageFile: null }),

  // UI 状态
  setShowAudioControls: (show) => set({ showAudioControls: show }),
  
  setShowVideoControls: (show) => set({ showVideoControls: show }),

  // 消息通知
  addMessage: (type, content, duration = 3000) => {
    const message: Message = {
      id: generateId(),
      type,
      content,
      duration,
    }
    set((state) => ({ messages: [...state.messages, message] }))
    
    // 自动移除消息
    if (duration) {
      setTimeout(() => {
        get().removeMessage(message.id)
      }, duration)
    }
  },
  
  removeMessage: (id) => {
    set((state) => ({
      messages: state.messages.filter((msg) => msg.id !== id),
    }))
  },
  
  clearMessages: () => set({ messages: [] }),

  // 错误处理
  setError: (error) => set({ error }),
  
  clearError: () => set({ error: null }),

  // 重置
  reset: () => set(initialState),
}))

