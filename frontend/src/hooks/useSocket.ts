/**
 * Socket 连接自定义 Hook
 */

import { socketService } from '@/services/SocketService'
import { useAppStore } from '@/store/useAppStore'
import type { GenerateStreamingRequest } from '@/types'
import { useCallback, useEffect } from 'react'

export function useSocket() {
  const {
    setConnectionStatus,
    setSessionId,
    setGenerationStatus,
    setGenerationProgress,
    startGeneration,
    completeGeneration,
    updateGenerationStats,
    addMessage,
    setError,
  } = useAppStore()

  // 连接到服务器
  const connect = useCallback(async () => {
    try {
      setConnectionStatus('connecting')
      await socketService.connect()
      setConnectionStatus('connected', socketService.getSocketId())
      addMessage('success', '已连接到服务器')
    } catch (error) {
      console.error('[useSocket] Connection failed:', error)
      setConnectionStatus('error')
      setError('连接失败，请刷新页面重试')
      addMessage('error', '连接失败')
    }
  }, [setConnectionStatus, addMessage, setError])

  // 断开连接
  const disconnect = useCallback(() => {
    socketService.disconnect()
    setConnectionStatus('disconnected')
  }, [setConnectionStatus])

  // 发送生成请求
  const generateVideo = useCallback((request: GenerateStreamingRequest) => {
    if (!socketService.isConnected()) {
      addMessage('error', '未连接到服务器')
      return
    }

    try {
      startGeneration()
      socketService.generateStreaming(request)
    } catch (error) {
      console.error('[useSocket] Failed to send generation request:', error)
      setError('发送请求失败')
      addMessage('error', '发送请求失败')
    }
  }, [startGeneration, addMessage, setError])

  // 设置事件监听
  useEffect(() => {
    // 连接状态变化
    const handleConnectionChange = (data: any) => {
      if (data.status === 'connected') {
        setConnectionStatus('connected', data.socketId)
      } else if (data.status === 'disconnected') {
        setConnectionStatus('disconnected')
        addMessage('warning', '与服务器断开连接')
      } else if (data.status === 'error') {
        setConnectionStatus('error')
        if (data.attempt && data.attempt < 5) {
          addMessage('warning', `连接失败，正在重试... (${data.attempt}/5)`)
        }
      }
    }

    // 生成开始
    const handleGenerationStarted = (data: any) => {
      console.log('[useSocket] Generation started:', data)
      setSessionId(data.session_id)
      setGenerationStatus('generating')
      socketService.joinSession(data.session_id)
      addMessage('success', '开始生成视频')
      
      if (data.estimated_frames) {
        updateGenerationStats({ totalFrames: data.estimated_frames })
      }
    }

    // 生成进度
    const handleGenerationProgress = (data: any) => {
      setGenerationProgress(data.progress)
      if (data.frames_generated) {
        updateGenerationStats({ framesGenerated: data.frames_generated })
      }
    }

    // 生成完成
    const handleGenerationComplete = (data: any) => {
      console.log('[useSocket] Generation complete:', data)
      completeGeneration()
      addMessage('success', '视频生成完成！')
      
      if (data.session_id) {
        socketService.leaveSession(data.session_id)
      }
    }

    // 生成错误
    const handleGenerationError = (data: any) => {
      console.error('[useSocket] Generation error:', data)
      setGenerationStatus('error')
      setError(data.error || '生成失败')
      addMessage('error', `生成失败: ${data.error}`)
      
      if (data.session_id) {
        socketService.leaveSession(data.session_id)
      }
    }

    // 服务器错误
    const handleError = (data: any) => {
      console.error('[useSocket] Server error:', data)
      setError(data.message)
      addMessage('error', data.message)
    }

    // 注册事件监听
    socketService.on('connectionChange', handleConnectionChange)
    socketService.on('generationStarted', handleGenerationStarted)
    socketService.on('generationProgress', handleGenerationProgress)
    socketService.on('generationComplete', handleGenerationComplete)
    socketService.on('generationError', handleGenerationError)
    socketService.on('error', handleError)

    // 清理
    return () => {
      socketService.off('connectionChange', handleConnectionChange)
      socketService.off('generationStarted', handleGenerationStarted)
      socketService.off('generationProgress', handleGenerationProgress)
      socketService.off('generationComplete', handleGenerationComplete)
      socketService.off('generationError', handleGenerationError)
      socketService.off('error', handleError)
    }
  }, [
    setConnectionStatus,
    setSessionId,
    setGenerationStatus,
    setGenerationProgress,
    completeGeneration,
    updateGenerationStats,
    addMessage,
    setError,
  ])

  return {
    connect,
    disconnect,
    generateVideo,
    isConnected: socketService.isConnected(),
  }
}

