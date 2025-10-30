/**
 * 视频播放器自定义 Hook
 */

import { socketService } from '@/services/SocketService'
import { useAppStore } from '@/store/useAppStore'
import type { VideoFrame, VideoFrameData } from '@/types'
import { useCallback, useEffect, useRef, useState } from 'react'

export function useVideoPlayer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [totalFrames, setTotalFrames] = useState(0)
  const [canvasReady, setCanvasReady] = useState(false)
  const [firstFrame, setFirstFrame] = useState<HTMLImageElement | null>(null)
  
  const frameQueueRef = useRef<VideoFrame[]>([])
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const lastFrameTimeRef = useRef<number>(0)
  
  const { updateGenerationStats } = useAppStore()

  const fps = 16 // 16 fps
  const frameInterval = 1000 / fps

  // 初始化 Canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctxRef.current = ctx
        setCanvasReady(true)
      }
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  // 播放下一帧
  const playNextFrame = useCallback((timestamp: number) => {
    if (!ctxRef.current || !canvasRef.current) return

    // 检查是否到了播放下一帧的时间
    if (timestamp - lastFrameTimeRef.current < frameInterval) {
      if (isPlaying) {
        animationFrameRef.current = requestAnimationFrame(playNextFrame)
      }
      return
    }

    const frameQueue = frameQueueRef.current
    if (frameQueue.length > 0) {
      const frame = frameQueue.shift()!
      
      const img = new Image()
      img.onload = () => {
        const canvas = canvasRef.current!
        const ctx = ctxRef.current!

        // 第一次设置 canvas 尺寸
        if (canvas.width === 0 || canvas.height === 0) {
          canvas.width = img.width
          canvas.height = img.height
        }

        // 保存第一帧
        if (!firstFrame || frame.frameNumber === 1) {
          setFirstFrame(img)
        }

        // 绘制图像
        ctx.drawImage(img, 0, 0)
        
        setCurrentFrame(frame.frameNumber)
        setTotalFrames(frame.totalFrames)
        
        lastFrameTimeRef.current = timestamp
      }

      img.onerror = () => {
        console.error('[VideoPlayer] Failed to load frame:', frame.frameNumber)
      }

      img.src = `data:image/jpeg;base64,${frame.imageData}`
    }

    // 继续播放
    if (isPlaying) {
      animationFrameRef.current = requestAnimationFrame(playNextFrame)
    }
  }, [isPlaying, frameInterval, firstFrame])

  // 开始播放
  const startPlayback = useCallback(() => {
    if (!canvasReady) {
      console.warn('[VideoPlayer] Canvas not ready')
      return
    }

    setIsPlaying(true)
    lastFrameTimeRef.current = performance.now()
    animationFrameRef.current = requestAnimationFrame(playNextFrame)
  }, [canvasReady, playNextFrame])

  // 停止播放
  const stopPlayback = useCallback(() => {
    setIsPlaying(false)
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
  }, [])

  // 重置播放器
  const reset = useCallback(() => {
    stopPlayback()
    frameQueueRef.current = []
    setCurrentFrame(0)
    setTotalFrames(0)
    setFirstFrame(null)
    
    // 清空 canvas
    if (ctxRef.current && canvasRef.current) {
      ctxRef.current.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      canvasRef.current.width = 0
      canvasRef.current.height = 0
    }
  }, [stopPlayback])

  // 显示第一帧
  const showFirstFrame = useCallback(() => {
    if (firstFrame && ctxRef.current && canvasRef.current) {
      ctxRef.current.drawImage(firstFrame, 0, 0)
    }
  }, [firstFrame])

  // 监听视频帧事件
  useEffect(() => {
    const handleVideoFrame = (data: VideoFrameData) => {
      console.log('[VideoPlayer] Received video frame:', data.frame_number, '/', data.total_frames)
      
      // 添加帧到队列
      const frame: VideoFrame = {
        frameNumber: data.frame_number,
        totalFrames: data.total_frames,
        chunkNumber: data.chunk_number,
        imageData: data.frame_data,
        timestamp: Date.now(),
      }
      
      frameQueueRef.current.push(frame)
      console.log('[VideoPlayer] Frame added to queue, queue length:', frameQueueRef.current.length)
      
      // 更新统计
      updateGenerationStats({
        framesGenerated: data.frame_number,
        totalFrames: data.total_frames,
      })

      // 如果还没开始播放，自动开始
      if (!isPlaying && frameQueueRef.current.length > 0) {
        console.log('[VideoPlayer] Auto-starting playback')
        startPlayback()
      }
    }

    socketService.on('videoFrame', handleVideoFrame)

    return () => {
      socketService.off('videoFrame', handleVideoFrame)
    }
  }, [isPlaying, startPlayback, updateGenerationStats])

  return {
    canvasRef,
    isPlaying,
    currentFrame,
    totalFrames,
    canvasReady,
    startPlayback,
    stopPlayback,
    reset,
    showFirstFrame,
  }
}

