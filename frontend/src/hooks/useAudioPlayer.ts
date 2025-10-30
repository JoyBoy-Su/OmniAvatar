/**
 * 音频播放器自定义 Hook
 */

import { audioService } from '@/services/AudioService'
import { socketService } from '@/services/SocketService'
import { useAppStore } from '@/store/useAppStore'
import type { AudioSegment, VideoFrameData } from '@/types'
import { base64ToArrayBuffer } from '@/utils/helpers'
import { useCallback, useEffect, useState } from 'react'

export function useAudioPlayer() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [volume, setVolume] = useState(0.5)
  const [isMuted, setIsMuted] = useState(false)
  const [initialized, setInitialized] = useState(false)

  const { updateGenerationStats } = useAppStore()

  // 初始化音频服务
  const initialize = useCallback(async () => {
    if (initialized) return

    try {
      await audioService.initialize()
      setInitialized(true)
      console.log('[AudioPlayer] Initialized')
    } catch (error) {
      console.error('[AudioPlayer] Failed to initialize:', error)
    }
  }, [initialized])

  // 开始播放
  const startPlayback = useCallback(async () => {
    await initialize()
    setIsPlaying(true)
  }, [initialize])

  // 停止播放
  const stopPlayback = useCallback(() => {
    audioService.stop()
    setIsPlaying(false)
  }, [])

  // 设置音量
  const changeVolume = useCallback((newVolume: number) => {
    setVolume(newVolume)
    audioService.setVolume(newVolume)
  }, [])

  // 切换静音
  const toggleMute = useCallback(() => {
    const muted = audioService.toggleMute()
    setIsMuted(muted)
  }, [])

  // 重置
  const reset = useCallback(() => {
    audioService.reset()
    setIsPlaying(false)
  }, [])

  // 监听音频片段事件
  useEffect(() => {
    let audioSegmentCount = 0

    const handleVideoFrame = async (data: VideoFrameData) => {
      // 如果有音频片段，添加到播放队列
      if (data.audio_segment) {
        try {
          // 确保音频服务已初始化
          if (!initialized) {
            await initialize()
          }

          const audioData = base64ToArrayBuffer(data.audio_segment)
          const segment: AudioSegment = {
            data: audioData,
            startTime: data.audio_start_time || 0,
            duration: data.audio_duration || 0.0625, // 默认 62.5ms (16fps)
            frameNumber: data.frame_number,
          }

          await audioService.addAudioSegment(segment)
          audioSegmentCount++

          // 更新统计
          updateGenerationStats({ audioSegments: audioSegmentCount })

          // 自动开始播放
          if (!isPlaying) {
            setIsPlaying(true)
          }
        } catch (error) {
          console.error('[AudioPlayer] Failed to process audio segment:', error)
        }
      }
    }

    socketService.on('videoFrame', handleVideoFrame)

    return () => {
      socketService.off('videoFrame', handleVideoFrame)
    }
  }, [initialized, isPlaying, initialize, updateGenerationStats])

  // 清理
  useEffect(() => {
    return () => {
      audioService.dispose()
    }
  }, [])

  return {
    isPlaying,
    volume,
    isMuted,
    initialized,
    initialize,
    startPlayback,
    stopPlayback,
    changeVolume,
    toggleMute,
    reset,
  }
}

