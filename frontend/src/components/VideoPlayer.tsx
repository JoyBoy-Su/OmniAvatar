/**
 * 视频播放器组件
 */

import { useVideoPlayer } from '@/hooks/useVideoPlayer'
import { useAppStore } from '@/store/useAppStore'
import { Play } from 'lucide-react'
import { useEffect } from 'react'

export function VideoPlayer() {
  const {
    canvasRef,
    isPlaying,
    currentFrame,
    totalFrames,
    canvasReady,
    reset,
    showFirstFrame,
  } = useVideoPlayer()

  const { generationStatus } = useAppStore()

  // 当生成完成时，显示第一帧
  useEffect(() => {
    if (generationStatus === 'completed') {
      setTimeout(() => {
        showFirstFrame()
      }, 1000)
    }
  }, [generationStatus, showFirstFrame])

  // 当开始新的生成时，重置播放器
  useEffect(() => {
    if (generationStatus === 'preparing') {
      reset()
    }
  }, [generationStatus, reset])

  return (
    <div className="relative bg-black rounded-xl overflow-hidden min-h-[400px] flex items-center justify-center">
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className={`max-w-full max-h-full ${canvasReady ? 'block' : 'hidden'}`}
      />

      {/* 占位符 */}
      {!canvasReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
          <Play className="w-16 h-16 mb-4 opacity-50" />
          <p className="text-lg">流式生成的视频将在这里实时显示</p>
        </div>
      )}

      {/* 播放信息覆盖层 */}
      {isPlaying && totalFrames > 0 && (
        <div className="absolute top-4 left-4 bg-black/70 text-white px-4 py-2 rounded-lg text-sm backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <span>
              帧数: {currentFrame} / {totalFrames}
            </span>
            <span>•</span>
            <span>
              进度: {totalFrames > 0 ? Math.round((currentFrame / totalFrames) * 100) : 0}%
            </span>
          </div>
        </div>
      )}

      {/* 播放状态指示器 */}
      {isPlaying && (
        <div className="absolute bottom-4 left-4 bg-green-500/80 text-white px-3 py-1.5 rounded-full text-xs backdrop-blur-sm flex items-center gap-2">
          <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
          正在播放
        </div>
      )}
    </div>
  )
}

