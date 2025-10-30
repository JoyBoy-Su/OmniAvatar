/**
 * 音频控制组件
 */

import { useAudioPlayer } from '@/hooks/useAudioPlayer'
import { useAppStore } from '@/store/useAppStore'
import { RotateCcw, Volume2, VolumeX } from 'lucide-react'

export function AudioControls() {
  const { volume, isMuted, isPlaying, changeVolume, toggleMute, reset } = useAudioPlayer()
  const { showAudioControls, generationStatus } = useAppStore()

  // 只在生成中或已完成时显示
  if (!showAudioControls && generationStatus !== 'generating' && generationStatus !== 'completed') {
    return null
  }

  return (
    <div className="fixed top-20 right-6 bg-black/80 text-white p-4 rounded-xl backdrop-blur-md shadow-xl z-50 min-w-[200px]">
      <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
        <Volume2 className="w-4 h-4" />
        音频控制
      </h4>

      {/* 按钮组 */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={toggleMute}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors text-sm"
        >
          {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          {isMuted ? '取消静音' : '静音'}
        </button>
        
        <button
          onClick={reset}
          className="flex items-center justify-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors text-sm"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* 音量滑块 */}
      <div className="space-y-2">
        <label className="text-xs text-gray-300">音量</label>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={(e) => changeVolume(parseFloat(e.target.value))}
            disabled={isMuted}
            className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
          />
          <span className="text-xs text-gray-300 w-12 text-right">
            {Math.round(volume * 100)}%
          </span>
        </div>
      </div>

      {/* 状态 */}
      <div className="mt-3 pt-3 border-t border-white/20">
        <p className="text-xs text-gray-400">
          {isPlaying ? '🔊 正在播放' : '⏸️ 待播放'}
        </p>
      </div>
    </div>
  )
}

