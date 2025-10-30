/**
 * 进度条组件
 */

import { useAppStore } from '@/store/useAppStore'

export function ProgressBar() {
  const { generationStatus, generationProgress } = useAppStore()

  if (generationStatus === 'idle' || generationStatus === 'completed') {
    return null
  }

  return (
    <div className="w-full space-y-2">
      <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-300 ease-out"
          style={{ width: `${generationProgress}%` }}
        ></div>
      </div>
      <p className="text-sm text-center text-gray-600">
        {generationStatus === 'preparing' && '准备中...'}
        {generationStatus === 'generating' && `生成中... ${Math.round(generationProgress)}%`}
      </p>
    </div>
  )
}

