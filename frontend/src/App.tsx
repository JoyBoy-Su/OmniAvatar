/**
 * 主应用组件
 */

import { AudioControls } from '@/components/AudioControls'
import { AudioRecorder } from '@/components/AudioRecorder'
import { ConnectionStatus } from '@/components/ConnectionStatus'
import { ImageUploader } from '@/components/ImageUploader'
import { MessageNotifications } from '@/components/MessageNotifications'
import { ProgressBar } from '@/components/ProgressBar'
import { VideoPlayer } from '@/components/VideoPlayer'
import { useSocket } from '@/hooks/useSocket'
import { useAppStore } from '@/store/useAppStore'
import { blobToBase64 } from '@/utils/helpers'
import { Play, Sparkles } from 'lucide-react'
import { useCallback, useEffect } from 'react'

function App() {
  const { connect, generateVideo, isConnected } = useSocket()
  
  const {
    audioFile,
    imageFile,
    generationStatus,
    setShowAudioControls,
    addMessage,
  } = useAppStore()

  // 初始化连接
  useEffect(() => {
    connect()
  }, [connect])

  // 当生成开始时，显示音频控制
  useEffect(() => {
    if (generationStatus === 'generating') {
      setShowAudioControls(true)
    }
  }, [generationStatus, setShowAudioControls])

  // 处理生成视频
  const handleGenerate = useCallback(async () => {
    if (!audioFile || !imageFile) {
      addMessage('error', '请先选择音频和图片文件')
      return
    }

    if (!isConnected) {
      addMessage('error', '未连接到服务器，请刷新页面')
      return
    }

    try {
      // 转换为 base64
      const audioBase64 = await blobToBase64(audioFile.file)
      const imageBase64 = await blobToBase64(imageFile.file)

      // 发送生成请求
      generateVideo({
        audio_base64: audioBase64,
        audio_mime_type: audioFile.type,
        image_base64: imageBase64,
      })
    } catch (error) {
      console.error('[App] Failed to generate video:', error)
      addMessage('error', '生成视频失败，请重试')
    }
  }, [audioFile, imageFile, isConnected, generateVideo, addMessage])

  const canGenerate = audioFile && imageFile && generationStatus === 'idle'
  const isGenerating = generationStatus === 'generating' || generationStatus === 'preparing'

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50">
      {/* 连接状态 */}
      <ConnectionStatus />

      {/* 音频控制 */}
      <AudioControls />

      {/* 消息通知 */}
      <MessageNotifications />

      {/* 主容器 */}
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* 头部 */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center justify-center gap-3 mb-4">
            <Sparkles className="w-10 h-10 text-primary-500" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-primary-500 to-purple-600 bg-clip-text text-transparent">
              OmniAvatar
            </h1>
          </div>
          <p className="text-xl text-gray-600">
            流式视频生成器 - 实时生成同步的说话视频
          </p>
        </header>

        {/* 主内容区 */}
        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
          {/* 输入区域 */}
          <div className="grid md:grid-cols-2 gap-8 p-8 bg-gradient-to-br from-white to-gray-50">
            {/* 音频输入 */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                <span className="flex items-center justify-center w-8 h-8 bg-primary-100 text-primary-600 rounded-full text-sm font-bold">
                  1
                </span>
                音频输入
              </h3>
              <AudioRecorder />
            </div>

            {/* 图片输入 */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                <span className="flex items-center justify-center w-8 h-8 bg-primary-100 text-primary-600 rounded-full text-sm font-bold">
                  2
                </span>
                图片输入
              </h3>
              <ImageUploader />
            </div>
          </div>

          {/* 生成按钮 */}
          <div className="px-8 py-6 bg-gray-50 border-t border-gray-200">
            <button
              onClick={handleGenerate}
              disabled={!canGenerate || isGenerating}
              className={`w-full flex items-center justify-center gap-3 px-8 py-4 rounded-xl text-lg font-semibold text-white transition-all ${
                !canGenerate || isGenerating
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-primary-500 to-purple-600 hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0'
              }`}
            >
              {isGenerating ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>生成中...</span>
                </>
              ) : (
                <>
                  <Play className="w-6 h-6" />
                  <span>开始流式生成</span>
                </>
              )}
            </button>

            {/* 进度条 */}
            <div className="mt-4">
              <ProgressBar />
            </div>
          </div>

          {/* 视频播放区域 */}
          <div className="p-8 bg-white">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 bg-primary-100 text-primary-600 rounded-full text-sm font-bold">
                3
              </span>
              视频预览
            </h3>
            <VideoPlayer />
          </div>
        </div>

        {/* 页脚 */}
        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>© 2025 OmniAvatar - Powered by React + TypeScript + Vite</p>
        </footer>
      </div>
    </div>
  )
}

export default App

