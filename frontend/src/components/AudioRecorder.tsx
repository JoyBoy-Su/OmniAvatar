/**
 * 音频录制组件
 */

import { useAppStore } from '@/store/useAppStore'
import type { MediaFile } from '@/types'
import { formatFileSize, formatTime, validateAudioFile } from '@/utils/helpers'
import { Mic, Square, Upload } from 'lucide-react'
import { useCallback, useRef, useState } from 'react'

interface AudioRecorderProps {
  onAudioReady?: (file: MediaFile) => void
}

export function AudioRecorder({ onAudioReady }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [duration, setDuration] = useState(0)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const startTimeRef = useRef<number>(0)
  const timerRef = useRef<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { audioFile, setAudioFile, addMessage } = useAppStore()

  // 开始录音
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 24000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })

      const options = { mimeType: 'audio/webm;codecs=opus' }
      const mediaRecorder = new MediaRecorder(stream, options)
      
      audioChunksRef.current = []
      startTimeRef.current = Date.now()

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        setRecordedBlob(blob)
        
        const file: MediaFile = {
          file: new File([blob], 'recording.webm', { type: 'audio/webm' }),
          name: 'recording.webm',
          size: blob.size,
          type: 'audio/webm',
          duration: duration,
        }
        
        setAudioFile(file)
        onAudioReady?.(file)
        
        addMessage('success', `录音完成！时长: ${formatTime(duration)}, 大小: ${formatFileSize(blob.size)}`)
        
        // 停止所有音轨
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start(100)
      mediaRecorderRef.current = mediaRecorder
      setIsRecording(true)
      
      // 启动计时器
      timerRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000)
        setDuration(elapsed)
      }, 1000)

      addMessage('info', '开始录音...')
    } catch (error) {
      console.error('[AudioRecorder] Failed to start recording:', error)
      addMessage('error', '录音失败，请检查麦克风权限')
    }
  }, [duration, setAudioFile, onAudioReady, addMessage])

  // 停止录音
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [isRecording])

  // 切换录音状态
  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording()
    } else {
      setDuration(0)
      startRecording()
    }
  }, [isRecording, startRecording, stopRecording])

  // 处理文件上传
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const validation = validateAudioFile(file, 50 * 1024 * 1024) // 50MB limit
    if (!validation.valid) {
      addMessage('error', validation.error!)
      return
    }

    const mediaFile: MediaFile = {
      file,
      name: file.name,
      size: file.size,
      type: file.type,
    }

    setAudioFile(mediaFile)
    onAudioReady?.(mediaFile)
    addMessage('success', `音频文件已上传: ${file.name}`)
  }, [setAudioFile, onAudioReady, addMessage])

  return (
    <div className="space-y-4">
      {/* 录音按钮 */}
      <button
        onClick={toggleRecording}
        disabled={!!audioFile}
        className={`w-full flex items-center justify-center gap-3 px-6 py-4 rounded-xl font-medium text-white transition-all ${
          isRecording
            ? 'bg-red-500 hover:bg-red-600 animate-pulse-slow'
            : audioFile
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-gradient-to-r from-primary-500 to-primary-600 hover:shadow-lg hover:-translate-y-0.5'
        }`}
      >
        {isRecording ? (
          <>
            <Square className="w-5 h-5" />
            <span>停止录音 ({formatTime(duration)})</span>
          </>
        ) : (
          <>
            <Mic className="w-5 h-5" />
            <span>开始录音</span>
          </>
        )}
      </button>

      {/* 分隔线 */}
      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-gray-200"></div>
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="px-2 bg-white text-gray-500">或</span>
        </div>
      </div>

      {/* 上传按钮 */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileUpload}
        className="hidden"
        disabled={!!audioFile}
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={!!audioFile}
        className={`w-full flex items-center justify-center gap-3 px-6 py-4 rounded-xl font-medium border-2 border-dashed transition-all ${
          audioFile
            ? 'border-gray-300 text-gray-400 cursor-not-allowed'
            : 'border-primary-300 text-primary-600 hover:border-primary-500 hover:bg-primary-50'
        }`}
      >
        <Upload className="w-5 h-5" />
        <span>上传音频文件</span>
      </button>

      {/* 文件信息 */}
      {audioFile && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-800">
                {audioFile.name}
              </p>
              <p className="text-xs text-green-600 mt-1">
                {formatFileSize(audioFile.size)}
              </p>
            </div>
            <button
              onClick={() => setAudioFile(null)}
              className="text-green-600 hover:text-green-800 text-sm font-medium"
            >
              清除
            </button>
          </div>
        </div>
      )}

      {/* 提示信息 */}
      <p className="text-xs text-gray-500 text-center">
        支持 WAV, MP3, M4A, WebM 格式，最大 50MB
      </p>
    </div>
  )
}

