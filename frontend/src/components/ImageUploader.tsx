/**
 * 图片上传组件
 */

import { useAppStore } from '@/store/useAppStore'
import type { MediaFile } from '@/types'
import { createImagePreview, formatFileSize, validateImageFile } from '@/utils/helpers'
import { Image as ImageIcon, Upload, X } from 'lucide-react'
import { useCallback, useRef, useState } from 'react'

interface ImageUploaderProps {
  onImageReady?: (file: MediaFile) => void
}

export function ImageUploader({ onImageReady }: ImageUploaderProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { imageFile, setImageFile, addMessage } = useAppStore()

  // 处理文件上传
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const validation = validateImageFile(file, 10 * 1024 * 1024) // 10MB limit
    if (!validation.valid) {
      addMessage('error', validation.error!)
      return
    }

    try {
      const preview = await createImagePreview(file)
      
      const mediaFile: MediaFile = {
        file,
        name: file.name,
        size: file.size,
        type: file.type,
        preview,
      }

      setImageFile(mediaFile)
      setPreviewUrl(preview)
      onImageReady?.(mediaFile)
      addMessage('success', `图片已上传: ${file.name}`)
    } catch (error) {
      console.error('[ImageUploader] Failed to create preview:', error)
      addMessage('error', '图片预览失败')
    }
  }, [setImageFile, onImageReady, addMessage])

  // 清除图片
  const clearImage = useCallback(() => {
    setImageFile(null)
    setPreviewUrl(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [setImageFile])

  // 拖拽上传
  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      // 模拟文件输入事件
      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(file)
      if (fileInputRef.current) {
        fileInputRef.current.files = dataTransfer.files
        handleFileUpload({ target: fileInputRef.current } as any)
      }
    }
  }, [handleFileUpload])

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
  }, [])

  return (
    <div className="space-y-4">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png,image/webp"
        onChange={handleFileUpload}
        className="hidden"
        disabled={!!imageFile}
      />

      {/* 上传区域 */}
      <div
        onClick={() => !imageFile && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className={`relative min-h-[200px] flex flex-col items-center justify-center gap-4 px-6 py-8 rounded-xl border-2 border-dashed transition-all cursor-pointer ${
          imageFile
            ? 'border-green-300 bg-green-50'
            : 'border-gray-300 hover:border-primary-400 hover:bg-primary-50'
        }`}
      >
        {previewUrl ? (
          <>
            <img
              src={previewUrl}
              alt="Preview"
              className="max-w-full max-h-[300px] rounded-lg shadow-md"
            />
            {imageFile && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  clearImage()
                }}
                className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </>
        ) : (
          <>
            <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center">
              <ImageIcon className="w-8 h-8 text-primary-600" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-gray-700">
                点击或拖拽上传图片
              </p>
              <p className="text-xs text-gray-500 mt-1">
                支持 JPG, PNG, WebP 格式
              </p>
            </div>
            <Upload className="w-5 h-5 text-gray-400" />
          </>
        )}
      </div>

      {/* 文件信息 */}
      {imageFile && (
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-700 truncate">
                {imageFile.name}
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                {formatFileSize(imageFile.size)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 提示信息 */}
      <p className="text-xs text-gray-500 text-center">
        支持 JPG, PNG, WebP 格式，最大 10MB
      </p>
    </div>
  )
}

