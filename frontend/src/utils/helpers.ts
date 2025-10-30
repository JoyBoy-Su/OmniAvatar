/**
 * 通用工具函数
 */

/**
 * 将 Blob 转换为 Base64
 */
export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      // 移除 data URL 前缀
      const base64 = result.split(',')[1]
      resolve(base64)
    }
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

/**
 * 将 File 转换为 Base64
 */
export function fileToBase64(file: File): Promise<string> {
  return blobToBase64(file)
}

/**
 * 将 Base64 转换为 ArrayBuffer
 */
export function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = atob(base64)
  const len = binaryString.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  return bytes.buffer
}

/**
 * 格式化文件大小
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
}

/**
 * 格式化时间（秒 -> mm:ss）
 */
export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

/**
 * 生成唯一 ID
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

/**
 * 防抖函数
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null
  
  return function(...args: Parameters<T>) {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

/**
 * 节流函数
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false
  
  return function(...args: Parameters<T>) {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

/**
 * 验证音频文件
 */
export function validateAudioFile(file: File, maxSize: number): { valid: boolean; error?: string } {
  const supportedFormats = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/webm']
  
  if (!supportedFormats.some(format => file.type.includes(format.split('/')[1]))) {
    return { valid: false, error: '不支持的音频格式' }
  }
  
  if (file.size > maxSize) {
    return { valid: false, error: `文件大小超过限制（${formatFileSize(maxSize)}）` }
  }
  
  return { valid: true }
}

/**
 * 验证图片文件
 */
export function validateImageFile(file: File, maxSize: number): { valid: boolean; error?: string } {
  const supportedFormats = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
  
  if (!supportedFormats.includes(file.type)) {
    return { valid: false, error: '不支持的图片格式' }
  }
  
  if (file.size > maxSize) {
    return { valid: false, error: `文件大小超过限制（${formatFileSize(maxSize)}）` }
  }
  
  return { valid: true }
}

/**
 * 创建图片预览 URL
 */
export function createImagePreview(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => resolve(e.target?.result as string)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

/**
 * 计算两个时间戳的差值（毫秒）
 */
export function timeDiff(start: number, end: number): number {
  return end - start
}

/**
 * 延迟函数
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

