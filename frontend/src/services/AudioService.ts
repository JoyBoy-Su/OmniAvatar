/**
 * 音频服务层
 * 处理音频播放、录制和同步
 */

import type { AudioSegment } from '@/types'

export class AudioService {
  private audioContext: AudioContext | null = null
  private gainNode: GainNode | null = null
  private audioQueue: AudioSegment[] = []
  private isPlaying = false
  private isProcessing = false // 防止并发处理
  private nextAudioTime = 0
  private currentSource: AudioBufferSourceNode | null = null
  private volume = 0.5
  private isMuted = false

  /**
   * 初始化音频上下文
   */
  async initialize(): Promise<void> {
    if (this.audioContext) return

    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      })

      // 创建音量控制节点
      this.gainNode = this.audioContext.createGain()
      this.gainNode.connect(this.audioContext.destination)
      this.gainNode.gain.value = this.volume

      // 恢复音频上下文（Safari 需要）
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume()
      }

      console.log('[AudioService] Initialized, state:', this.audioContext.state)
    } catch (error) {
      console.error('[AudioService] Failed to initialize:', error)
      throw error
    }
  }

  /**
   * 添加音频片段到队列
   */
  async addAudioSegment(segment: AudioSegment): Promise<void> {
    if (!this.audioContext) {
      await this.initialize()
    }

    this.audioQueue.push(segment)
    console.log('[AudioService] Audio segment added, queue length:', this.audioQueue.length)

    // 如果还没开始播放，启动播放
    if (!this.isPlaying) {
      this.startPlayback()
    } else if (!this.isProcessing) {
      // 如果已经在播放，但没有正在处理队列，需要继续处理
      console.log('[AudioService] Already playing but not processing, triggering queue processing')
      // 延迟一点以确保音频上下文准备好
      setTimeout(() => this.processQueue(), 0)
    }
  }

  /**
   * 开始播放音频队列
   */
  private async startPlayback(): Promise<void> {
    if (!this.audioContext || !this.gainNode) {
      console.error('[AudioService] Audio context not initialized')
      return
    }

    // 确保音频上下文正在运行
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume()
    }

    this.isPlaying = true
    // 初始化下一个音频时间，添加小缓冲以防止点击声
    this.nextAudioTime = this.audioContext.currentTime + 0.05

    console.log('[AudioService] Starting audio playback')
    this.processQueue()
  }

  /**
   * 处理音频队列
   */
  private async processQueue(): Promise<void> {
    if (!this.isPlaying || !this.audioContext || !this.gainNode) {
      console.log('[AudioService] Cannot process queue - isPlaying:', this.isPlaying, 'audioContext:', !!this.audioContext, 'gainNode:', !!this.gainNode)
      return
    }

    // 防止并发处理
    if (this.isProcessing) {
      console.log('[AudioService] Already processing queue, skipping')
      return
    }

    if (this.audioQueue.length === 0) {
      console.log('[AudioService] Audio queue empty, waiting for more segments...')
      return
    }

    this.isProcessing = true
    const segment = this.audioQueue.shift()!
    console.log('[AudioService] Processing audio segment, remaining in queue:', this.audioQueue.length)
    
    try {
      // 解码音频数据
      const audioBuffer = await this.audioContext.decodeAudioData(segment.data.slice())
      
      // 创建音频源
      const source = this.audioContext.createBufferSource()
      source.buffer = audioBuffer
      source.connect(this.gainNode)

      const currentTime = this.audioContext.currentTime
      let scheduledTime: number

      // 如果下一个音频时间已经过去，立即播放
      if (this.nextAudioTime <= currentTime) {
        scheduledTime = currentTime + 0.01 // 10ms 缓冲
        console.warn('[AudioService] Audio segment late, playing immediately')
      } else {
        scheduledTime = this.nextAudioTime
      }

      // 播放音频
      source.start(scheduledTime)
      this.currentSource = source

      // 更新下一个音频时间
      this.nextAudioTime = scheduledTime + audioBuffer.duration

      console.log(`[AudioService] Playing segment, duration: ${audioBuffer.duration.toFixed(3)}s, next at: ${this.nextAudioTime.toFixed(3)}s`)

      // 设置结束回调
      source.onended = () => {
        this.isProcessing = false
        setTimeout(() => this.processQueue(), 10)
      }
    } catch (error) {
      console.error('[AudioService] Error decoding audio:', error)
      this.isProcessing = false
      // 继续处理下一个片段
      setTimeout(() => this.processQueue(), 10)
    }
  }

  /**
   * 停止播放
   */
  stop(): void {
    this.isPlaying = false
    this.isProcessing = false
    this.audioQueue = []
    this.nextAudioTime = 0

    if (this.currentSource) {
      try {
        this.currentSource.stop()
      } catch (e) {
        // 忽略已停止的错误
      }
      this.currentSource = null
    }

    console.log('[AudioService] Playback stopped')
  }

  /**
   * 设置音量 (0-1)
   */
  setVolume(volume: number): void {
    this.volume = Math.max(0, Math.min(1, volume))
    if (this.gainNode && !this.isMuted) {
      this.gainNode.gain.value = this.volume
    }
  }

  /**
   * 获取当前音量
   */
  getVolume(): number {
    return this.volume
  }

  /**
   * 切换静音
   */
  toggleMute(): boolean {
    this.isMuted = !this.isMuted
    if (this.gainNode) {
      this.gainNode.gain.value = this.isMuted ? 0 : this.volume
    }
    return this.isMuted
  }

  /**
   * 获取静音状态
   */
  isMutedState(): boolean {
    return this.isMuted
  }

  /**
   * 重置音频服务
   */
  reset(): void {
    this.stop()
    this.audioQueue = []
  }

  /**
   * 清理资源
   */
  dispose(): void {
    this.stop()
    
    if (this.audioContext) {
      this.audioContext.close()
      this.audioContext = null
    }
    
    this.gainNode = null
    this.audioQueue = []
  }

  /**
   * 获取当前状态
   */
  getState() {
    return {
      isPlaying: this.isPlaying,
      queueLength: this.audioQueue.length,
      volume: this.volume,
      isMuted: this.isMuted,
      contextState: this.audioContext?.state,
    }
  }
}

// 导出单例实例
export const audioService = new AudioService()

