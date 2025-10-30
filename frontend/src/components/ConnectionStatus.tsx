/**
 * 连接状态指示器组件
 */

import { useAppStore } from '@/store/useAppStore'
import type { ConnectionStatus as Status } from '@/types'
import { Wifi, WifiOff } from 'lucide-react'

export function ConnectionStatus() {
  const { connectionStatus, socketId } = useAppStore()

  const getStatusConfig = (status: Status) => {
    switch (status) {
      case 'connected':
        return {
          icon: Wifi,
          text: '已连接',
          className: 'bg-green-500 text-white',
        }
      case 'connecting':
        return {
          icon: Wifi,
          text: '连接中...',
          className: 'bg-yellow-500 text-white animate-pulse',
        }
      case 'disconnected':
        return {
          icon: WifiOff,
          text: '未连接',
          className: 'bg-red-500 text-white',
        }
      case 'error':
        return {
          icon: WifiOff,
          text: '连接错误',
          className: 'bg-red-600 text-white',
        }
    }
  }

  const config = getStatusConfig(connectionStatus)
  const Icon = config.icon

  return (
    <div
      className={`fixed top-6 right-6 px-4 py-2 rounded-full text-sm font-medium flex items-center gap-2 shadow-lg z-50 ${config.className}`}
      title={socketId ? `Socket ID: ${socketId}` : undefined}
    >
      <Icon className="w-4 h-4" />
      <span>{config.text}</span>
    </div>
  )
}

