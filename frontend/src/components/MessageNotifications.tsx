/**
 * 消息通知组件
 */

import { useAppStore } from '@/store/useAppStore'
import type { Message, MessageType } from '@/types'
import { AlertCircle, AlertTriangle, CheckCircle, Info, X } from 'lucide-react'

const getMessageConfig = (type: MessageType) => {
  switch (type) {
    case 'success':
      return {
        icon: CheckCircle,
        className: 'bg-green-50 border-green-200 text-green-800',
        iconClassName: 'text-green-500',
      }
    case 'error':
      return {
        icon: AlertCircle,
        className: 'bg-red-50 border-red-200 text-red-800',
        iconClassName: 'text-red-500',
      }
    case 'warning':
      return {
        icon: AlertTriangle,
        className: 'bg-yellow-50 border-yellow-200 text-yellow-800',
        iconClassName: 'text-yellow-500',
      }
    case 'info':
    default:
      return {
        icon: Info,
        className: 'bg-blue-50 border-blue-200 text-blue-800',
        iconClassName: 'text-blue-500',
      }
  }
}

function MessageItem({ message }: { message: Message }) {
  const { removeMessage } = useAppStore()
  const config = getMessageConfig(message.type)
  const Icon = config.icon

  return (
    <div
      className={`flex items-start gap-3 p-4 rounded-lg border shadow-lg animate-slide-in ${config.className}`}
    >
      <Icon className={`w-5 h-5 flex-shrink-0 ${config.iconClassName}`} />
      <p className="flex-1 text-sm font-medium">{message.content}</p>
      <button
        onClick={() => removeMessage(message.id)}
        className="flex-shrink-0 hover:opacity-70 transition-opacity"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

export function MessageNotifications() {
  const { messages } = useAppStore()

  return (
    <div className="fixed bottom-6 right-6 z-50 space-y-3 max-w-md">
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
    </div>
  )
}

