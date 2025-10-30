# OmniAvatar Frontend

现代化的 React + TypeScript 前端应用，用于 OmniAvatar 流式视频生成。

## 特性

- ✨ **现代技术栈**: React 18 + TypeScript + Vite
- 🎨 **美观 UI**: Tailwind CSS + Lucide Icons
- 🔌 **实时通信**: Socket.IO 客户端
- 🎬 **流式播放**: 实时视频帧渲染和音频同步
- 📦 **状态管理**: Zustand 轻量级状态管理
- 🎯 **类型安全**: 完整的 TypeScript 类型定义
- 🔊 **音频控制**: Web Audio API 高性能音频播放
- 📱 **响应式设计**: 支持各种屏幕尺寸

## 架构设计

### 目录结构

```
frontend/
├── src/
│   ├── components/        # React 组件
│   │   ├── AudioRecorder.tsx       # 音频录制组件
│   │   ├── AudioControls.tsx       # 音频控制面板
│   │   ├── ImageUploader.tsx       # 图片上传组件
│   │   ├── VideoPlayer.tsx         # 视频播放器
│   │   ├── ConnectionStatus.tsx    # 连接状态指示器
│   │   ├── MessageNotifications.tsx # 消息通知
│   │   └── ProgressBar.tsx         # 进度条
│   ├── hooks/            # 自定义 Hooks
│   │   ├── useSocket.ts           # Socket 连接 Hook
│   │   ├── useVideoPlayer.ts      # 视频播放 Hook
│   │   └── useAudioPlayer.ts      # 音频播放 Hook
│   ├── services/         # 服务层
│   │   ├── SocketService.ts       # WebSocket 服务
│   │   └── AudioService.ts        # 音频服务
│   ├── store/            # 状态管理
│   │   └── useAppStore.ts         # 应用状态 Store
│   ├── types/            # TypeScript 类型定义
│   │   ├── socket.ts              # Socket 事件类型
│   │   ├── media.ts               # 媒体类型
│   │   └── app.ts                 # 应用类型
│   ├── utils/            # 工具函数
│   │   └── helpers.ts             # 通用辅助函数
│   ├── styles/           # 样式文件
│   │   └── index.css              # 全局样式
│   ├── App.tsx           # 主应用组件
│   └── main.tsx          # 应用入口
├── public/               # 静态资源
├── index.html           # HTML 模板
├── package.json         # 依赖配置
├── tsconfig.json        # TypeScript 配置
├── vite.config.ts       # Vite 配置
└── tailwind.config.js   # Tailwind 配置
```

### 核心设计原则

#### 1. 高内聚低耦合

- **服务层**: `SocketService` 和 `AudioService` 封装所有外部通信
- **组件化**: 每个组件负责单一功能，独立可复用
- **Hooks**: 自定义 Hooks 封装业务逻辑，组件只负责渲染

#### 2. 类型安全

- 完整的 TypeScript 类型定义
- Socket 事件类型化，避免运行时错误
- 严格的编译器检查

#### 3. 状态管理

- Zustand 提供轻量级状态管理
- 单一数据源，避免状态不一致
- 清晰的状态更新逻辑

#### 4. 音视频同步机制

- **视频播放**: 使用 Canvas 绘制帧，requestAnimationFrame 控制帧率
- **音频播放**: Web Audio API 精确控制音频时序
- **同步策略**: 
  - 帧队列管理，确保顺序播放
  - 音频片段精确调度，避免卡顿
  - 独立的播放控制，互不干扰

## 开发指南

### 安装依赖

```bash
npm install
```

### 开发模式

```bash
npm run dev
```

访问 http://localhost:8080

### 生产构建

```bash
npm run build
```

### 预览生产版本

```bash
npm run preview
```

## Socket 接口

### 客户端事件

- `join_session`: 加入会话
- `leave_session`: 离开会话
- `generate_streaming_base64`: 发送生成请求

### 服务端事件

- `connected`: 连接确认
- `generation_started`: 生成开始
- `video_frame`: 视频帧数据（主要模式）
- `video_chunk`: 视频块数据（备用模式）
- `generation_progress`: 生成进度
- `generation_complete`: 生成完成
- `generation_error`: 生成错误
- `error`: 服务器错误

## 技术栈

- **React 18**: UI 框架
- **TypeScript**: 类型安全
- **Vite**: 构建工具
- **Tailwind CSS**: 样式框架
- **Zustand**: 状态管理
- **Socket.IO Client**: WebSocket 通信
- **Lucide React**: 图标库

## 浏览器支持

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## 性能优化

- 懒加载组件
- 防抖/节流处理
- 虚拟滚动（如需要）
- 图片压缩
- 音频/视频流式传输

## 许可

查看项目根目录的 LICENSE 文件

