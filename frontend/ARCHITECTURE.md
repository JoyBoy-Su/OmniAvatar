# OmniAvatar Frontend 架构文档

## 概述

本前端应用采用现代化的 React + TypeScript 架构，专为实时视频生成场景设计，具有高内聚低耦合的特点。

## 核心架构

### 1. 分层架构

```
┌─────────────────────────────────────┐
│          View Layer (UI)            │
│    React Components + Tailwind     │
├─────────────────────────────────────┤
│      Business Logic Layer          │
│    Custom Hooks + State Store      │
├─────────────────────────────────────┤
│        Service Layer               │
│   SocketService + AudioService     │
├─────────────────────────────────────┤
│      Data/Type Layer               │
│   TypeScript Types + Interfaces    │
└─────────────────────────────────────┘
```

### 2. 模块划分

#### 组件层 (Components)
- **AudioRecorder**: 音频录制和文件上传
- **AudioControls**: 音频播放控制面板
- **ImageUploader**: 图片上传和预览
- **VideoPlayer**: 视频帧渲染和播放
- **ConnectionStatus**: WebSocket 连接状态
- **MessageNotifications**: 消息通知系统
- **ProgressBar**: 生成进度显示

每个组件都是独立、可复用的，遵循单一职责原则。

#### 业务逻辑层 (Hooks)
- **useSocket**: WebSocket 连接和事件管理
- **useVideoPlayer**: 视频播放逻辑
- **useAudioPlayer**: 音频播放逻辑

自定义 Hooks 封装了所有业务逻辑，组件只负责 UI 渲染。

#### 服务层 (Services)
- **SocketService**: WebSocket 通信封装
  - 连接管理
  - 事件订阅/发布
  - 类型安全的 API
  
- **AudioService**: 音频处理封装
  - Web Audio API 管理
  - 音频队列处理
  - 精确时序控制

#### 状态管理 (Store)
使用 Zustand 进行状态管理：
- 轻量级，零样板代码
- TypeScript 友好
- 易于调试和测试

#### 类型层 (Types)
完整的 TypeScript 类型定义：
- Socket 事件类型
- 媒体数据类型
- 应用状态类型

## 关键技术决策

### 1. 为什么选择 React？
- 组件化开发，易于维护
- 丰富的生态系统
- 优秀的开发者体验
- 强大的社区支持

### 2. 为什么选择 TypeScript？
- 类型安全，减少运行时错误
- 更好的 IDE 支持
- 易于重构
- 自文档化

### 3. 为什么选择 Zustand？
- 比 Redux 更轻量
- API 简洁直观
- 无需 Provider 包裹
- 完美的 TypeScript 支持

### 4. 为什么选择 Tailwind CSS？
- 快速开发
- 一致的设计系统
- 按需生成，体积小
- 易于定制

## 音视频同步机制

### 视频播放流程

```
Socket 接收视频帧
     ↓
添加到帧队列 (frameQueue)
     ↓
requestAnimationFrame 循环
     ↓
按帧率 (16fps) 取出帧
     ↓
加载图片并绘制到 Canvas
     ↓
更新当前帧状态
```

### 音频播放流程

```
Socket 接收音频片段
     ↓
Base64 解码为 ArrayBuffer
     ↓
添加到音频队列 (audioQueue)
     ↓
Web Audio API 解码
     ↓
精确调度播放时间
     ↓
播放音频 Buffer
     ↓
递归处理下一个片段
```

### 同步策略

1. **独立队列管理**
   - 视频帧队列独立管理
   - 音频片段队列独立管理
   - 避免相互阻塞

2. **时间对齐**
   - 音频使用 AudioContext.currentTime 精确调度
   - 视频使用 requestAnimationFrame 固定帧率
   - 通过时间戳进行粗略对齐

3. **缓冲机制**
   - 视频帧缓冲确保流畅播放
   - 音频添加小缓冲避免点击声
   - 动态调整以适应网络波动

## 数据流

```
用户交互
    ↓
App Component
    ↓
Custom Hooks (useSocket, useVideoPlayer, useAudioPlayer)
    ↓
Service Layer (SocketService, AudioService)
    ↓
WebSocket / Web Audio API
    ↓
Backend Server
    ↓
Socket Events
    ↓
Event Handlers in Hooks
    ↓
State Updates (Zustand Store)
    ↓
React Re-render
    ↓
UI Update
```

## 错误处理

### 1. 网络错误
- 自动重连机制（最多 5 次）
- 连接状态实时反馈
- 用户友好的错误提示

### 2. 媒体错误
- 文件格式验证
- 大小限制检查
- 加载失败重试

### 3. 音频错误
- AudioContext 初始化失败处理
- 解码错误恢复
- 播放错误跳过

## 性能优化

### 1. 组件优化
- React.memo 防止不必要的重渲染
- useCallback 缓存回调函数
- useMemo 缓存计算结果

### 2. 资源优化
- 图片懒加载
- 按需加载组件
- Tailwind CSS 按需生成

### 3. 渲染优化
- requestAnimationFrame 控制帧率
- Canvas 离屏渲染（如需要）
- 虚拟滚动（长列表场景）

### 4. 内存管理
- 及时清理事件监听器
- 释放 AudioContext 资源
- 清空队列和缓存

## 测试策略

### 1. 单元测试
- 工具函数测试
- Hooks 逻辑测试
- 服务层测试

### 2. 组件测试
- React Testing Library
- 用户交互测试
- 快照测试

### 3. 集成测试
- Socket 通信测试
- 音视频播放测试
- 端到端流程测试

## 部署策略

### 1. 开发环境
```bash
npm run dev
```
- Vite 开发服务器
- HMR 热更新
- Source Map

### 2. 生产构建
```bash
npm run build
```
- 代码压缩
- Tree Shaking
- 资源优化

### 3. 部署
- 静态文件部署到 CDN
- 或与后端集成部署
- 配置反向代理

## 扩展性考虑

### 1. 新增功能
- 模块化设计便于添加新功能
- Hooks 可复用于新组件
- 服务层易于扩展

### 2. 国际化
- 预留 i18n 接口
- 字符串统一管理
- 多语言支持

### 3. 主题定制
- Tailwind 配置易于定制
- CSS 变量支持
- 动态主题切换

## 维护指南

### 1. 代码规范
- ESLint 规则
- Prettier 格式化
- TypeScript 严格模式

### 2. 文档维护
- README 使用说明
- 代码注释
- 类型定义即文档

### 3. 版本管理
- 语义化版本
- CHANGELOG 记录
- Git 分支策略

## 总结

本架构设计遵循以下原则：
1. **高内聚低耦合**: 模块独立，接口清晰
2. **关注点分离**: UI、逻辑、数据分层
3. **类型安全**: TypeScript 全面覆盖
4. **性能优先**: 优化渲染和资源使用
5. **易于维护**: 代码清晰，结构合理
6. **可扩展性**: 便于添加新功能

这个架构适合中小型实时媒体应用，随着项目规模增长可以进一步优化。

