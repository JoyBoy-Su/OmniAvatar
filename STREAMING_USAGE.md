# OmniAvatar 流式视频生成服务器使用指南

本实现基于Flask框架，提供了流式视频生成功能，可以在生成视频片段的同时实时返回给客户端。

## 关键特性

1. **流式生成**: 每个视频片段生成完成后立即返回，无需等待整个视频完成
2. **两种流式方式**: Server-Sent Events (SSE) 和 Chunked Transfer Encoding
3. **实时进度**: 客户端可以实时接收生成进度和片段数据
4. **并发处理**: 支持边生成下一个片段边返回当前片段

## 文件说明

- `streaming_server.py` - 基于Flask的流式服务器
- `streaming_client.py` - 支持流式接收的客户端
- `client.py` - 原始的非流式客户端（兼容性）

## 服务器API端点

### 1. `/health` (GET)
检查服务器健康状态和模型加载状态

### 2. `/generate-stream` (POST) 
使用Server-Sent Events进行流式视频生成

### 3. `/generate-stream-chunked` (POST)
使用Chunked Transfer Encoding进行流式视频生成

### 4. `/generate` (POST)
传统非流式生成（兼容性接口）

## 使用方法

### 启动服务器
```bash
python streaming_server.py
```

### 使用流式客户端
```bash
# 基础文本到视频流式生成
python streaming_client.py \
    --prompt "A person speaking with natural facial expressions" \
    --output-dir streaming_output \
    --concatenate

# 包含音频和图像的流式生成
python streaming_client.py \
    --prompt "A person lip-syncing to the audio" \
    --audio input_audio.wav \
    --image reference_image.jpg \
    --output-dir streaming_output \
    --steps 4 \
    --concatenate
```

### 使用curl测试SSE流式接口
```bash
curl -X POST http://localhost:8080/generate-stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A person speaking",
    "height": 720,
    "width": 720,
    "num_steps": 4
  }' \
  --no-buffer
```

## 流式响应格式

### Server-Sent Events 格式
```
event: start
data: {"message": "Video generation started"}

event: segment
data: {"segment_index": 0, "total_segments": 3, "segment_base64": "...", "is_final": false}

event: segment
data: {"segment_index": 1, "total_segments": 3, "segment_base64": "...", "is_final": false}

event: complete
data: {"message": "Video generation completed", "total_segments": 3}
```

### Chunked Transfer 格式
```json
{"status": "started", "message": "Video generation started", "streaming": true}
{"status": "segment", "segment_index": 0, "total_segments": 3, "segment_base64": "..."}
{"status": "segment", "segment_index": 1, "total_segments": 3, "segment_base64": "..."}
{"status": "completed", "message": "Video generation completed", "total_segments": 3, "final": true}
```

## 核心实现原理

### 服务器端流式生成
```python
def forward_streaming(self, ...):
    """生成器函数，每生成一个片段就yield返回"""
    for t in range(times):  # times是总片段数
        # 生成当前片段
        frames = self.pipe.log_video(...)
        
        # 转换为base64
        segment_base64 = self.frames_to_base64(frames, t)
        
        # 返回片段信息
        yield {
            "segment_index": t,
            "total_segments": times,
            "segment_base64": segment_base64,
            "is_final": t == times - 1
        }
```

### Flask流式响应
```python
@app.route('/generate-stream', methods=['POST'])
def generate_video_stream():
    def generate():
        for result in model_pipeline.forward_streaming(...):
            if result.get("complete"):
                yield f"event: complete\ndata: {json.dumps(...)}\n\n"
            else:
                yield f"event: segment\ndata: {json.dumps(...)}\n\n"
    
    return Response(generate(), content_type='text/event-stream')
```

### 客户端流式接收
```python
def handle_streaming_response(response):
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if data.get('segment_base64'):
                # 立即保存并处理片段
                save_segment(data['segment_base64'])
```

## 优势

1. **低延迟**: 用户可以立即看到第一个生成的视频片段
2. **内存效率**: 服务器不需要存储整个视频，生成完立即释放
3. **用户体验**: 实时进度反馈，避免长时间等待
4. **可扩展性**: 支持长视频生成，不受内存限制

## 注意事项

1. **模型初始化**: 服务器启动时需要时间加载模型
2. **GPU内存**: 确保有足够的GPU内存运行模型
3. **网络稳定**: 流式传输需要稳定的网络连接
4. **临时文件**: 服务器会自动清理临时音频/图像文件

## 故障排除

1. **服务器无法启动**: 检查CUDA设备和模型路径
2. **流式中断**: 检查网络连接和超时设置
3. **内存不足**: 降低批处理大小或使用模型优化
4. **片段生成失败**: 检查输入参数和模型配置