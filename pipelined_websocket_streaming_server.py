#!/usr/bin/env python3

import os
import sys
import base64
import tempfile
import traceback
from datetime import datetime
import threading
import torch
import torch.multiprocessing as mp
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np
import soundfile as sf
from openai import OpenAI

# import pdb;pdb.set_trace()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from OmniAvatar.utils.args_config import parse_args
# pdb.set_trace()
from scripts.pipelined_inference import PipelinedWanInferencePipeline, set_seed
# pdb.set_trace()
from scripts.pipelined_causal_inference import PipelinedCausalInferencePipeline
# pdb.set_trace()
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4

# pdb.set_trace()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = 'omniavatar-pipelined-streaming'
# Increase max_http_buffer_size to handle large base64 images (default is 1MB)
# engineio_logger=True will show debug info
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=100*1024*1024,
    ping_timeout=60,
    ping_interval=25
)

# Global variables
model_pipeline = None
causal_model_pipeline = None
args = None
active_sessions = {}  # session_id -> thread info
use_causal_mode = True  # Flag to switch between normal and causal mode
qwen_omni_talker = None  # Qwen-Omni talker instance

class QwenOmniTalker:
    """Qwen-Omni语音对话处理器 - 支持多轮对话"""
    
    def __init__(self, api_key="sk-63ad221681734d339b8171797204f105", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.system_message = {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        }
        # 存储每个会话的对话历史 {session_id: [messages]}
        self.conversation_history = {}
        # 默认最大历史轮数（可根据需要调整）
        self.max_history_turns = 10
    
    def process_audio_conversation(self, audio_path, session_id=None, prompt="Analyze this audio and respond naturally."):
        """
        处理音频对话，返回回复的音频文件路径（支持多轮对话）

        Args:
            audio_path: 输入音频文件路径
            session_id: 会话ID，用于生成唯一的输出文件名和管理对话历史
            prompt: 文本提示词，默认为分析音频内容

        Returns:
            tuple: (reply_audio_path, reply_text) 回复音频路径和文本内容
        """
        try:
            # 使用默认session_id如果未提供
            if session_id is None:
                session_id = "default"
            
            # 初始化该会话的历史记录（如果不存在）
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            # 读取音频文件并编码为base64
            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            # 构建当前用户消息
            current_user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio_base64}", "format": "wav"}},
                ],
            }
            
            # 构建完整的消息列表：系统消息 + 历史消息 + 当前消息
            messages = [self.system_message] + self.conversation_history[session_id] + [current_user_message]
            
            # 调用Qwen-Omni API
            completion = self.client.chat.completions.create(
                model="qwen3-omni-flash",
                messages=messages,
                modalities=["text", "audio"],
                audio={
                    "voice": "Cherry",  # Cherry, Ethan, Serena, Chelsie is available
                    "format": "wav"
                },
                stream=True,
                stream_options={"include_usage": True}
            )
            
            # 收集响应
            text_parts = []
            audio_string = ""
            
            for chunk in completion:
                if chunk.choices:
                    if hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio:
                        try:
                            if "data" in chunk.choices[0].delta.audio:
                                audio_string += chunk.choices[0].delta.audio["data"]
                            elif "transcript" in chunk.choices[0].delta.audio:
                                text_parts.append(chunk.choices[0].delta.audio["transcript"])
                        except Exception as e:
                            print(f"Error processing audio chunk: {e}")
                    elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        text_parts.append(chunk.choices[0].delta.content)
                else:
                    if hasattr(chunk, 'usage') and chunk.usage:
                        print(f"Usage: {chunk.usage}")
            
            reply_text = "".join(text_parts)
            print(f"Qwen-Omni reply text: {reply_text}")
            
            # 保存对话历史：将用户消息和助手回复添加到历史记录
            # 用户消息（包含文本和音频数据）
            user_history_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio_base64}", "format": "wav"}},
                ]
            }
            
            # 助手回复消息
            assistant_history_message = {
                "role": "assistant",
                "content": reply_text
            }
            
            # 添加到历史记录
            self.conversation_history[session_id].append(user_history_message)
            self.conversation_history[session_id].append(assistant_history_message)
            
            # 限制历史记录长度（保留最近的 max_history_turns 轮对话）
            # 每轮对话包含2条消息（用户+助手），所以总共保留 max_history_turns * 2 条消息
            max_messages = self.max_history_turns * 2
            if len(self.conversation_history[session_id]) > max_messages:
                self.conversation_history[session_id] = self.conversation_history[session_id][-max_messages:]
            
            print(f"Session {session_id} history: {len(self.conversation_history[session_id])} messages")
            
            # 保存音频文件
            if audio_string:
                wav_bytes = base64.b64decode(audio_string)
                wav_array = np.frombuffer(wav_bytes, dtype=np.int16)
                
                # 生成唯一的输出文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_suffix = f"_{session_id}" if session_id else ""
                reply_audio_path = f"demo_out/qwen_omni_reply_{timestamp}{session_suffix}.wav"
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(reply_audio_path), exist_ok=True)
                
                # 保存音频文件
                sf.write(reply_audio_path, wav_array, samplerate=24000)
                print(f"Qwen-Omni reply audio saved to: {reply_audio_path}")
                
                return reply_audio_path, reply_text
            else:
                print("Warning: No audio data received from Qwen-Omni")
                return None, reply_text
                
        except Exception as e:
            print(f"Error in Qwen-Omni conversation: {e}")
            traceback.print_exc()
            return None, None
    
    def clear_session_history(self, session_id=None):
        """
        清除指定会话的对话历史
        
        Args:
            session_id: 会话ID，如果为None则清除所有会话历史
        """
        if session_id is None:
            self.conversation_history.clear()
            print("All conversation history cleared")
        elif session_id in self.conversation_history:
            del self.conversation_history[session_id]
            print(f"Session {session_id} history cleared")
        else:
            print(f"Session {session_id} not found")
    
    def get_session_history(self, session_id):
        """
        获取指定会话的对话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            list: 对话历史消息列表
        """
        return self.conversation_history.get(session_id, [])
    
    def set_max_history_turns(self, max_turns):
        """
        设置最大历史轮数
        
        Args:
            max_turns: 最大保留的对话轮数
        """
        self.max_history_turns = max_turns
        print(f"Max history turns set to {max_turns}")
    
    def get_session_count(self):
        """
        获取当前活跃会话数量
        
        Returns:
            int: 活跃会话数量
        """
        return len(self.conversation_history)
    
    def list_sessions(self):
        """
        列出所有活跃会话ID
        
        Returns:
            list: 会话ID列表
        """
        return list(self.conversation_history.keys())

def initialize_model(causal_mode=True):
    """Initialize the pipelined model pipeline"""
    global model_pipeline, causal_model_pipeline, args, use_causal_mode, qwen_omni_talker
    
    # Set sys.argv to only include model-related args
    sys.argv = ['pipelined_websocket_streaming_server.py', '--config', 'configs/causal_inference.yaml']
    
    args = parse_args()
    print(f"args: {args}")
    from OmniAvatar.utils.args_config import args as new_args
    print(f"new_args: {new_args}")
    # import pdb; pdb.set_trace()

    # Override some args for server mode
    args.rank = 0
    args.local_rank = 0
    args.debug = True
    args.i2v = True
    args.use_audio = True
    
    # Causal inference specific parameters
    if not hasattr(args, 'num_blocks'):
        args.num_blocks = 7
    if not hasattr(args, 'num_frame_per_block'):
        args.num_frame_per_block = 3
    if not hasattr(args, 'num_transformer_blocks'):
        args.num_transformer_blocks = 30
    if not hasattr(args, 'frame_seq_length'):
        args.frame_seq_length = 1560
    if not hasattr(args, 'independent_first_frame'):
        args.independent_first_frame = False

    use_causal_mode = causal_mode

    # Set CUDA device for causal mode (needs multiple GPUs)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    set_seed(args.seed)
    print("Initializing pipelined causal model pipeline...")
    causal_model_pipeline = PipelinedCausalInferencePipeline(args)
    print("Pipelined causal model pipeline initialized successfully!")
    
    # Initialize Qwen-Omni talker
    print("Initializing Qwen-Omni talker...")
    qwen_omni_talker = QwenOmniTalker()
    print("Qwen-Omni talker initialized successfully!")

def streaming_callback(data):
    """Callback function for streaming data from pipelined inference pipeline"""
    try:
        # Get session_id from data
        session_id = data.get('session_id')
        if not session_id:
            print("Warning: No session_id in streaming callback data")
            return

        event_type = data.get('type')
        
        # 如果是视频帧数据，尝试添加对应的音频片段
        if event_type == 'video_frame' and 'frame_number' in data:
            # 添加音频片段信息到数据中
            add_audio_segment_to_frame(data, session_id)
        elif event_type == 'video_chunk' and 'chunk_number' in data:
            # 为视频块添加音频片段
            add_audio_segment_to_chunk(data, session_id)
        
        # Send data to frontend via WebSocket
        socketio.emit(event_type, data, room=session_id)

    except Exception as e:
        print(f"[PIPELINED_BACKEND] Error in streaming callback: {e}")
        traceback.print_exc()

def add_audio_segment_to_frame(data, session_id):
    # import pdb; pdb.set_trace()
    """为视频帧添加对应的音频片段"""
    try:
        # 获取当前会话的音频信息
        if session_id in active_sessions:
            session_info = active_sessions[session_id]
            audio_path = session_info.get('reply_audio_path')
            
            if audio_path and os.path.exists(audio_path):
                frame_number = data.get('frame_number', 0)
                total_frames = data.get('total_frames', 1)
                
                # 计算音频片段的时间范围（假设25fps）
                fps = 16
                frame_duration = 1.0 / fps
                start_time = (frame_number - 1) * frame_duration
                end_time = frame_number * frame_duration
                
                # 提取音频片段
                audio_segment = extract_audio_segment(audio_path, start_time, end_time)
                if audio_segment:
                    data['audio_segment'] = audio_segment
                    data['audio_start_time'] = start_time
                    data['audio_duration'] = frame_duration
                    
    except Exception as e:
        print(f"Error adding audio segment to frame: {e}")

def add_audio_segment_to_chunk(data, session_id):
    # import pdb; pdb.set_trace()
    """为视频块添加对应的音频片段"""
    try:
        # 获取当前会话的音频信息
        if session_id in active_sessions:
            session_info = active_sessions[session_id]
            audio_path = session_info.get('reply_audio_path')
            
            if audio_path and os.path.exists(audio_path):
                chunk_number = data.get('chunk_number', 0)
                frames_in_chunk = data.get('frames_in_chunk', 3)
                
                # 计算音频片段的时间范围（假设25fps）
                fps = 16
                frame_duration = 1.0 / fps
                chunk_duration = frames_in_chunk * frame_duration
                start_time = chunk_number * chunk_duration
                end_time = start_time + chunk_duration
                
                # 提取音频片段
                audio_segment = extract_audio_segment(audio_path, start_time, end_time)
                if audio_segment:
                    data['audio_segment'] = audio_segment
                    data['audio_start_time'] = start_time
                    data['audio_duration'] = chunk_duration
                    
    except Exception as e:
        print(f"Error adding audio segment to chunk: {e}")

def extract_audio_segment(audio_path, start_time, end_time):
    """提取音频片段并转换为base64"""
    try:
        import soundfile as sf
        
        # 读取音频文件
        audio_data, sample_rate = sf.read(audio_path)
        
        # 计算样本索引
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # 确保索引在有效范围内
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return None
            
        # 提取音频片段
        audio_segment = audio_data[start_sample:end_sample]
        
        # 如果音频是立体声，转换为单声道
        if len(audio_segment.shape) > 1:
            audio_segment = np.mean(audio_segment, axis=1)
        
        # 转换为16位整数
        audio_segment = (audio_segment * 32767).astype(np.int16)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_segment, sample_rate)
            
            # 读取文件内容并转换为base64
            with open(temp_file.name, 'rb') as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # 删除临时文件
            os.unlink(temp_file.name)
            
            return audio_base64
            
    except Exception as e:
        print(f"Error extracting audio segment: {e}")
        return None

def generate_video_streaming(session_id, prompt, audio_path=None, image_path=None, use_conversation=True):
    """Generate video in pipelined streaming mode"""
    global model_pipeline, causal_model_pipeline, args, use_causal_mode, qwen_omni_talker
    # print(f"Input Args: num_steps: {num_steps}, guidance_scale: {guidance_scale}, audio_scale: {audio_scale}")
    # print(f"Causal Args: num_blocks: {num_blocks}, frames_per_block: {frames_per_block}")
    try:
        # Update session status
        active_sessions[session_id] = {
            'status': 'generating',
            'thread': threading.current_thread(),
            'start_time': datetime.now(),
            'audio_path': audio_path  # 保存音频路径用于后续音频片段提取
        }

        # Process audio conversation with Qwen-Omni if enabled and audio is provided
        final_audio_path = audio_path
        conversation_text = None
        # import pdb; pdb.set_trace()
        if use_conversation and audio_path and qwen_omni_talker:
            try:
                # Send conversation processing status
                socketio.emit('conversation_processing', {
                    'type': 'conversation_processing',
                    'session_id': session_id,
                    'message': 'Processing audio conversation with Qwen-Omni...'
                }, room=session_id)
                
                print(f"Processing conversation for session {session_id} with input audio: {audio_path}")
                reply_audio_path, reply_text = qwen_omni_talker.process_audio_conversation(audio_path, session_id)
                
                if reply_audio_path:
                    final_audio_path = reply_audio_path
                    conversation_text = reply_text
                    print(f"Using Qwen-Omni reply audio: {final_audio_path}")
                    active_sessions[session_id]["reply_audio_path"] = reply_audio_path
                    # Send conversation completion status
                    socketio.emit('conversation_complete', {
                        'type': 'conversation_complete',
                        'session_id': session_id,
                        'message': 'Audio conversation completed, starting video generation...',
                        'reply_text': reply_text
                    }, room=session_id)
                else:
                    print("Failed to get reply audio from Qwen-Omni, using original audio")
                    socketio.emit('conversation_warning', {
                        'type': 'conversation_warning',
                        'session_id': session_id,
                        'message': 'Conversation processing failed, using original audio for video generation'
                    }, room=session_id)
                    
            except Exception as e:
                print(f"Error in conversation processing: {e}")
                traceback.print_exc()
                socketio.emit('conversation_error', {
                    'type': 'conversation_error',
                    'session_id': session_id,
                    'message': f'Conversation processing failed: {str(e)}, using original audio'
                }, room=session_id)
        audio_data, sample_rate = sf.read(final_audio_path)
        duration: float = audio_data.size / sample_rate
        frames = int(duration * 16)
        print(f"audio_data: {audio_data.size}, duration: {duration}, frames: {frames}")
        with torch.no_grad():
            # Use causal pipeline
            # if num_blocks is None:
            #     num_blocks = getattr(args, 'num_blocks', 7)
            # if frames_per_block is None:
            #     frames_per_block = getattr(args, 'num_frame_per_block', 3)
            if frames % 12 == 0:
                num_blocks = frames // 12
            else:
                num_blocks = frames // 12 + 1
            noise = torch.randn([1, num_blocks * 3, 16, 50, 90], device="cuda", dtype=torch.bfloat16)
            results = causal_model_pipeline(
                noise=noise,
                text_prompts=prompt,
                image_path=image_path,
                audio_path=final_audio_path,  # Use processed audio from Qwen-Omni
                initial_latent=None,
                return_latents=False,
                streaming_callback=streaming_callback,
                session_id=session_id
            )
            
            # Process causal results
            video = None
            if results:
                all_videos = []
                for result in results:
                    if "video" in result:
                        all_videos.append(result["video"])
                if all_videos:
                    video = torch.cat(all_videos, dim=1)

        # Save final video
        if video is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_prefix = "causal_" if use_causal_mode else ""
            output_dir = f'demo_out/{mode_prefix}pipelined_streaming_output_{timestamp}'
            os.makedirs(output_dir, exist_ok=True)

            save_video_as_grid_and_mp4(
                video,
                output_dir,
                args.fps,
                prompt=prompt,
                audio_path=final_audio_path if args.use_audio else None,
                prefix=f'{mode_prefix}pipelined_streaming_result_{session_id}'
            )
            
            # Save conversation info if available
            if conversation_text:
                conversation_info_path = os.path.join(output_dir, f'conversation_info_{session_id}.txt')
                with open(conversation_info_path, 'w', encoding='utf-8') as f:
                    f.write(f"Session ID: {session_id}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Original Audio: {audio_path}\n")
                    f.write(f"Reply Audio: {final_audio_path}\n")
                    f.write(f"Conversation Reply: {conversation_text}\n")
                    f.write(f"Video Prompt: {prompt}\n")

        # Send completion event
        completion_data = {
            'type': 'generation_complete',
            'session_id': session_id,
            'message': "Causal Pipelined video generation completed"
        }
        
        # Add conversation info if available
        if conversation_text:
            completion_data.update({
                'conversation_reply': conversation_text,
                'used_conversation': True,
                'original_audio': audio_path,
                'reply_audio': final_audio_path
            })
        else:
            completion_data['used_conversation'] = False
        # if use_causal_mode:
        #     completion_data.update({
        #         'total_blocks': num_blocks,
        #         # 'frames_per_block': frames_per_block,
        #         'total_frames': num_blocks * frames_per_block
        #     })
        socketio.emit('generation_complete', completion_data, room=session_id)

        print(f"Pipelined causal streaming generation completed for session {session_id}")

    except Exception as e:
        error_data = {
            'type': 'generation_error',
            'session_id': session_id,
            'error': str(e),
            'message': f'{"Causal " if use_causal_mode else ""}Pipelined video generation failed'
        }
        socketio.emit('generation_error', error_data, room=session_id)
        print("Error in pipelined causal streaming generation for session {session_id}: {e}")
        traceback.print_exc()

    finally:
        # Clean up session
        if session_id in active_sessions:
            active_sessions[session_id]['status'] = 'completed'
            active_sessions[session_id]['end_time'] = datetime.now()

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_sid = request.sid
    print(f"Client connected: {client_sid}")
    emit('connected', {'message': 'Connected to pipelined streaming server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_sid = request.sid
    print(f"Client disconnected: {client_sid}")

@socketio.on('join_session')
def handle_join_session(data):
    """Handle joining a session room"""
    session_id = data.get('session_id')
    client_sid = request.sid
    if session_id:
        join_room(session_id)
        print(f"Client {client_sid} joined session {session_id}")
        emit('session_joined', {'session_id': session_id})
    else:
        emit('error', {'message': 'No session_id provided'})

@socketio.on('leave_session')
def handle_leave_session(data):
    """Handle leaving a session room"""
    session_id = data.get('session_id')
    client_sid = request.sid
    if session_id:
        leave_room(session_id)
        print(f"Client {client_sid} left session {session_id}")
        emit('session_left', {'session_id': session_id})

@socketio.on('generate_streaming_base64')
def handle_generate_streaming_base64(data):
    """Handle pipelined streaming video generation request with base64 data"""
    global model_pipeline, causal_model_pipeline, use_causal_mode

    print(f"[DEBUG] Received generate_streaming_base64 request")
    current_pipeline = causal_model_pipeline if use_causal_mode else model_pipeline
    if current_pipeline is None:
        mode_str = "causal " if use_causal_mode else ""
        emit('error', {'message': f'Pipelined {mode_str}model not initialized'})
        return

    try:
        # Extract parameters
        prompt = data.get('prompt', 'A person speaking naturally with lip sync to the audio')
        if not prompt:
            prompt = 'A person speaking naturally with lip sync to the audio'

        audio_base64 = data.get('audio_base64')
        audio_mime_type = data.get('audio_mime_type', 'audio/wav')  # Get MIME type from frontend
        image_base64 = data.get('image_base64')
        height = data.get('height', 720)
        width = data.get('width', 720)
        num_steps = data.get('num_steps', args.num_steps)
        guidance_scale = data.get('guidance_scale', args.guidance_scale)
        audio_scale = data.get('audio_scale', args.audio_scale)
        negative_prompt = data.get('negative_prompt', args.negative_prompt)

        # Causal-specific parameters
        num_blocks = data.get('num_blocks', getattr(args, 'num_blocks', 7))
        frames_per_block = data.get('frames_per_block', getattr(args, 'num_frame_per_block', 3))

        # Conversation parameters
        use_conversation = data.get('use_conversation', True)  # Enable conversation by default
        
        print(f"num steps: {num_steps}, args.num steps: {args.num_steps}")
        print(f"use_conversation: {use_conversation}")
        if use_causal_mode:
            print(f"Causal params: num_blocks: {num_blocks}, frames_per_block: {frames_per_block}")

        # Create temporary files
        audio_path = None
        image_path = None
        temp_files = []

        try:
            # Handle audio file
            if audio_base64:
                audio_data = base64.b64decode(audio_base64)

                # Determine file extension from MIME type
                extension_map = {
                    'audio/wav': '.wav',
                    'audio/webm': '.webm',
                    'audio/webm;codecs=opus': '.webm',
                    'audio/ogg': '.ogg',
                    'audio/mp3': '.mp3',
                    'audio/mpeg': '.mp3',
                    'audio/mp4': '.mp4',
                    'audio/m4a': '.m4a'
                }
                file_ext = extension_map.get(audio_mime_type, '.webm')

                # Save the audio file with correct extension
                audio_temp = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
                audio_temp.write(audio_data)
                audio_temp.close()
                audio_path_original = audio_temp.name
                temp_files.append(audio_path_original)

                # Convert to WAV if needed (for better compatibility with librosa/soundfile)
                if file_ext != '.wav':
                    try:
                        import ffmpeg
                        audio_path_wav = audio_path_original.replace(file_ext, '.wav')
                        print(f"Converting {file_ext} to WAV: {audio_path_original} -> {audio_path_wav}")

                        # Convert to WAV using ffmpeg
                        stream = ffmpeg.input(audio_path_original)
                        stream = ffmpeg.output(stream, audio_path_wav, acodec='pcm_s16le', ar='24000', ac=1)
                        ffmpeg.run(stream, overwrite_output=True, quiet=True)

                        audio_path = audio_path_wav
                        temp_files.append(audio_path_wav)
                        print(f"Audio converted successfully to WAV")
                    except Exception as e:
                        print(f"Warning: Failed to convert audio to WAV: {e}")
                        print(f"Using original format {file_ext}, librosa should handle it")
                        audio_path = audio_path_original
                else:
                    audio_path = audio_path_original

                print(f"Audio file saved: {audio_path}, original format: {audio_mime_type}")

            # Handle image file
            if image_base64:
                image_data = base64.b64decode(image_base64)
                image_temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                image_temp.write(image_data)
                image_temp.close()
                image_path = image_temp.name
                temp_files.append(image_path)

            # Generate session ID
            mode_prefix = "causal_" if use_causal_mode else ""
            session_id = f"{mode_prefix}pipelined_session_{int(torch.rand(1).item() * 1000000)}"

            # Join the session room
            join_room(session_id)

            # Send success response
            response_data = {
                'session_id': session_id,
                'message': f'{"Causal " if use_causal_mode else ""}Pipelined streaming generation started'
            }
            if use_causal_mode:
                response_data.update({
                    'num_blocks': num_blocks,
                    'frames_per_block': frames_per_block,
                    'total_frames': num_blocks * frames_per_block
                })
            emit('generation_started', response_data)

            # Start generation in background thread
            generation_thread = threading.Thread(
                target=generate_video_streaming,
                args=(session_id, prompt, audio_path, image_path, use_conversation)
            )
            generation_thread.daemon = True
            generation_thread.start()

        except Exception as e:
            emit('error', {'message': f'Failed to start pipelined generation: {str(e)}'})
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    except Exception as e:
        emit('error', {'message': str(e)})
        traceback.print_exc()

@socketio.on('generate_causal_streaming_base64')
def handle_generate_causal_streaming_base64(data):
    """Handle causal pipelined streaming video generation request with base64 data"""
    global use_causal_mode
    
    # Temporarily switch to causal mode for this request
    original_mode = use_causal_mode
    use_causal_mode = True
    
    try:
        handle_generate_streaming_base64(data)
    finally:
        # Restore original mode
        use_causal_mode = original_mode

@socketio.on('switch_mode')
def handle_switch_mode(data):
    """Handle switching between normal and causal mode"""
    global use_causal_mode, model_pipeline, causal_model_pipeline
    
    requested_mode = data.get('mode', 'normal')  # 'normal' or 'causal'
    
    if requested_mode == 'causal':
        if causal_model_pipeline is None:
            emit('error', {'message': 'Causal model not initialized. Please restart server with causal mode.'})
            return
        use_causal_mode = True
        emit('mode_switched', {'mode': 'causal', 'message': 'Switched to causal inference mode'})
    else:
        if model_pipeline is None:
            emit('error', {'message': 'Normal model not initialized. Please restart server with normal mode.'})
            return
        use_causal_mode = False
        emit('mode_switched', {'mode': 'normal', 'message': 'Switched to normal inference mode'})
    
    print(f"Mode switched to: {'causal' if use_causal_mode else 'normal'}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_pipeline, causal_model_pipeline, use_causal_mode, qwen_omni_talker
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "causal_model_loaded": causal_model_pipeline is not None,
        "qwen_omni_loaded": qwen_omni_talker is not None,
        "current_mode": "causal" if use_causal_mode else "normal",
        "streaming_supported": True,
        "websocket_enabled": True,
        "pipelined_mode": True,
        "conversation_enabled": True,
        "active_sessions": len(active_sessions)
    }

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    global args, use_causal_mode, qwen_omni_talker
    info = {
        "current_mode": "causal" if use_causal_mode else "normal",
        "supported_features": ["i2v", "audio", "streaming", "pipelined", "conversation"]
    }
    
    if qwen_omni_talker:
        info["conversation_model"] = "qwen-omni-turbo"
        info["conversation_voices"] = ["Cherry", "Ethan", "Serena", "Chelsie"]
    
    if use_causal_mode:
        info.update({
            "model_type": "causal_pipelined",
            "num_blocks": getattr(args, 'num_blocks', 7),
            "frames_per_block": getattr(args, 'num_frame_per_block', 3),
            "num_transformer_blocks": getattr(args, 'num_transformer_blocks', 30),
            "frame_seq_length": getattr(args, 'frame_seq_length', 1560),
            "independent_first_frame": getattr(args, 'independent_first_frame', False)
        })
        info["supported_features"].append("causal_inference")
    else:
        info.update({
            "model_type": "normal_pipelined"
        })
    
    return info

def main():
    # Initialize model in main process
    mp.set_start_method('spawn', force=True)

    print("Starting OmniAvatar Pipelined WebSocket Streaming Server...")
    

    initialize_model(causal_mode=True)
    port = 20143  # Changed from 20144 to avoid conflict
    print(f"Starting Flask-SocketIO server on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    main()
