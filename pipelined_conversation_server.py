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
from scripts.pipelined_inference import set_seed
# pdb.set_trace()
from scripts.pipelined_causal_inference import PipelinedCausalInferencePipeline
from scripts.pipelined_conversation import PipelinedConversation
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
conversation_pipeline = None
args = None
active_sessions = {}  # session_id -> thread info

def initialize_model():
    """Initialize the pipelined model pipeline"""
    global conversation_pipeline, args
    
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

    # Set CUDA device for causal mode (needs multiple GPUs)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    set_seed(args.seed)
    print("Initializing pipelined causal model pipeline...")
    conversation_pipeline = PipelinedConversation(args)
    print("Pipelined conversation pipeline initialized successfully!")

def streaming_callback(data):
    """Callback function for streaming data from pipelined inference pipeline"""
    try:
        # Get session_id from data
        session_id = data.get('session_id')
        if not session_id:
            print("Warning: No session_id in streaming callback data")
            return

        event_type = data.get('type')
        
        # Send data to frontend via WebSocket
        socketio.emit(event_type, data, room=session_id)

    except Exception as e:
        print(f"[PIPELINED_BACKEND] Error in streaming callback: {e}")
        traceback.print_exc()

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

def generate_video_streaming(session_id, prompt, audio_path=None, image_path=None):
    """Generate video in pipelined streaming mode"""
    global conversation_pipeline, args
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
        # import pdb; pdb.set_trace()
        audio_data, sample_rate = sf.read(final_audio_path)
        duration: float = audio_data.size / sample_rate
        frames = int(duration * 16)
        print(f"audio_data: {audio_data.size}, duration: {duration}, frames: {frames}")
        with torch.no_grad():
            # Use causal pipeline
            if frames % 12 == 0:
                num_blocks = frames // 12
            else:
                num_blocks = frames // 12 + 1
            noise = torch.randn([1, num_blocks * 3, 16, 50, 90], device="cuda", dtype=torch.bfloat16)
            results = conversation_pipeline(
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
            mode_prefix = ""
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

        # Send completion event
        completion_data = {
            'type': 'generation_complete',
            'session_id': session_id,
            'message': "Causal Pipelined video generation completed"
        }
        
        socketio.emit('generation_complete', completion_data, room=session_id)

        print(f"Pipelined causal streaming generation completed for session {session_id}")

    except Exception as e:
        error_data = {
            'type': 'generation_error',
            'session_id': session_id,
            'error': str(e),
            'message': 'Pipelined video generation failed'
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
    global conversation_pipeline

    print(f"[DEBUG] Received generate_streaming_base64 request")
    current_pipeline = conversation_pipeline
    if current_pipeline is None:
        mode_str = ""
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
            mode_prefix = ""
            session_id = f"{mode_prefix}pipelined_session_{int(torch.rand(1).item() * 1000000)}"

            # Join the session room
            join_room(session_id)

            # Send success response
            response_data = {
                'session_id': session_id,
                'message': 'Pipelined streaming generation started'
            }
            emit('generation_started', response_data)

            # Start generation in background thread
            generation_thread = threading.Thread(
                target=generate_video_streaming,
                args=(session_id, prompt, audio_path, image_path)
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
    
    handle_generate_streaming_base64(data)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global conversation_pipeline
    return {
        "status": "healthy",
        "conversation_loaded": conversation_pipeline is not None,
        "streaming_supported": True,
        "websocket_enabled": True,
        "pipelined_mode": True,
        "conversation_enabled": True,
        "active_sessions": len(active_sessions)
    }

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    global args
    info = {
        "supported_features": ["i2v", "audio", "streaming", "pipelined", "conversation"]
    }
    info.update({
        "model_type": "normal_pipelined"
    })
    
    return info

def main():
    # Initialize model in main process
    mp.set_start_method('spawn', force=True)

    print("Starting OmniAvatar Pipelined WebSocket Streaming Server...")
    

    initialize_model()
    port = 20143  # Changed from 20144 to avoid conflict
    print(f"Starting Flask-SocketIO server on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    main()
