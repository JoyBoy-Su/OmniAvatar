#!/usr/bin/env python3

import os
import sys
import io
import json
import base64
import tempfile
import traceback
from datetime import datetime
import asyncio
import threading
import queue
import torch
import torch.multiprocessing as mp
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from OmniAvatar.utils.args_config import parse_args
from scripts.inference import WanInferencePipeline, set_seed
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = 'omniavatar-streaming'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
model_pipeline = None
args = None
active_sessions = {}  # session_id -> thread info

def initialize_model():
    """Initialize the model pipeline"""
    global model_pipeline, args

    # Parse arguments with default config
    sys.argv = ['websocket_streaming_server.py', '--config', 'configs/inference_1.3B.yaml']
    args = parse_args()

    # Override some args for server mode
    args.rank = 0
    args.local_rank = 0
    args.debug = True
    args.i2v = True
    args.use_audio = True

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    set_seed(args.seed)
    print("Initializing model pipeline...")
    model_pipeline = WanInferencePipeline(args)
    print("Model pipeline initialized successfully!")

def streaming_callback(data):
    """Callback function for streaming data from inference pipeline"""
    try:
        # Get session_id from data
        session_id = data.get('session_id')
        if not session_id:
            print("Warning: No session_id in streaming callback data")
            return

        event_type = data.get('type')
        print(f"[BACKEND] Sending {event_type} event to session {session_id}")

        if event_type == 'video_frame':
            print(f"[BACKEND] Frame data length: {len(data.get('frame_data', ''))}")
            print(f"[BACKEND] Frame number: {data.get('frame_number')}/{data.get('total_frames')}")
        elif event_type == 'video_saved':
            print(f"[BACKEND] Video saved to: {data.get('output_path')}")
        elif event_type == 'video_save_error':
            print(f"[BACKEND] Video save error: {data.get('error')}")
        elif event_type == 'streaming_start':
            print(f"[BACKEND] Starting streaming transmission: {data.get('total_frames')} frames")

        # Send data to frontend via WebSocket
        socketio.emit(event_type, data, room=session_id)
        print(f"[BACKEND] Successfully sent {event_type} event to session {session_id}")

    except Exception as e:
        print(f"[BACKEND] Error in streaming callback: {e}")
        traceback.print_exc()

def generate_video_streaming(session_id, prompt, audio_path=None, image_path=None,
                           height=720, width=720, num_steps=4, guidance_scale=4.5,
                           audio_scale=None, negative_prompt=""):
    """Generate video in streaming mode"""
    global model_pipeline, args

    try:
        print(f"Starting streaming generation for session {session_id}")

        # Update session status
        active_sessions[session_id] = {
            'status': 'generating',
            'thread': threading.current_thread(),
            'start_time': datetime.now()
        }

        with torch.no_grad():
            video = model_pipeline(
                prompt=prompt,
                image_path=image_path,
                audio_path=audio_path,
                height=height,
                width=width,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                audio_scale=audio_scale,
                negative_prompt=negative_prompt,
                streaming_mode=True,  # 启用流式模式
                streaming_callback=streaming_callback,  # 传入回调函数
                session_id=session_id
            )

        # Save final video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'demo_out/streaming_output_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

        save_video_as_grid_and_mp4(
            video,
            output_dir,
            args.fps,
            prompt=prompt,
            audio_path=audio_path if args.use_audio else None,
            prefix=f'streaming_result_{session_id}'
        )

        print(f"Streaming generation completed for session {session_id}")

    except Exception as e:
        error_data = {
            'type': 'generation_error',
            'session_id': session_id,
            'error': str(e),
            'message': 'Video generation failed'
        }
        socketio.emit('generation_error', error_data, room=session_id)
        print(f"Error in streaming generation for session {session_id}: {e}")
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
    emit('connected', {'message': 'Connected to streaming server'})

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
    """Handle streaming video generation request with base64 data"""
    global model_pipeline

    if model_pipeline is None:
        emit('error', {'message': 'Model not initialized'})
        return

    try:
        # Extract parameters
        prompt = data.get('prompt', 'A person speaking naturally with lip sync to the audio')
        if not prompt:
            prompt = 'A person speaking naturally with lip sync to the audio'

        audio_base64 = data.get('audio_base64')
        image_base64 = data.get('image_base64')
        height = data.get('height', 720)
        width = data.get('width', 720)
        num_steps = data.get('num_steps', args.num_steps)
        guidance_scale = data.get('guidance_scale', args.guidance_scale)
        audio_scale = data.get('audio_scale', args.audio_scale)
        negative_prompt = data.get('negative_prompt', args.negative_prompt)

        # Create temporary files
        audio_path = None
        image_path = None
        temp_files = []

        try:
            # Handle audio file
            if audio_base64:
                audio_data = base64.b64decode(audio_base64)
                audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                audio_temp.write(audio_data)
                audio_temp.close()
                audio_path = audio_temp.name
                temp_files.append(audio_path)

            # Handle image file
            if image_base64:
                image_data = base64.b64decode(image_base64)
                image_temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                image_temp.write(image_data)
                image_temp.close()
                image_path = image_temp.name
                temp_files.append(image_path)

            # Generate session ID
            session_id = f"session_{int(torch.rand(1).item() * 1000000)}"

            # Join the session room
            join_room(session_id)

            # Send success response
            emit('generation_started', {
                'session_id': session_id,
                'message': 'Streaming generation started'
            })

            # Start generation in background thread
            generation_thread = threading.Thread(
                target=generate_video_streaming,
                args=(session_id, prompt, audio_path, image_path, height, width,
                      num_steps, guidance_scale, audio_scale, negative_prompt)
            )
            generation_thread.daemon = True
            generation_thread.start()

        except Exception as e:
            emit('error', {'message': f'Failed to start generation: {str(e)}'})
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    except Exception as e:
        emit('error', {'message': str(e)})
        traceback.print_exc()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_pipeline
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "streaming_supported": True,
        "websocket_enabled": True,
        "active_sessions": len(active_sessions)
    }

def main():
    # Initialize model in main process
    mp.set_start_method('spawn', force=True)

    print("Starting OmniAvatar WebSocket Streaming Server...")
    initialize_model()

    print("Starting Flask-SocketIO server on port 9090...")
    socketio.run(app, host='0.0.0.0', port=9090, debug=False)

if __name__ == '__main__':
    main()
