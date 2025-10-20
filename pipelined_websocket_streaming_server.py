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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
model_pipeline = None
causal_model_pipeline = None
args = None
active_sessions = {}  # session_id -> thread info
use_causal_mode = True  # Flag to switch between normal and causal mode

def initialize_model(causal_mode=True):
    """Initialize the pipelined model pipeline"""
    global model_pipeline, causal_model_pipeline, args, use_causal_mode
    
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

def generate_video_streaming(session_id, prompt, audio_path=None, image_path=None):
    """Generate video in pipelined streaming mode"""
    global model_pipeline, causal_model_pipeline, args, use_causal_mode
    # print(f"Input Args: num_steps: {num_steps}, guidance_scale: {guidance_scale}, audio_scale: {audio_scale}")
    # print(f"Causal Args: num_blocks: {num_blocks}, frames_per_block: {frames_per_block}")
    try:
        # Update session status
        active_sessions[session_id] = {
            'status': 'generating',
            'thread': threading.current_thread(),
            'start_time': datetime.now()
        }

        with torch.no_grad():
            # Use causal pipeline
            # if num_blocks is None:
            #     num_blocks = getattr(args, 'num_blocks', 7)
            # if frames_per_block is None:
            #     frames_per_block = getattr(args, 'num_frame_per_block', 3)
            noise = torch.randn([1, 21, 16, 50, 90], device="cuda", dtype=torch.bfloat16)
            results = causal_model_pipeline(
                noise=noise,
                text_prompts=prompt,
                image_path=image_path,
                audio_path=audio_path,
                initial_latent=None,
                return_latents=False
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
                audio_path=audio_path if args.use_audio else None,
                prefix=f'{mode_prefix}pipelined_streaming_result_{session_id}'
            )

        # Send completion event
        completion_data = {
            'type': 'generation_complete',
            'session_id': session_id,
            'message': "Causal Pipelined video generation completed"
        }
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

    # print(f"data: {data}")
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
        
        print(f"num steps: {num_steps}, args.num steps: {args.num_steps}")
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_pipeline, causal_model_pipeline, use_causal_mode
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "causal_model_loaded": causal_model_pipeline is not None,
        "current_mode": "causal" if use_causal_mode else "normal",
        "streaming_supported": True,
        "websocket_enabled": True,
        "pipelined_mode": True,
        "active_sessions": len(active_sessions)
    }

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    global args, use_causal_mode
    info = {
        "current_mode": "causal" if use_causal_mode else "normal",
        "supported_features": ["i2v", "audio", "streaming", "pipelined"]
    }
    
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
    port = 20143
    print(f"Starting Flask-SocketIO server on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    main()
