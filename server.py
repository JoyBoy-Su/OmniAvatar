#!/usr/bin/env python3

import os
import sys
import io
import base64
import tempfile
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import torch
import torch.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from OmniAvatar.utils.args_config import parse_args
from scripts.inference import WanInferencePipeline, set_seed
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4

app = Flask(__name__)

# Global variables to hold the model
model_pipeline = None
args = None

def initialize_model():
    """Initialize the model pipeline"""
    global model_pipeline, args
    
    # Parse arguments with default config
    sys.argv = ['server.py', '--config', 'configs/inference.yaml']
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model_pipeline is not None})

@app.route('/generate', methods=['POST'])
def generate_video():
    """Generate video from audio and image"""
    global model_pipeline, args
    
    if model_pipeline is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Required fields
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Optional fields with defaults
        audio_base64 = data.get('audio_base64')
        image_base64 = data.get('image_base64')
        height = data.get('height', 720)
        width = data.get('width', 720)
        num_steps = data.get('num_steps', args.num_steps)
        guidance_scale = data.get('guidance_scale', args.guidance_scale)
        audio_scale = data.get('audio_scale', args.audio_scale)
        negative_prompt = data.get('negative_prompt', args.negative_prompt)
        
        # Create temporary files for audio and image
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
            
            print(f"Generating video with prompt: {prompt}")
            print(f"Audio path: {audio_path}")
            print(f"Image path: {image_path}")
            
            # Generate video
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
                    negative_prompt=negative_prompt
                )
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'demo_out/server_output_{timestamp}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save video
            save_video_as_grid_and_mp4(
                video, 
                output_dir, 
                args.fps, 
                prompt=prompt,
                audio_path=audio_path if args.use_audio else None, 
                prefix='result'
            )
            
            # Find the generated video file
            video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
            if not video_files:
                return jsonify({"error": "Video generation failed - no output file"}), 500
            
            video_path = os.path.join(output_dir, video_files[0])
            
            # Read video file and encode to base64
            with open(video_path, 'rb') as f:
                video_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            return jsonify({
                "success": True,
                "video_base64": video_base64,
                "output_path": video_path,
                "prompt": prompt
            })
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve generated video files"""
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

def main():
    # Initialize model in main process
    mp.set_start_method('spawn', force=True)
    
    print("Starting OmniAvatar server...")
    initialize_model()
    
    print("Starting Flask server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)

if __name__ == '__main__':
    main()