#!/usr/bin/env python3

import os
import sys
import json
import base64
import tempfile
import traceback
from datetime import datetime
import threading
import queue
import torch
import torch.multiprocessing as mp
from flask import Flask, request, jsonify, Response, stream_template
import numpy as np

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
    sys.argv = ['streaming_server.py', '--config', 'configs/inference.yaml']
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
    model_pipeline = StreamingWanInferencePipeline(args)
    print("Model pipeline initialized successfully!")

class StreamingWanInferencePipeline(WanInferencePipeline):
    """Extended pipeline that supports streaming video generation"""
    
    def forward_streaming(self, 
                         prompt,             
                         image_path=None,    
                         audio_path=None,    
                         seq_len=101,        
                         height=720, 
                         width=720,
                         overlap_frame=None,
                         num_steps=None,
                         negative_prompt=None,
                         guidance_scale=None,
                         audio_scale=None,
                         progress_callback=None):
        """
        Streaming version of forward that yields video segments as they are generated.
        
        Args:
            progress_callback: Optional callback function called after each segment is generated
                             Signature: callback(segment_index, total_segments, video_segment_base64)
        """
        overlap_frame = overlap_frame if overlap_frame is not None else self.args.overlap_frame
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale

        # Process image and audio inputs (same as original)
        if image_path is not None:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = self.match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = self.resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
        else:
            image = None
            select_size = [height, width]

        # Calculate video frames and setup (same as original)
        L = int(args.max_tokens * 16 * 16 * 4 / select_size[0] / select_size[1])
        L = L // 4 * 4 + 1 if L % 4 != 0 else L - 3
        T = (L + 3) // 4

        if self.args.i2v:
            if self.args.random_prefix_frames:
                fixed_frame = overlap_frame
                assert fixed_frame % 4 == 1
            else:
                fixed_frame = 1
            prefix_lat_frame = (3 + fixed_frame) // 4
            first_fixed_frame = 1
        else:
            fixed_frame = 0
            prefix_lat_frame = 0
            first_fixed_frame = 0

        # Process audio (same as original implementation)
        if audio_path is not None and args.use_audio:
            import librosa
            import math
            import torch.nn.functional as F
            
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
            input_values = np.squeeze(self.wav_feature_extractor(audio, sampling_rate=16000).input_values)
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            ori_audio_len = audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
            input_values = input_values.unsqueeze(0)
            
            if audio_len < L - first_fixed_frame:
                audio_len = audio_len + ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
            elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
                audio_len = audio_len + ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
            
            input_values = F.pad(input_values, (0, audio_len * int(self.args.sample_rate / self.args.fps) - input_values.shape[1]), mode='constant', value=0)
            
            with torch.no_grad():
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            
            seq_len = audio_len
            audio_embeddings = audio_embeddings.squeeze(0)
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
        else:
            audio_embeddings = None
            ori_audio_len = seq_len

        # Calculate loop parameters
        times = (seq_len - L + first_fixed_frame) // (L-fixed_frame) + 1
        if times * (L-fixed_frame) + fixed_frame < seq_len:
            times += 1

        # Initialize streaming variables
        video_segments = []
        image_emb = {}
        img_lat = None
        
        if args.i2v:
            self.pipe.load_models_to_device(['vae'])
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
            msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1])
            image_cat = img_lat.repeat(1, 1, T, 1, 1)
            msk[:, :, 1:] = 1
            image_emb["y"] = torch.cat([image_cat, msk], dim=1)

        # Main generation loop - now with streaming
        for t in range(times):
            print(f"[{t+1}/{times}] Generating segment...")
            
            # Audio processing for current segment (same as original)
            audio_emb = {}
            if t == 0:
                overlap = first_fixed_frame
            else:
                overlap = fixed_frame
                image_emb["y"][:, -1:, :prefix_lat_frame] = 0
                
            prefix_overlap = (3 + overlap) // 4
            
            if audio_embeddings is not None:
                if t == 0:
                    audio_tensor = audio_embeddings[:min(L - overlap, audio_embeddings.shape[0])]
                else:
                    audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                    audio_tensor = audio_embeddings[audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])]
                audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                audio_prefix = audio_tensor[-fixed_frame:]
                audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                audio_emb["audio_emb"] = audio_tensor
            else:
                audio_prefix = None
                
            if image is not None and img_lat is None:
                self.pipe.load_models_to_device(['vae'])
                img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
                assert img_lat.shape[2] == prefix_overlap
                
            img_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))], dim=2)
            
            # Generate current segment
            frames, _, latents = self.pipe.log_video(
                lat=img_lat,
                prompt=prompt,
                fixed_frame=prefix_overlap,
                image_emb=image_emb,
                audio_emb=audio_emb,
                negative_prompt=negative_prompt,
                cfg_scale=guidance_scale,
                audio_cfg_scale=audio_scale if audio_scale is not None else guidance_scale,
                num_inference_steps=num_steps,
                tea_cache_l1_thresh=args.tea_cache_l1_thresh,
                tea_cache_model_id="Wan2.1-T2V-14B",
                return_latent=True
            )
            
            img_lat = None
            image = (frames[:, -fixed_frame:].clip(0, 1) * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()
            
            # Process current segment
            if t == 0:
                current_segment = frames
                video_segments.append(frames)
            else:
                current_segment = frames[:, overlap:]
                video_segments.append(current_segment)
            
            # Convert segment to base64 for streaming
            segment_base64 = self.frames_to_base64(current_segment, t)
            
            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(t, times, segment_base64)
                except Exception as e:
                    print(f"Error in progress callback: {e}")
            
            # Yield segment information
            yield {
                "segment_index": t,
                "total_segments": times,
                "segment_base64": segment_base64,
                "is_final": t == times - 1
            }
        
        # Return final concatenated video
        final_video = torch.cat(video_segments, dim=1)
        final_video = final_video[:, :ori_audio_len + 1]
        
        yield {
            "segment_index": -1,  # Final result marker
            "total_segments": times,
            "final_video": final_video,
            "is_final": True,
            "complete": True
        }

    def frames_to_base64(self, frames, segment_idx):
        """Convert video frames tensor to base64 encoded video"""
        try:
            # Create temporary file for segment
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = f'demo_out/streaming_temp_{timestamp}'
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save segment as video
            save_video_as_grid_and_mp4(
                frames, 
                temp_dir, 
                args.fps, 
                prompt=f"segment_{segment_idx}",
                prefix=f'segment_{segment_idx}'
            )
            
            # Find generated video file
            video_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp4') and f'segment_{segment_idx}' in f]
            if video_files:
                video_path = os.path.join(temp_dir, video_files[0])
                with open(video_path, 'rb') as f:
                    video_base64 = base64.b64encode(f.read()).decode('utf-8')
                # Clean up temp file
                os.remove(video_path)
                return video_base64
            else:
                return None
        except Exception as e:
            print(f"Error converting frames to base64: {e}")
            return None
    
    def match_size(self, image_size, h, w):
        """Helper method from original inference.py"""
        ratio_ = 9999
        size_ = 9999
        select_size = None
        for image_s in image_size:
            ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
            size_tmp = abs(max(image_s) - max(w, h))
            if ratio_tmp < ratio_:
                ratio_ = ratio_tmp
                size_ = size_tmp
                select_size = image_s
            if ratio_ == ratio_tmp:
                if size_ == size_tmp:
                    select_size = image_s
        return select_size
    
    def resize_pad(self, image, ori_size, tgt_size):
        """Helper method from original inference.py"""
        import torchvision.transforms as transforms
        import torch.nn.functional as F
        
        h, w = ori_size
        scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
        scale_h = int(h * scale_ratio)
        scale_w = int(w * scale_ratio)
        
        image = transforms.Resize(size=[scale_h, scale_w])(image)
        
        padding_h = tgt_size[0] - scale_h
        padding_w = tgt_size[1] - scale_w
        pad_top = padding_h // 2
        pad_bottom = padding_h - pad_top
        pad_left = padding_w // 2
        pad_right = padding_w - pad_left
        
        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return image

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_pipeline
    return jsonify({
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "streaming_supported": True
    })

@app.route('/generate', methods=['POST'])
def generate_video():
    """Traditional non-streaming video generation"""
    global model_pipeline, args
    
    if model_pipeline is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Process similar to original server.py implementation
        # (Implementation details omitted for brevity)
        
        return jsonify({
            "success": True,
            "message": "Non-streaming generation not implemented in this example"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-stream', methods=['POST'])
def generate_video_stream():
    """Streaming video generation with Server-Sent Events"""
    global model_pipeline, args
    
    if model_pipeline is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        def generate():
            try:
                yield f"event: start\ndata: {json.dumps({'message': 'Video generation started'})}\n\n"
                
                # Prepare generation parameters
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
                    if audio_base64:
                        audio_data = base64.b64decode(audio_base64)
                        audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        audio_temp.write(audio_data)
                        audio_temp.close()
                        audio_path = audio_temp.name
                        temp_files.append(audio_path)
                    
                    if image_base64:
                        image_data = base64.b64decode(image_base64)
                        image_temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        image_temp.write(image_data)
                        image_temp.close()
                        image_path = image_temp.name
                        temp_files.append(image_path)
                    
                    # Stream video generation
                    with torch.no_grad():
                        for result in model_pipeline.forward_streaming(
                            prompt=prompt,
                            image_path=image_path,
                            audio_path=audio_path,
                            height=height,
                            width=width,
                            num_steps=num_steps,
                            guidance_scale=guidance_scale,
                            audio_scale=audio_scale,
                            negative_prompt=negative_prompt
                        ):
                            if result.get("complete"):
                                # Final result
                                yield f"event: complete\ndata: {json.dumps({'message': 'Video generation completed', 'total_segments': result.get('total_segments')})}\n\n"
                                break
                            else:
                                # Segment result
                                segment_data = {
                                    "segment_index": result.get("segment_index"),
                                    "total_segments": result.get("total_segments"),
                                    "segment_base64": result.get("segment_base64"),
                                    "is_final": result.get("is_final")
                                }
                                yield f"event: segment\ndata: {json.dumps(segment_data)}\n\n"
                
                finally:
                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                            
            except Exception as e:
                traceback.print_exc()
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate(), 
                       content_type='text/event-stream',
                       headers={
                           'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'
                       })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-stream-chunked', methods=['POST'])
def generate_video_stream_chunked():
    """Alternative streaming implementation using chunked transfer encoding"""
    global model_pipeline, args
    
    if model_pipeline is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        def generate():
            try:
                # Send initial response
                initial_response = {
                    "status": "started",
                    "message": "Video generation started",
                    "streaming": True
                }
                yield json.dumps(initial_response) + "\n"
                
                # Prepare generation parameters
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
                    if audio_base64:
                        audio_data = base64.b64decode(audio_base64)
                        audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        audio_temp.write(audio_data)
                        audio_temp.close()
                        audio_path = audio_temp.name
                        temp_files.append(audio_path)
                    
                    if image_base64:
                        image_data = base64.b64decode(image_base64)
                        image_temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        image_temp.write(image_data)
                        image_temp.close()
                        image_path = image_temp.name
                        temp_files.append(image_path)
                    
                    # Stream video generation
                    with torch.no_grad():
                        for result in model_pipeline.forward_streaming(
                            prompt=prompt,
                            image_path=image_path,
                            audio_path=audio_path,
                            height=height,
                            width=width,
                            num_steps=num_steps,
                            guidance_scale=guidance_scale,
                            audio_scale=audio_scale,
                            negative_prompt=negative_prompt
                        ):
                            if result.get("complete"):
                                # Final result
                                final_response = {
                                    "status": "completed",
                                    "message": "Video generation completed",
                                    "total_segments": result.get("total_segments"),
                                    "final": True
                                }
                                yield json.dumps(final_response) + "\n"
                                break
                            else:
                                # Segment result
                                segment_response = {
                                    "status": "segment",
                                    "segment_index": result.get("segment_index"),
                                    "total_segments": result.get("total_segments"),
                                    "segment_base64": result.get("segment_base64"),
                                    "is_final": result.get("is_final")
                                }
                                yield json.dumps(segment_response) + "\n"
                
                finally:
                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                            
            except Exception as e:
                traceback.print_exc()
                error_response = {
                    "status": "error",
                    "error": str(e)
                }
                yield json.dumps(error_response) + "\n"
        
        return Response(generate(), 
                       content_type='application/json',
                       headers={
                           'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'
                       })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    # Initialize model in main process
    mp.set_start_method('spawn', force=True)
    
    print("Starting OmniAvatar Flask streaming server...")
    initialize_model()
    
    print("Starting Flask server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)

if __name__ == '__main__':
    main()