#!/usr/bin/env python3

import os
import sys
import base64
import json
import argparse
import requests
import time
from typing import Optional, Callable

def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_base64_to_file(base64_data: str, output_path: str) -> None:
    """Save base64 data to a file"""
    with open(output_path, 'wb') as f:
        f.write(base64.b64decode(base64_data))

def handle_streaming_response(response, segment_callback: Optional[Callable] = None, output_dir: str = "streaming_output"):
    """
    Handle Server-Sent Events streaming response
    
    Args:
        response: requests Response object with stream=True
        segment_callback: Optional callback for each segment: callback(segment_index, total_segments, segment_path)
        output_dir: Directory to save video segments
    """
    
    os.makedirs(output_dir, exist_ok=True)
    segments_received = 0
    total_segments = 0
    
    print("Receiving streaming video generation...")
    
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            # Parse Server-Sent Events format
            if line.startswith('event: '):
                event_type = line[7:].strip()
            elif line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    
                    if event_type == 'start':
                        print(f"✓ {data.get('message', 'Generation started')}")
                    
                    elif event_type == 'segment':
                        segment_index = data.get('segment_index', 0)
                        total_segments = data.get('total_segments', 1)
                        segment_base64 = data.get('segment_base64')
                        is_final = data.get('is_final', False)
                        
                        if segment_base64:
                            # Save segment
                            segment_filename = f"segment_{segment_index:03d}.mp4"
                            segment_path = os.path.join(output_dir, segment_filename)
                            save_base64_to_file(segment_base64, segment_path)
                            
                            segments_received += 1
                            print(f"✓ Segment {segment_index + 1}/{total_segments} received and saved: {segment_path}")
                            
                            # Call segment callback if provided
                            if segment_callback:
                                try:
                                    segment_callback(segment_index, total_segments, segment_path)
                                except Exception as e:
                                    print(f"Error in segment callback: {e}")
                        
                        if is_final:
                            print(f"✓ Final segment received!")
                    
                    elif event_type == 'complete':
                        total_segments = data.get('total_segments', segments_received)
                        print(f"✓ {data.get('message', 'Generation completed')} - Total segments: {total_segments}")
                        break
                    
                    elif event_type == 'error':
                        error_msg = data.get('error', 'Unknown error')
                        print(f"✗ Server error: {error_msg}")
                        return False
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing SSE data: {e}")
                    continue
    
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user")
        return False
    except Exception as e:
        print(f"Error handling streaming response: {e}")
        return False
    
    print(f"\nStreaming completed! {segments_received} segments received in {output_dir}/")
    return True

def send_streaming_generation_request(
    server_url: str,
    prompt: str,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
    height: int = 720,
    width: int = 720,
    num_steps: int = 4,
    guidance_scale: float = 4.5,
    audio_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    output_dir: str = "streaming_output",
    segment_callback: Optional[Callable] = None
) -> bool:
    """Send streaming video generation request to the server"""
    
    # Prepare request data
    data = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale
    }
    
    # Add optional parameters
    if audio_scale is not None:
        data["audio_scale"] = audio_scale
    if negative_prompt is not None:
        data["negative_prompt"] = negative_prompt
    
    # Encode audio file if provided
    if audio_path and os.path.exists(audio_path):
        print(f"Encoding audio file: {audio_path}")
        data["audio_base64"] = encode_file_to_base64(audio_path)
    
    # Encode image file if provided
    if image_path and os.path.exists(image_path):
        print(f"Encoding image file: {image_path}")
        data["image_base64"] = encode_file_to_base64(image_path)
    
    try:
        print(f"Sending streaming request to {server_url}/generate-stream...")
        print(f"Prompt: {prompt}")
        print(f"Output directory: {output_dir}")
        
        # Send streaming request
        response = requests.post(
            f"{server_url}/generate-stream", 
            json=data, 
            stream=True,  # Enable streaming
            timeout=None  # No timeout for streaming
        )
        
        if response.status_code == 200:
            return handle_streaming_response(response, segment_callback, output_dir)
        else:
            print(f"Request failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Failed to connect to server. Is the server running?")
        return False
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

def concatenate_segments(output_dir: str, final_output: str = "final_video.mp4"):
    """
    Concatenate video segments using ffmpeg (if available)
    
    Args:
        output_dir: Directory containing video segments
        final_output: Output path for concatenated video
    """
    
    # Find all segment files
    segment_files = []
    for file in sorted(os.listdir(output_dir)):
        if file.startswith("segment_") and file.endswith(".mp4"):
            segment_files.append(os.path.join(output_dir, file))
    
    if not segment_files:
        print("No segments found to concatenate")
        return False
    
    try:
        import subprocess
        
        # Create temporary file list for ffmpeg
        list_file = os.path.join(output_dir, "segments.txt")
        with open(list_file, 'w') as f:
            for segment_file in segment_files:
                f.write(f"file '{os.path.abspath(segment_file)}'\n")
        
        # Concatenate using ffmpeg
        final_path = os.path.join(output_dir, final_output)
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', list_file,
            '-c', 'copy',  # Copy without re-encoding
            final_path,
            '-y'  # Overwrite output file
        ]
        
        print(f"Concatenating {len(segment_files)} segments...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Final video saved: {final_path}")
            # Clean up
            os.remove(list_file)
            return True
        else:
            print(f"✗ ffmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to concatenate segments automatically.")
        print(f"Segments are available individually in: {output_dir}")
        return False
    except Exception as e:
        print(f"Error concatenating segments: {e}")
        return False

def check_server_health(server_url: str) -> bool:
    """Check if the server is healthy and supports streaming"""
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"Server status: {health_data.get('status', 'unknown')}")
            print(f"Model loaded: {health_data.get('model_loaded', False)}")
            print(f"Streaming supported: {health_data.get('streaming_supported', False)}")
            return health_data.get('model_loaded', False)
        else:
            print(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def segment_callback_example(segment_index: int, total_segments: int, segment_path: str):
    """Example callback function for handling received segments"""
    print(f"  -> Callback: Processing segment {segment_index + 1}/{total_segments}: {segment_path}")
    # You can add custom processing here, e.g.:
    # - Display the segment immediately
    # - Apply post-processing
    # - Stream to another service
    # etc.

def main():
    parser = argparse.ArgumentParser(description="OmniAvatar Streaming Client - Receive video segments in real-time")
    parser.add_argument("--server-url", type=str, default="http://localhost:8080", 
                       help="Server URL (default: http://localhost:8080)")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="Text prompt for video generation")
    parser.add_argument("--audio", type=str, 
                       help="Path to audio file (WAV format recommended)")
    parser.add_argument("--image", type=str, 
                       help="Path to reference image file")
    parser.add_argument("--output-dir", type=str, default="streaming_output",
                       help="Directory to save video segments (default: streaming_output)")
    parser.add_argument("--height", type=int, default=720,
                       help="Video height (default: 720)")
    parser.add_argument("--width", type=int, default=720,
                       help="Video width (default: 720)")
    parser.add_argument("--steps", type=int, default=4,
                       help="Number of denoising steps (default: 4)")
    parser.add_argument("--guidance-scale", type=float, default=4.5,
                       help="Guidance scale for text conditioning (default: 4.5)")
    parser.add_argument("--audio-scale", type=float,
                       help="Audio guidance scale (optional)")
    parser.add_argument("--negative-prompt", type=str,
                       help="Negative prompt to avoid certain content")
    parser.add_argument("--check-health", action="store_true",
                       help="Only check server health status")
    parser.add_argument("--concatenate", action="store_true",
                       help="Automatically concatenate segments into final video using ffmpeg")
    parser.add_argument("--final-output", type=str, default="final_video.mp4",
                       help="Filename for concatenated final video (default: final_video.mp4)")
    
    args = parser.parse_args()
    
    # Check server health
    if args.check_health:
        check_server_health(args.server_url)
        return
    
    # Validate inputs
    if args.audio and not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
        
    if args.image and not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Check server health first
    print("Checking server health...")
    if not check_server_health(args.server_url):
        print("Server is not ready. Please make sure the server is running and the model is loaded.")
        sys.exit(1)
    
    # Set up segment callback if needed
    segment_callback = segment_callback_example if args.concatenate else None
    
    # Send streaming generation request
    success = send_streaming_generation_request(
        server_url=args.server_url,
        prompt=args.prompt,
        audio_path=args.audio,
        image_path=args.image,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        audio_scale=args.audio_scale,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        segment_callback=segment_callback
    )
    
    if success:
        print("✓ Streaming video generation completed successfully!")
        
        # Concatenate segments if requested
        if args.concatenate:
            print("\nConcatenating segments...")
            concat_success = concatenate_segments(args.output_dir, args.final_output)
            if concat_success:
                print("✓ Final concatenated video created!")
            else:
                print("⚠ Concatenation failed, but individual segments are available")
        
        sys.exit(0)
    else:
        print("✗ Streaming video generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()