#!/usr/bin/env python3

import os
import sys
import base64
import argparse
import json
import requests
from typing import Optional

def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_base64_to_file(base64_data: str, output_path: str) -> None:
    """Save base64 data to a file"""
    with open(output_path, 'wb') as f:
        f.write(base64.b64decode(base64_data))

def send_generation_request(
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
    output_path: str = "output_video.mp4"
) -> bool:
    """Send video generation request to the server"""
    
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
        print(f"Sending request to {server_url}/generate...")
        print(f"Prompt: {prompt}")
        
        # Send request
        response = requests.post(f"{server_url}/generate", json=data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("Video generation successful!")
                
                # Save the generated video
                video_base64 = result.get("video_base64")
                if video_base64:
                    save_base64_to_file(video_base64, output_path)
                    print(f"Video saved to: {output_path}")
                
                # Print server output path for reference
                server_path = result.get("output_path")
                if server_path:
                    print(f"Server output path: {server_path}")
                
                return True
            else:
                print(f"Generation failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"Request failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out. The server might be processing a large request.")
        return False
    except requests.exceptions.ConnectionError:
        print("Failed to connect to server. Is the server running?")
        return False
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

def check_server_health(server_url: str) -> bool:
    """Check if the server is healthy"""
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"Server status: {health_data.get('status', 'unknown')}")
            print(f"Model loaded: {health_data.get('model_loaded', False)}")
            return health_data.get('model_loaded', False)
        else:
            print(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="OmniAvatar Client - Generate videos from audio and images")
    parser.add_argument("--server-url", type=str, default="http://localhost:8080", 
                       help="Server URL (default: http://localhost:8080)")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="Text prompt for video generation")
    parser.add_argument("--audio", type=str, 
                       help="Path to audio file (WAV format recommended)")
    parser.add_argument("--image", type=str, 
                       help="Path to reference image file")
    parser.add_argument("--output", type=str, default="output_video.mp4",
                       help="Output video file path (default: output_video.mp4)")
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
    
    # Send generation request
    success = send_generation_request(
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
        output_path=args.output
    )
    
    if success:
        print("Video generation completed successfully!")
        sys.exit(0)
    else:
        print("Video generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()