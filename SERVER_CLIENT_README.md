# OmniAvatar Server-Client Implementation

This implementation provides a server-client architecture for the OmniAvatar video generation model, allowing you to generate videos from audio and image inputs via HTTP API.

## Files Overview

- `server.py` - HTTP server that loads the OmniAvatar model and provides REST API endpoints
- `client.py` - Command-line client to send requests to the server
- `test_server_client.py` - Test script to validate the server-client setup

## Prerequisites

Install required Python packages:
```bash
pip install flask requests
```

Ensure all OmniAvatar dependencies are installed and model checkpoints are available.

## Usage

### 1. Start the Server

```bash
python server.py
```

The server will:
- Load the OmniAvatar model using configurations from `configs/inference.yaml`
- Start listening on `http://localhost:8080`
- Initialize the model (this may take several minutes)

### 2. Use the Client

#### Check Server Health
```bash
python client.py --check-health
```

#### Generate Video from Text Prompt Only
```bash
python client.py --prompt "A person speaking with natural facial expressions" --output output.mp4
```

#### Generate Video with Reference Image
```bash
python client.py \
    --prompt "A person speaking" \
    --image reference_image.jpg \
    --output output.mp4
```

#### Generate Video with Audio and Image
```bash
python client.py \
    --prompt "A person speaking synchronized with the audio" \
    --audio input_audio.wav \
    --image reference_image.jpg \
    --output output.mp4
```

#### Advanced Parameters
```bash
python client.py \
    --prompt "A person speaking" \
    --audio input_audio.wav \
    --image reference_image.jpg \
    --output output.mp4 \
    --height 720 \
    --width 720 \
    --steps 4 \
    --guidance-scale 4.5 \
    --audio-scale 3.0 \
    --negative-prompt "blurred, low quality, distorted"
```

### 3. Test the Implementation

```bash
python test_server_client.py
```

## API Endpoints

### GET /health
Check server health and model status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### POST /generate
Generate video from inputs.

**Request Body:**
```json
{
    "prompt": "A person speaking with natural expressions",
    "audio_base64": "base64_encoded_audio_data",
    "image_base64": "base64_encoded_image_data",
    "height": 720,
    "width": 720,
    "num_steps": 4,
    "guidance_scale": 4.5,
    "audio_scale": 3.0,
    "negative_prompt": "blurred, low quality"
}
```

**Response:**
```json
{
    "success": true,
    "video_base64": "base64_encoded_video_data",
    "output_path": "/path/to/server/output/video.mp4",
    "prompt": "A person speaking with natural expressions"
}
```

## Client Parameters

- `--server-url`: Server URL (default: http://localhost:8080)
- `--prompt`: Text prompt for video generation (required)
- `--audio`: Path to audio file (optional, WAV format recommended)
- `--image`: Path to reference image file (optional)
- `--output`: Output video file path (default: output_video.mp4)
- `--height`: Video height (default: 720)
- `--width`: Video width (default: 720)
- `--steps`: Number of denoising steps (default: 4)
- `--guidance-scale`: Text guidance scale (default: 4.5)
- `--audio-scale`: Audio guidance scale (optional)
- `--negative-prompt`: Negative prompt to avoid certain content (optional)
- `--check-health`: Only check server health status

## Notes

1. **Model Loading**: The server may take several minutes to initialize the model on first startup.

2. **Memory Requirements**: The server requires significant GPU memory to load the OmniAvatar model.

3. **File Formats**: 
   - Audio: WAV format recommended
   - Image: JPG/PNG formats supported
   - Output: MP4 video format

4. **Generation Time**: Video generation time depends on the input parameters and hardware capabilities.

5. **Error Handling**: Both server and client include comprehensive error handling and logging.

## Troubleshooting

1. **Server won't start**: Check CUDA availability and model checkpoint paths in `configs/inference.yaml`

2. **Connection refused**: Ensure the server is running and accessible on the specified port

3. **Out of memory errors**: Reduce batch size or use model optimizations in the configuration

4. **Generation fails**: Check input file formats and ensure prompts are appropriate

5. **Slow generation**: Consider reducing `num_steps` or adjusting model optimization settings