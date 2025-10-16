#!/bin/bash

# Start the causal pipelined streaming server
echo "Starting OmniAvatar Causal Pipelined Streaming Server..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="/data2/jdsu/projects/OmniAvatar"

# Start server in causal mode
python pipelined_websocket_streaming_server.py --config configs/causal_inference.yaml

echo "Server stopped."
