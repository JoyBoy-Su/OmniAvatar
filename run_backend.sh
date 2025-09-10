#!/bin/bash
CUDA_VISIBLE_DEVICE=5,6,7 \
    python pipelined_websocket_streaming_server.py \
    --config configs/inference_1.3B.yaml
