#!/bin/bash
CUDA_VISIBLE_DEVICE=0,1,2 python pipelined_websocket_streaming_server.py --config configs/inference_1.3B.yaml