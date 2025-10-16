#!/bin/bash

VIDEO=$1
AUDIO=$2
MERGED=$3

echo "Merge ${VIDEO} and ${AUDIO} to ${MERGED}"

ffmpeg -i $VIDEO -i $AUDIO -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -async 1 -shortest -y $MERGED
