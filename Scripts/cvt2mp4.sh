#!/bin/bash

nb_params=$#
if [ "$nb_params" -ge 1 ]; then
	quality=26
	if [ "$nb_params" -ge 2 ]; then
		quality=$2
	fi
	video_filename=$1
	output_filename=$video_filename".mp4"
	ffmpeg -i "$video_filename" -c:v libx264 -preset slow -crf $quality -an "$output_filename"
fi
