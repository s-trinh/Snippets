#!/bin/bash

nb_params=$#
if [ "$nb_params" -ge 1 ]; then
	quality=5
	if [ "$nb_params" -ge 2 ]; then
		quality=$2
	fi
	video_filename=$1
	output_filename=$video_filename".wmv"
	ffmpeg -i "$video_filename" -c:v msmpeg4v2 -q:v $quality -an "$output_filename"
	#~ ffmpeg -i $video_filename -c:v msmpeg4v2 -q:v $quality -c:a wmav2 $output_filename
fi
