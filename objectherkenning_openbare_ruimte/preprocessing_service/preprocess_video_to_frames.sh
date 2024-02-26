#!/bin/bash

VIDEOS_FOLDER="../raw_data/"
OUTPUT_FOLDER="../raw_frames/"

while true; do
    for video_file in "$VIDEOS_FOLDER"/*.mp4; do
        if [ -f "$video_file" ]; then
            file_name=$(basename "$video_file")
            mkdir -p "$OUTPUT_FOLDER/$file_name"
            ffmpeg -i "$video_file" -vf "setpts=N/FRAME_RATE/TB" -fps_mode passthrough "$OUTPUT_FOLDER/$file_name/$file_name%04d.jpg"
            rm "$video_file"
        fi
    done
    sleep 5
done
