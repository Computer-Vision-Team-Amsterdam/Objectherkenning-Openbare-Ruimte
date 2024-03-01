#!/bin/bash

VIDEOS_FOLDER="/raw_videos"
OUTPUT_FOLDER="/raw_frames"

while true; do
  printf "%s\n" $VIDEOS_FOLDER
  for video_file in "$VIDEOS_FOLDER"/*.mp4; do
    printf "%s\n" $video_file
    if [ -f "$video_file" ]; then
        file_name=$(basename "$video_file")
        mkdir -p "$OUTPUT_FOLDER/$file_name"
        ffmpeg -i "$video_file" -c:v copy -bsf:v "$OUTPUT_FOLDER/$file_name/$file_name%04d.jpg"
        rm "$video_file"
    fi
  done
  sleep 5
done
