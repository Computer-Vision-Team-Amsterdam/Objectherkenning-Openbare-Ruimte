#!/bin/bash

VIDEOS_FOLDER="/raw_videos"
OUTPUT_FOLDER="/raw_frames"
LOGS_FOLDER="/cvt_logs"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOGS_FOLDER/preprocess_$TIMESTAMP.txt"
while true; do
  printf "%s\n" $VIDEOS_FOLDER
  for video_file in "$VIDEOS_FOLDER"/*.mp4; do
    printf "%s\n" "$video_file" >> "$LOG_FILE"
    printf "%s\n" "$video_file"
    if [ -f "$video_file" ]; then
        file_name=$(basename "$video_file")
        file_name=${file_name%.*}
        mkdir -p "$OUTPUT_FOLDER/$file_name"
        ffmpeg -i "$video_file" -vf "settb=AVTB,setpts=N/TB,lenscorrection=cx=0.509:cy=0.488:k1=-0.241:k2=0.106:i=bilinear" -vsync passthrough -q:v 1 "$OUTPUT_FOLDER/$file_name/$file_name"_frame_%04d.jpg
        RESULT=$?
        if [ $RESULT -eq 0 ]; then
          rm "$video_file"
          printf "ffmpeg completed, file removed: %s\n" "$video_file"
          printf "ffmpeg completed, file removed: %s\n" "$video_file" >> "$LOG_FILE"
        else
          printf "Error running ffmpeg on: %s\n" "$video_file"
          printf "Error running ffmpeg on: %s\n" "$video_file" >> "$LOG_FILE"
        fi
    fi
  done
  sleep 5
done
