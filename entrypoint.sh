#!/bin/bash

#bash objectherkenning_openbare_ruimte/preprocessing_service/preprocess_video_to_frames.sh &

#source /venv/bin/activate
PYTHONPATH=. python3 -u objectherkenning_openbare_ruimte/data_delivery_pipeline/run_data_delivery_pipeline.py &

tail -F /dev/null
