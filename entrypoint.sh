#!/bin/bash

PYTHONPATH=. python -u objectherkenning_openbare_ruimte/data_delivery_pipeline/run_data_delivery_pipeline.py

bash objectherkenning_openbare_ruimte/preprocessing_service/preprocess_video_to_frames.sh

tail -F /dev/null
