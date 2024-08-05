#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/usr/src/ultralytics/"
export PYTHONPATH="${PYTHONPATH}:/usr/src/objectherkenning_openbare_ruimte/"

python3 -u objectherkenning_openbare_ruimte/on_edge/performance_monitoring/run_performance_monitoring.py &
python3 -u objectherkenning_openbare_ruimte/on_edge/detection_pipeline/run_detection_pipeline.py &
python3 -u objectherkenning_openbare_ruimte/on_edge/data_delivery_pipeline/run_data_delivery_pipeline.py &

tail -F /dev/null
