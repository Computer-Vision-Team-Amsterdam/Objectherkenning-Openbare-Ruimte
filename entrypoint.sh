#!/bin/bash

PYTHONPATH=. poetry run python -u objectherkenning_openbare_ruimte/on_edge/performance_monitoring/run_performance_monitoring.py &
PYTHONPATH=. poetry run python -u objectherkenning_openbare_ruimte/on_edge/detection_pipeline/run_detection_pipeline.py &
PYTHONPATH=. poetry run python -u objectherkenning_openbare_ruimte/on_edge/data_delivery_pipeline/run_data_delivery_pipeline.py &

tail -F /dev/null
