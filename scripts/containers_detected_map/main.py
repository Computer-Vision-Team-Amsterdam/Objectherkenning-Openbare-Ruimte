from scripts.containers_detected_map.containers_detected_map_generator import (
    ContainersDetectedMapGenerator,
)

date_of_the_ride = "2024-10-07"
ContainersDetectedMapGenerator(
    date_of_the_ride=date_of_the_ride, confidence_threshold=0.8
).create_and_store_map()
