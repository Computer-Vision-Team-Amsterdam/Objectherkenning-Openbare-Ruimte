from scripts.containers_detected_map.containers_detected_map_generator import (
    ContainersDetectedMapGenerator,
)

date_of_the_ride = "2024-08-05"
ContainersDetectedMapGenerator(date_of_the_ride=date_of_the_ride).create_and_store_map()
