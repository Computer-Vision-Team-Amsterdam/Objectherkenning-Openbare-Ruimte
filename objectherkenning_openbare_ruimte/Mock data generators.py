# Databricks notebook source
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

class FrameDataGenerator:
    def __init__(self):
        # Set initial date to January 1, 2024, 09:00:00
        self.start_date = datetime(2024, 1, 1, 9, 0, 0)
        self.output_folder = "mock_frame_metadata" 
    
    def generate_mock_data(self, num_files=6, start_time=None):
        if start_time:
            self.start_date = start_time
        
        for _ in range(num_files):
            # Number of frames in five minutes
            frames_per_second = 2
            total_seconds = 5 * 60  # 5 minutes
            num_frames = frames_per_second * total_seconds

            # Set initial timestamp to the start date
            current_time = self.start_date
            
            data = []
            
            for _ in range(num_frames):
                timestamp = current_time.timestamp()
                frame_counter = random.randint(0, 10000)
                frame_timestamp = timestamp
                imu_state = random.choice([0, 1])
                imu_pitch = round(random.uniform(-90, 90), 4)
                imu_roll = round(random.uniform(-90, 90), 4)
                imu_heading = round(random.uniform(0, 360), 4)
                imu_gx = round(random.uniform(-1000, 1000), 1)
                imu_gy = round(random.uniform(-1000, 1000), 1)
                imu_gz = round(random.uniform(-1000, 1000), 1)
                gps_timestamp = timestamp
                gps_state = random.choice([0, 1])
                
                # 90% of the time use Amsterdam coordinates, 10% use null (None)
                if random.random() < 0.9:
                    gps_lat = round(52.3667 + random.uniform(-0.001, 0.001), 7)
                    gps_lon = round(4.8945 + random.uniform(-0.001, 0.001), 7)
                else:
                    gps_lat = None
                    gps_lon = None

                gps_time = current_time.strftime("%H:%M:%S")
                gps_date = current_time.strftime("%d/%m/%Y")
                gps_internal_timestamp = timestamp
                image_name = f"1-0-D{current_time.strftime('%dM%mY%Y-H%HM%MS%S')}_{frame_counter:04d}"
                model_name = "oor"
                model_version = "1"
                code_version = "0.1.0"
                
                data.append([
                    timestamp, frame_counter, frame_timestamp, imu_state, imu_pitch,
                    imu_roll, imu_heading, imu_gx, imu_gy, imu_gz, gps_timestamp,
                    gps_state, gps_lat, gps_lon, gps_time, gps_date, gps_internal_timestamp, image_name, model_name, model_version, code_version
                ])
                
                # Increment the current time by half a second (since 2 frames per second)
                current_time += timedelta(seconds=0.5)
            
            # Create DataFrame
            columns = [
                "timestamp", "pylon://0_frame_counter", "pylon://0_frame_timestamp", "imu_state",
                "imu_pitch", "imu_roll", "imu_heading", "imu_gx", "imu_gy", "imu_gz",
                "gps_timestamp", "gps_state", "gps_lat", "gps_lon", "gps_time",
                "gps_date", "gps_internal_timestamp", "image_name", "model_name", "model_version", "code_version"
            ]
            df = pd.DataFrame(data, columns=columns)
            
            # Ensure output folder exists
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            
            # Save DataFrame to CSV in the output folder
            file_name = os.path.join(self.output_folder, f"1-D{self.start_date.strftime('%dM%mY%Y-H%HM%MS%S')}.csv")
            df.to_csv(file_name, index=False)
            
            # Update start_date to next interval (5 minutes)
            self.start_date += timedelta(minutes=5)

# COMMAND ----------

class DetectionMetadataGenerator:
    def __init__(self, frame_data_folder, detection_output_folder):
        self.frame_data_folder = frame_data_folder
        self.detection_output_folder = detection_output_folder
    
    def generate_detection_metadata(self):
        # Ensure output folder exists
        if not os.path.exists(self.detection_output_folder):
            os.makedirs(self.detection_output_folder)
        
        # List all frame metadata files
        frame_files = [f for f in os.listdir(self.frame_data_folder) if f.endswith('.csv')]
        
        for file_name in frame_files:
            frame_file_path = os.path.join(self.frame_data_folder, file_name)
            df = pd.read_csv(frame_file_path)
            detection_data = []

            for _, row in df.iterrows():
                if random.random() < 0.1:  # 10% chance to create detection data for this frame
                    image_name = row['image_name']
                    num_detections = random.randint(1, 5)  # Random number of detections for this frame
                    for _ in range(num_detections):
                        object_class = random.randint(2, 4)
                        x_center = round(random.uniform(0, 1), 6)
                        y_center = round(random.uniform(0, 1), 6)
                        width = round(random.uniform(0, 1), 6)
                        height = round(random.uniform(0, 1), 6)
                        confidence = round(random.uniform(0.5, 1), 6)
                        tracking_id = random.randint(-1, 300)
                        
                        detection_data.append([
                            image_name, object_class, x_center, y_center, width, height, confidence, tracking_id
                        ])
            
            # Create DataFrame for detection metadata
            detection_columns = [
                "image_name", "object_class", "x_center", "y_center", "width", "height", "confidence", "tracking_id"
            ]
            detection_df = pd.DataFrame(detection_data, columns=detection_columns)
            
            # Save detection DataFrame to CSV in the output folder
            detection_file_path = os.path.join(self.detection_output_folder, file_name)
            detection_df.to_csv(detection_file_path, index=False)

# COMMAND ----------

if __name__ == "__main__":
    generator = FrameDataGenerator()
    
    # Generate 6 CSV files starting from the current self.start_date
    generator.generate_mock_data(num_files=6)
    
    # Generate another 6 CSV files for the next day at 10:00 AM
    next_day_start_time = datetime(2024, 1, 2, 10, 0, 0)
    generator.generate_mock_data(num_files=6, start_time=next_day_start_time)
    
    detection_metadata_generator = DetectionMetadataGenerator("mock_frame_metadata", "mock_detection_metadata")
    detection_metadata_generator.generate_detection_metadata()
    

# COMMAND ----------

df = pd.read_csv("mock_frame_metadata/1-D01M01Y2024-H09M00S00.csv")
display(df)
