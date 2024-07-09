# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()

import sys
from datetime import datetime
from helpers.utils_signalen import SignalHandler
from helpers.databricks_workspace import get_catalog_name
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# LAT = 52.38837746564135 
# LON = 4.914059828302194
# date_of_notification = "2024-06-06"

# image_to_upload = "/Volumes/dpcv_dev/default/landingzone/test-diana/0-D19M03Y2024-H16M17S04_frame_0100.jpg"


# notification_json = SignalHandler.fill_incident_details(incident_date=date_of_notification,
#                                                        lon=LON,
#                                                        lat=LAT,
#                                                        )

# signalHandler = SignalHandler()
# id = signalHandler.post_signal_with_image_attachment(json_content=notification_json, filename=image_to_upload)


# Check status of notification 
#notification = signalHandler.get_signal(sig_id=id)
#print(notification["status"])

if __name__ == "__main__":
   sparkSession = SparkSession.builder.appName("SignalHandler").getOrCreate()
   signalHandler = SignalHandler(sparkSession)
   top_scores_df = signalHandler.get_top_pending_records(table_name="silver_objects_per_day", limit=10)

   if top_scores_df.count() == 0:
        print("04: No data found for creating notifications. Stopping execution.")
        sys.exit()  
        
   print(f"04: Loaded {top_scores_df.count()} rows with top 10 scores from {signalHandler.catalog_name}.oor.silver_objects_per_day.")

   date_of_notification = datetime.today().strftime('%Y-%m-%d')
   top_scores_df = top_scores_df.withColumn("notification_date", F.to_date(F.lit(date_of_notification)))

   successful_notifications = []
   unsuccessful_notifications = []

   # Convert to a list of row objects that are iterable
   for entry in top_scores_df.collect():

      LAT = entry["object_lat"]
      LON = entry["object_lon"]
      detection_id = entry["detection_id"]

      image_upload_path = signalHandler.get_image_upload_path(detection_id=detection_id,date_of_notification=date_of_notification)

      try:
         # Check if image exists
         dbutils.fs.head(image_upload_path)
         notification_json = SignalHandler.fill_incident_details(incident_date=date_of_notification,
                                                       lon=LON,
                                                       lat=LAT,
                                                       )
         id = signalHandler.post_signal_with_image_attachment(json_content=notification_json, filename=image_upload_path)
         print(f"Created signal {id} with image on {date_of_notification} with lat {LAT} and lon {LON}." )
         successful_notifications.append(entry)
      except Exception as e:
         if 'java.io.FileNotFoundException' in str(e):
            print(f"Image not found: {image_upload_path}. Skip creating notification...")
            unsuccessful_notifications.append(entry)

         else:
            print(f"An error occurred: {e}")
            unsuccessful_notifications.append(entry)

   if successful_notifications:
      successful_df = spark.createDataFrame(successful_notifications, schema=top_scores_df.schema) 
      successful_df.write.mode('append').saveAsTable(f'{signalHandler.catalog_name}.oor.gold_signal_notifications')
      print(f"04: Appended {len(successful_notifications)} rows to gold_signal_notifications.")
   else:
      print("Appended 0 rows to gold_signal_notifications.")

   if unsuccessful_notifications:
      unsuccessful_df = spark.createDataFrame(unsuccessful_notifications, schema=top_scores_df.schema)
      print(f"{unsuccessful_df.count()} unsuccessful notifications.")

   signalHandler.update_status(table_name="silver_objects_per_day") 
   