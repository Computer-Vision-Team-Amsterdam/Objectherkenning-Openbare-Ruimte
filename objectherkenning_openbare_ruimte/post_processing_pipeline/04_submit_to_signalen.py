# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()

from datetime import datetime
from helpers.utils_signalen import SignalHandler
from helpers.databricks_workspace import get_catalog_name
from pyspark.sql import SparkSession

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
   print(f"04: Loaded {top_scores_df.count()} rows with top 10 scores from {signalHandler.catalog_name}.oor.silver_objects_per_day.")

   date_of_notification = datetime.today().strftime('%Y-%m-%d')

   # Convert to a list of row objects that are iterable
   top_scores_list = top_scores_df.collect()
   for entry in top_scores_df:
      LAT = entry["object_lat"]
      LON = entry["object_lon"]

      detection_id = entry["detection_id"]
      # TODO get image_name based on detection_id in silver_detection_metadata like below
      query = f"""
               SELECT silver_detection_metadata.image_name
               FROM silver_detection_metadata
               WHERE silver_detection_metadata.id = {detection_id}
               """

      image_base_name = spark.sql(query)

      image_to_upload = f'/Volumes/{signalHandler.catalog_name}/default/landingzone/Luna/images/{date_of_notification}/{image_base_name}'

      notification_json = SignalHandler.fill_incident_details(incident_date=date_of_notification,
                                                       lon=LON,
                                                       lat=LAT,
                                                       )
      id = signalHandler.post_signal_with_image_attachment(json_content=notification_json, filename=image_to_upload)
      

   # TODO append date of notification column
   #top_scores_df.withColumn("notification_date", date_of_notification)

   top_scores_df.write.mode('append').saveAsTable(f'{signalHandler.catalog_name}.oor.gold_signal_notifications')
   print(f"04: Appended {top_scores_df.count()} rows to gold_signal_notifications.")

   signalHandler.update_status(table_name="silver_objects_per_day") 
   