# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()

from datetime import datetime
from helpers.utils_signalen import SignalHandler
from helpers.databricks_workspace import get_catalog_name
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
import requests
requests.packages.urllib3.disable_warnings() 


def main():
   sparkSession = SparkSession.builder.appName("SignalHandler").getOrCreate()
   signalHandler = SignalHandler(sparkSession)

   
   gold_signal_notifications = sparkSession.table(f"{signalHandler.catalog_name}.oor.gold_signal_notifications")
        
   top_scores_df= signalHandler.get_top_pending_records(table_name="silver_objects_per_day", limit=10)

   if top_scores_df.count() == 0:
        print("04: No data found for creating notifications. Stopping execution.")
        return
        
   print(f"04: Loaded {top_scores_df.count()} rows with top 10 scores from {signalHandler.catalog_name}.oor.silver_objects_per_day.")

   date_of_notification = datetime.today().strftime('%Y-%m-%d')
   top_scores_df_with_date = top_scores_df.withColumn("notification_date", F.to_date(F.lit(date_of_notification)))

   successful_notifications = []
   unsuccessful_notifications = []

   # Convert to a list of row objects that are iterable
   for entry in top_scores_df_with_date.collect():

      LAT = float(entry["object_lat"])
      LON = float(entry["object_lon"])
      detection_id = entry["detection_id"]

      image_upload_path = signalHandler.get_image_upload_path(detection_id=detection_id)
      entry_dict = entry.asDict()
      try:
         # Check if image exists
         dbutils.fs.head(image_upload_path)
         notification_json = SignalHandler.fill_incident_details(incident_date=date_of_notification, lon=LON, lat=LAT,)
         id = signalHandler.post_signal_with_image_attachment(json_content=notification_json, filename=image_upload_path)
         print(f"Created signal {id} with image on {date_of_notification} with lat {LAT} and lon {LON}.\n\n" )
         entry_dict['signal_id'] = id
         updated_entry = Row(**entry_dict)         
         successful_notifications.append(updated_entry)
      except Exception as e:

         entry_dict.pop('notification_date', None)  #
         updated_failed_entry = Row(**entry_dict)
         if 'java.io.FileNotFoundException' in str(e):
            print(f"Image not found: {image_upload_path}. Skip creating notification...\n\n")
            unsuccessful_notifications.append(updated_failed_entry)

         else:
            print(f"An error occurred: {e}\n\n")
            unsuccessful_notifications.append(updated_failed_entry)
   
   if successful_notifications:
      successful_df = spark.createDataFrame(successful_notifications, schema=gold_signal_notifications.schema) 
      successful_df.write.mode('append').saveAsTable(f'{signalHandler.catalog_name}.oor.gold_signal_notifications')
      print(f"04: Appended {len(successful_notifications)} rows to gold_signal_notifications.")
   else:
      print("Appended 0 rows to gold_signal_notifications.")
   
   if unsuccessful_notifications:
      unsuccessful_df = spark.createDataFrame(unsuccessful_notifications, schema=top_scores_df.schema)
      print(f"{unsuccessful_df.count()} unsuccessful notifications.")
      unsuccessful_df.write.mode('append').saveAsTable(f'{signalHandler.catalog_name}.oor.silver_objects_per_day_quarantine')
      print(f"04: Appended {len(unsuccessful_notifications)} rows to silver_objects_per_day_quarantine.")
   
   signalHandler.update_status(table_name="silver_objects_per_day") 

if __name__ == "__main__":
   main()

   