# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()
import os 
import sys
sys.path.append('..')

from datetime import datetime
from pyspark.sql.types import StructType
from post_processing_pipeline.helpers.utils_signalen import SignalHandler
from post_processing_pipeline.helpers.databricks_workspace import get_catalog_name, set_job_process_time
from post_processing_pipeline.helpers.table_manager import TableManager
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import requests

requests.packages.urllib3.disable_warnings() 

#{'text': 'Dit betreft geen container.', 'user': 's.moring@amsterdam.nl', 'state': 'o', //'state_display': 'Afgehandeld', 'target_api': None, 'extra_properties': None, 'send_email': True, 'created_at': '2024-07-16T14:21:04.722426+02:00', 'email_override': None}

# {'text': 'Morgen word alles opgeruimd en afgehandeld.', 'user': 'p.herrebrugh@amsterdam.nl', 'state': 'o', 'state_display': 'Afgehandeld', 'target_api': None, 'extra_properties': None, 'send_email': True, 'created_at': '2024-07-18T12:54:06.659573+02:00', 'email_override': None}

   

if __name__ == "__main__":
   sparkSession = SparkSession.builder.appName("SignalFeedback").getOrCreate()
   signalHandler = SignalHandler(sparkSession)
   tableManager = TableManager(sparkSession, catalog=signalHandler.catalog_name)
   job_process_time = set_job_process_time()
   job_process_time = datetime.fromisoformat(job_process_time)

   signalen_feedback_entries = []
   ids_of_not_updated_status = []
   for entry in signalHandler.get_pending_signalen_notifications().collect():
      id = entry["id"]
      signal_status = signalHandler.get_signal(sig_id=entry["signal_id"])["status"]
      if signal_status["state_display"] != "Gemeld":
         print(f"status is {signal_status['state_display']}")
         print(f"id is {entry['id']}")
         status, text, user, status_update_time = signal_status["state_display"], signal_status["text"], signal_status["user"], signal_status["created_at"]
         # Convert the date string to a datetime object
         date_obj = datetime.fromisoformat(status_update_time)

         formatted_date= datetime(
                  year=date_obj.year,
                  month=date_obj.month,
                  day=date_obj.day,
                  hour=date_obj.hour,
                  minute=date_obj.minute,
                  second=date_obj.second
               )
         print(type(job_process_time))  
         print(f"job time is: {job_process_time}")             
         table_entry = (entry["signal_id"], status, text, user, formatted_date)
         signalen_feedback_entries.append(table_entry)
      else:
         ids_of_not_updated_status.append(id)
   
   signalen_feedback_df = signalHandler.get_signalen_feedback() 
   filtered_schema = StructType([field for field in signalen_feedback_df.schema.fields if field.name not in {'id', 'processed_at'}])

   print(signalen_feedback_entries)
   signalen_feedback_df = spark.createDataFrame(signalen_feedback_entries, filtered_schema) 
   display(signalen_feedback_df)
   signalen_feedback_df.write.mode('append').saveAsTable(f'{signalHandler.catalog_name}.oor.bronze_signal_notifications_feedback')
   print(f"01: Appended {len(signalen_feedback_entries)} rows to bronze_signal_notifications_feedback.")

   print(ids_of_not_updated_status)
   tableManager.update_status(table_name="gold_signal_notifications", job_process_time=job_process_time, exclude_ids=ids_of_not_updated_status)
   