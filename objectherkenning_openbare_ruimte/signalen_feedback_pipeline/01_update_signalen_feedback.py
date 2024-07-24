# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()
import os 
import sys
sys.path.append('..')

from datetime import datetime
from post_processing_pipeline.helpers.utils_signalen import SignalHandler
from post_processing_pipeline.helpers.databricks_workspace import get_catalog_name
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import requests

requests.packages.urllib3.disable_warnings() 

#{'text': 'Dit betreft geen container.', 'user': 's.moring@amsterdam.nl', 'state': 'o', //'state_display': 'Afgehandeld', 'target_api': None, 'extra_properties': None, 'send_email': True, 'created_at': '2024-07-16T14:21:04.722426+02:00', 'email_override': None}

# {'text': 'Morgen word alles opgeruimd en afgehandeld.', 'user': 'p.herrebrugh@amsterdam.nl', 'state': 'o', 'state_display': 'Afgehandeld', 'target_api': None, 'extra_properties': None, 'send_email': True, 'created_at': '2024-07-18T12:54:06.659573+02:00', 'email_override': None}

if __name__ == "__main__":
   sparkSession = SparkSession.builder.appName("SignalFeedback").getOrCreate()
   signalHandler = SignalHandler(sparkSession)

   signalen_feedback_entries = []
   # for entry in signalHandler.get_pending_signalen_notifications().collect():
   #    signal = signalHandler.get_signal(sig_id=entry["signal_id"]))
   for entry in ["16864", "16865", "16866"]:
      signal_status = signalHandler.get_signal(sig_id=entry)["status"]
      if signal_status["state_display"] != "Gemeld":
         status, text, user, status_update_time = signal_status["state_display"], signal_status["text"], signal_status["user"], signal_status["created_at"]
   #       signalen_feedback_entries.append((status, comment, status_update_time))

   # signalen_feedback_df = spark.createDataFrame(signalen_feedback_entries, schema= signalHandler.get_signalen_feedback().schema) 
   # signalen_feedback_df.write.mode('append').saveAsTable(f'{signalHandler.catalog_name}.oor.bronze_signal_notifications_feedback')
   