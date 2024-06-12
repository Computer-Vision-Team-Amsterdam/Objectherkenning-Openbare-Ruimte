from utils_signalen import SignalHandler

LAT = 52.38837746564135 
LON = 4.914059828302194
# 52.38837746564135, 4.914059828302194
date_of_notification = "2024-06-06"

image_to_upload = "/Volumes/dpcv_dev/default/landingzone/test-diana/0-D19M03Y2024-H16M17S04_frame_0100.jpg"


notification_json = SignalHandler.fill_incident_details(incident_date=date_of_notification,
                                                       lon=LON,
                                                       lat=LAT,
                                                       )

signalHandler = SignalHandler()
# id = signalHandler.post_signal_with_image_attachment(json_content=notification_json, filename=image_to_upload)


# Check status of notification 
#notification = signalHandler.get_signal(sig_id=id)
#print(notification["status"])