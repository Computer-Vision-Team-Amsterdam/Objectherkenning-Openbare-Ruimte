from utils_signalen import SignalHandler

LAT = 52.368920318405074  #
LON = 4.9031888157681935  #
date_of_notification = "2024-05-24"
image_to_upload = "objectherkenning_openbare_ruimte/example.png"  #

notification_json = SignalHandler.fill_incident_details(incident_date=date_of_notification,
                                                        lon=LON,
                                                        lat=LAT,
                                                        )

signalHandler = SignalHandler()
signalHandler.post_signal_with_image_attachment(json_content=notification_json, filename=image_to_upload)
