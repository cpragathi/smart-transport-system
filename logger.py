import csv
import os

FILE_NAME = "analysis_history.csv"

def save_result(driver_behavior, sentiment, confidence, timestamp):

    file_exists = os.path.isfile(FILE_NAME)

    with open(FILE_NAME, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(["Driver Behavior", "Sentiment", "Confidence", "Timestamp"])

        writer.writerow([driver_behavior, sentiment, confidence, timestamp])