import csv
from Utils import CsvReader, DateConverter
from datetime import datetime

with open("./data/weather_ajaccio.csv", "rb") as f:
    reader = csv.reader(f)
    i = reader.next()
    rest = [row for row in reader]

print i

reader = CsvReader()
last_raw = reader.get_last_raw_date("./data/weather_ajaccio.csv")
print "last_raw", last_raw

date_string = last_raw
dtConverter = DateConverter()

actual_date = dtConverter.convert_to_datetime(date_string)
print actual_date
next_date = dtConverter.get_next_day(actual_date)
print next_date