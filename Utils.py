from datetime import datetime, timedelta
import csv
import random

class Timer:

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = datetime.now()

	def stop(self):
		end_dt = datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))


class Writer:

	def write_predict(self, date, value, file_name):
		with open(file_name, 'a') as temp_file:
			temp_writer = csv.writer(temp_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
			temp_writer.writerow([date, value])

	def set_file_to_write(self, file):
		return file

class CsvReader:

	def __init__(self):
		self.csvfilename = None

	def get_last_raw_date(self, csvfilename):
		row_index = 0
		with open(csvfilename, "r") as scraped:
			reader = csv.reader(scraped, delimiter=',')
			for row in reader:
				if row:  # avoid blank lines
					row_index += 1
					last_date = row[0]
		return last_date

	def get_last_raw_temperature(self, csvfilename):
		row_index = 0
		with open(csvfilename, "r") as scraped:
			reader = csv.reader(scraped, delimiter=',')
			for row in reader:
				if row:  # avoid blank lines
					row_index += 1
					last_temperature = row[1]
		float_temp = float(last_temperature)
		int_temp = int(float_temp)
		return int_temp

	def get_last_row(self, csvfilename):
		row_index = 0
		with open(csvfilename, "r") as scraped:
			reader = csv.reader(scraped, delimiter=',')
			for row in reader:
				if row:  # avoid blank lines
					row_index += 1
					columns = [str(row_index), row[0], row[1]]
		return columns


class DateConverter:

	def convert_to_datetime(self, string_date):
		date = datetime.strptime(string_date, "%Y-%m-%d")
		date = date.date()
		return date

	def get_next_day(self, actual_date):
		actual_date += timedelta(days=1)
		return actual_date


class DateGenerator:

	def generator(self, numberOfDate):
		reader = CsvReader()
		converter = DateConverter()
		writer = Writer()
		filename = "./data/weather_ajaccio.csv"
		last_row_date = reader.get_last_raw_date(filename)
		last_row_date_to_data = converter.convert_to_datetime(last_row_date)
		next_date = converter.get_next_day(last_row_date_to_data)
		last_row_temp = reader.get_last_raw_temperature(filename)
		for x in range(numberOfDate):
			if next_date.month == 1 or next_date.month == 2 or next_date.month == 3:
				random_plus_temp = random.randint(-15, 0)
			elif next_date.month == 4 or next_date.month == 5:
				random_plus_temp = random.randint(-2, 2)
			elif next_date.month == 6 or next_date.month == 7 or next_date.month == 8:
				random_plus_temp = random.randint(4, 15)
			elif next_date.month == 9 or next_date.month == 10:
				random_plus_temp = random.randint(-4, 4)
			else:
				random_plus_temp = random.randint(-8, 2)

			temp = last_row_temp + random_plus_temp
			writer.write_predict(next_date,temp,filename)
			next_date = converter.get_next_day(next_date)
