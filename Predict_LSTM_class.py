import json
import build_model
import data_helper
from Utils import CsvReader

def train_predict(train_file, model_state):
	# Train and predict time series data

	# Load command line arguments
	parameter_file = 'training_config.json'

	# Load training parameters
	params = json.loads(open(parameter_file).read())

	# Load time series dataset, and split it into train and test
	x_train, y_train, x_test, y_test, x_test_raw, y_test_raw,\
		last_window_raw, last_window = data_helper.load_timeseries(train_file, params)

	print "model_state train predict", model_state
	# Build RNN (LSTM) model
	lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
	model = build_model.rnn_lstm(lstm_layer, params)

	# Train RNN (LSTM) model with train set
	model.fit(
		x_train,
		y_train,
		batch_size=params['batch_size'],
		epochs=params['epochs'],
		validation_split=params['validation_split'])

	# Check the model against test set
	predicted = build_model.predict_next_timestamp(model, x_test)
	predicted_raw = []
	for i in range(len(x_test_raw)):
		predicted_raw.append((predicted[i] + 1) * x_test_raw[i][0])

	return model, last_window, last_window_raw

	# Plot graph: predicted VS actual
	# plt.subplot(111)
	# plt.plot(predicted_raw, label='Actual')
	# plt.plot(y_test_raw, label='Predicted')
	# plt.legend()
	# plt.show()


def start_predict(model, last_window, last_window_raw, filename, first_train):
	# Predict next time stamp
	if first_train:
		csv_reader = CsvReader()
		new_last_window_raw = csv_reader.get_last_raw_temperature(filename)
		next_timestamp = build_model.predict_next_timestamp(model, last_window)
		print "last_temprature if", new_last_window_raw
		next_timestamp_raw = (next_timestamp[0] + 1) * new_last_window_raw
		return format(next_timestamp_raw)
	else:
		next_timestamp = build_model.predict_next_timestamp(model, last_window)
		print "last_temprature else", last_window_raw
		next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw
		return format(next_timestamp_raw)


#if __name__ == '__main__':
	# python3 train_predict.py ./data/sales.csv ./training_config.json_
	# train_predict('./data/weather_ajaccio_multivariate.csv')