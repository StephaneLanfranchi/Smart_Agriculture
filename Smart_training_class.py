from pypdevs.DEVS import AtomicDEVS
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import pandas as pd
import Statement as State
import json
import numpy as np


class SmartTraining(AtomicDEVS):
    """
    Trainning class
    """
    def __init__(self, name, filename):
        AtomicDEVS.__init__(self, name)
        self.name = name
        self.filename = filename
        self.state = State.waiting
        self.is_model_train = False
        self.model = Sequential()
        self.last_window = None
        self.last_window_raw = None
        self.outport = self.addOutPort("outport")
        self.inport = self.addInPort("IN")

    def intTransition(self):
        """
        Internal Transition Function
        """

        # Check if model is train
        if self.is_model_train:
            return State.waiting
        else:
            # Train model
            self.model, self.last_window, self.last_window_raw = train_predict(self.filename)
            self.is_model_train = True
            return [("isTrain", self.is_model_train), ("model", self.model), ("last_window", self.last_window),
                                   ("last_window_raw", self.last_window_raw)]

    def outputFnc(self):
        """
        Output function of Smart_training_class to Smart_predict_class
        """
        return {self.outport: [("isTrain", self.is_model_train), ("model", self.model), ("last_window", self.last_window),
                                   ("last_window_raw", self.last_window_raw)]}

    def timeAdvance(self):
        return 1.0


def train_predict(train_file):
    """ Train time series data"""

    # Load command line arguments
    parameter_file = 'training_config.json'

    # Load training parameters
    params = json.loads(open(parameter_file).read())

    # Load time series dataset, and split it into train and test
    x_train, y_train, x_test, y_test, x_test_raw, y_test_raw, \
    last_window_raw, last_window = load_timeseries(train_file, params)

    # Build RNN (LSTM) model
    lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
    model = rnn_lstm(lstm_layer, params)

    # Train RNN (LSTM) model with train set
    model.fit(
        x_train,
        y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        validation_split=params['validation_split'])

    # Check the model against test set
    predicted = predict_next_timestamp(model, x_test)
    predicted_raw = []
    for i in range(len(x_test_raw)):
        predicted_raw.append((predicted[i] + 1) * x_test_raw[i][0])

    return model, last_window, last_window_raw


def load_timeseries(filename, params):
    """Load time series dataset"""

    series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True)
    data = series.values

    adjusted_window = params['window_size'] + 1

    # Split data into windows
    raw = []
    for index in range(len(data) - adjusted_window):
        raw.append(data[index: index + adjusted_window])

    # Normalize data
    result = normalize_windows(raw)

    raw = np.array(raw)
    result = np.array(result)

    # Split the input dataset into train and test
    split_ratio = round(params['train_test_split'] * result.shape[0])
    train = result[:int(split_ratio), :]
    np.random.shuffle(train)

    # x_train and y_train, for training
    x_train = train[:, :-1]
    y_train = train[:, -1]

    # x_test and y_test, for testing
    x_test = result[int(split_ratio):, :-1]
    y_test = result[int(split_ratio):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    x_test_raw = raw[int(split_ratio):, :-1]
    y_test_raw = raw[int(split_ratio):, -1]

    # Last window, for next time stamp prediction
    last_raw = [data[-params['window_size']:]]
    last = normalize_windows(last_raw)
    last = np.array(last)
    last = np.reshape(last, (last.shape[0], last.shape[1], 1))

    return [x_train, y_train, x_test, y_test, x_test_raw, y_test_raw, last_raw, last]


def normalize_windows(window_data):
    """Normalize data"""

    normalized_data = []
    for window in window_data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return normalized_data


def rnn_lstm(layers, params):
    """Build RNN (LSTM) model"""

    model = Sequential()
    model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[1], return_sequences=True))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("tanh"))

    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return model


def predict_next_timestamp(model, history):
    """Predict the next time stamp given a sequence of history data"""

    prediction = model.predict(history)
    prediction = np.reshape(prediction, (prediction.size,))
    return prediction

