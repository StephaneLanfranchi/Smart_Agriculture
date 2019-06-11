from pypdevs.DEVS import AtomicDEVS
import Statement as State
from Utils import CsvReader
import numpy as np


class SmartPredict(AtomicDEVS):
    """
    Predict class
    """
    def __init__(self, name):
        AtomicDEVS.__init__(self, name)
        self.name = name
        self.state = State.waiting
        self.csv_reader = CsvReader()
        self.file_name = "./data/weather_ajaccio.csv"
        self.is_model_train = False
        self.is_first_train = True
        self.prediction = 0
        self.dict_data = {}
        self.inport = self.addInPort("inport")
        self.outport = self.addOutPort("outport")

    def extTransition(self, inputs):
        for values in inputs[self.inport]:
            print(values)
            self.dict_data[values[0]] = values[1]
        self.is_model_train = self.dict_data.get("isTrain")

        if self.is_model_train:
            return State.working
        else:
            return State.waiting

    def intTransition(self):
        # Internal Transition Function
        if self.is_model_train:
            last_temp = self.csv_reader.get_last_raw_temperature(self.file_name)
            last_row = self.csv_reader.get_last_row(self.file_name)
            self.prediction = start_predict(self.dict_data.get("model"), self.dict_data.get("last_window"),
                                            last_temp, self.file_name, self.is_first_train)
            self.is_first_train = False
            self.prediction = float(self.prediction)
            self.prediction = int(self.prediction)
            return {self.outport: [("prediction", self.prediction), ("isTrain", self.is_model_train)]}
        else:
            return State.waiting

    def outputFnc(self):
        # Send the amount of messages sent on the output port
        if self.is_model_train:
            return {self.outport: [("prediction", self.prediction), ("isTrain", self.is_model_train)]}
        else:
            return {self.outport: [("isTrain", self.is_model_train), ("State", State.waiting)]}

    def timeAdvance(self):
        return 1.0


def start_predict(model, last_window, last_window_raw, filename, first_train):
    # Predict next time stamp
    if first_train:
        csv_reader = CsvReader()
        new_last_window_raw = csv_reader.get_last_raw_temperature(filename)
        next_timestamp = predict_next_timestamp(model, last_window)
        print "last_temprature if", new_last_window_raw
        next_timestamp_raw = (next_timestamp[0] + 1) * new_last_window_raw
        return format(next_timestamp_raw)
    else:
        next_timestamp = predict_next_timestamp(model, last_window)
        print "last_temprature else", last_window_raw
        next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw
        return format(next_timestamp_raw)


def predict_next_timestamp(model, history):
    """Predict the next time stamp given a sequence of history data"""

    prediction = model.predict(history)
    prediction = np.reshape(prediction, (prediction.size,))
    return prediction