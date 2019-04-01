from pypdevs.DEVS import AtomicDEVS
import Statement as State
import Predict_LSTM_class as plstmc
from Utils import CsvReader

class SmartPredict(AtomicDEVS):
    """
    Predict class
    """
    def __init__(self, name, algo_name):
        AtomicDEVS.__init__(self, name)
        self.name = name
        self.algo_name = algo_name
        self.state = State.waiting
        self.csv_reader = CsvReader()
        self.file_name = "./data/weather_ajaccio.csv"
        self.is_model_train = False
        self.is_first_predict = True
        self.prediction = 0
        self.dict_data = {}
        self.inport = self.addInPort("inport")
        self.outport = self.addOutPort("outport")

    def extTransition(self, inputs):
        for values in inputs[self.inport]:
            self.dict_data[values[0]] = values[1]
        self.is_model_train = self.dict_data.get("isTrain")

        if self.is_model_train:
            return State.working
        else:
            return State.waiting

    def intTransition(self):
        # Internal Transition Function
        if self.is_model_train:
            return State.working
        else:
            return State.waiting

    def outputFnc(self):
        # Send the amount of messages sent on the output port
        if self.state == State.working:
            last_temp = self.csv_reader.get_last_raw_temperature(self.file_name)
            last_row = self.csv_reader.get_last_row(self.file_name)
            print last_row
            self.prediction = plstmc.start_predict(self.dict_data.get("model"), self.dict_data.get("last_window"),
                                                   last_temp, self.file_name, self.is_first_predict)
            self.prediction = float(self.prediction)
            self.prediction = int(self.prediction)
            self.is_first_predict = False
            return {self.outport: [self.prediction, self.is_model_train]}
        else:
            return {self.outport: [State.waiting]}

    def timeAdvance(self):
        return 1.0

