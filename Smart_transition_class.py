from pypdevs.DEVS import AtomicDEVS
from Utils import Writer, CsvReader, DateConverter
import Statement as State


class SmartTransition(AtomicDEVS):
    """
    Transition class when train and predict end class
    """
    def __init__(self, name, algo_name):
        AtomicDEVS.__init__(self, name)
        self.name = name
        self.state = State.waiting
        self.algo_name = algo_name
        self.isModelTrain = False
        self.writer = Writer()
        self.csv_reader = CsvReader()
        self.date_converter = DateConverter()
        self.prediction = 0
        self.predict_data = []
        self.file_name = self.writer.set_file_to_write("./data/weather_ajaccio.csv")
        self.inport = self.addInPort("inport")
        self.outport = self.addOutPort("outport")

    def extTransition(self, inputs):
        # Internal retrieve data Function.
        if len(inputs[self.inport]) >= 2:
            for value in inputs[self.inport]:
                self.predict_data.append(value)
            self.prediction = self.predict_data[0]
            self.isModelTrain = self.predict_data[1]
            return State.working
        else:
            return State.waiting

    def intTransition(self):
        # Internal Transition Function
        if self.isModelTrain:
            return State.working
        else:
            return State.waiting

    def outputFnc(self):
        # Send the amount of messages sent on the output port
        if self.state == State.working:
            last_date = self.csv_reader.get_last_raw_date("./data/weather_ajaccio.csv")
            last_date = self.date_converter.convert_to_datetime(last_date)
            next_date = self.date_converter.get_next_day(last_date)
            self.writer.write_predict(next_date, self.prediction, self.file_name)
            return {self.outport: [self.prediction]}
        else:
            return {self.outport: [State.waiting]}

    def timeAdvance(self):
        return 1.0

