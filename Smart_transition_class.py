from pypdevs.DEVS import AtomicDEVS
from Utils import Writer, CsvReader, DateConverter
import Statement as State


class SmartTransition(AtomicDEVS):
    """
    Transition class when train and predict end class
    """
    def __init__(self, name):
        AtomicDEVS.__init__(self, name)
        self.name = name
        self.state = State.waiting
        self.is_model_train = False
        self.writer = Writer()
        self.csv_reader = CsvReader()
        self.date_converter = DateConverter()
        self.prediction = 0
        self.dict_data = {}
        self.file_name = self.writer.set_file_to_write("./data/weather_ajaccio.csv")
        self.inport = self.addInPort("inport")
        self.outport = self.addOutPort("outport")

    def extTransition(self, inputs):
        # Internal retrieve data Function.
        if len(inputs[self.inport]) >= 2:
            for value in inputs[self.inport]:
                self.dict_data[value[0]] = value[1]
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
            last_date = self.csv_reader.get_last_raw_date("./data/weather_ajaccio.csv")
            last_date = self.date_converter.convert_to_datetime(last_date)
            next_date = self.date_converter.get_next_day(last_date)
            print "la prediction", self.dict_data.get("prediction")
            self.writer.write_predict(next_date, self.dict_data.get("prediction"), self.file_name)
            return {self.outport: [self.prediction]}
        else:
            return {self.outport: [State.waiting]}

    def timeAdvance(self):
        return 1.0

