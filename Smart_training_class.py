from pypdevs.DEVS import AtomicDEVS
import Statement as State
import Predict_LSTM_class as plstmc

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
        self.new_entry = 0
        self.outport = self.addOutPort("outport")
        self.inport = self.addInPort("IN")

    def intTransition(self):
        # Internal Transition Function
        if self.is_model_train:
            return State.waiting
        else:
            return State.working

    def outputFnc(self):
        # Output function to Smart_training_class
        if self.state == State.working:
            print "self.isModelTrain", self.is_model_train
            model, last_window, last_window_raw = plstmc.train_predict(self.filename, self.is_model_train)
            self.is_model_train = True
            return {self.outport: [("isTrain", self.is_model_train), ("model", model), ("last_window", last_window),
                                   ("last_window_raw", last_window_raw)]}
        else:
            return {self.outport: [("isTrain", self.is_model_train), ("State", State.waiting)]}

    def timeAdvance(self):
        return 1.0
