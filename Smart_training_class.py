from pypdevs.DEVS import AtomicDEVS
import Statement as State
import Predict_LSTM_class as plstmc

class SmartTraining(AtomicDEVS):
    """
    Trainning class
    """
    def __init__(self, name, algo_name, filename):
        AtomicDEVS.__init__(self, name)
        self.name = name
        self.algo_name = algo_name
        self.filename = filename
        self.state = State.working
        self.isModelTrain = False
        self.new_entry = 0
        self.outport = self.addOutPort("outport")
        self.inport = self.addInPort("inport")

    def intTransition(self):
        # Internal Transition Function
        if self.isModelTrain:
            return State.waiting
        else:
            return State.working

    def outputFnc(self):
        # Output function to Smart_training_class
        if self.state == State.working:
            model, last_window, last_window_raw = plstmc.train_predict(self.filename)
            self.isModelTrain = True
            return {self.outport: [("isTrain", self.isModelTrain), ("model", model), ("last_window", last_window),
                                   ("last_window_raw", last_window_raw)]}
        else:
            return {self.outport: [State.waiting]}

    def timeAdvance(self):
        return 1.0
