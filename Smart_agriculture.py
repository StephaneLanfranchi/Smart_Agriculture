from pypdevs.DEVS import CoupledDEVS
from pypdevs.simulator import Simulator
from Smart_predict_class import SmartPredict
from Smart_transition_class import SmartTransition
from Smart_training_class import SmartTraining


class System(CoupledDEVS):
    def __init__(self, name=None):
        CoupledDEVS.__init__(self, name)

        # Load command line arguments
        train_file = './data/weather_ajaccio.csv'

        algo_name = "LSTM"

        self.models = []
        # First model
        self.models.append(self.addSubModel(SmartTraining("_SmartTraining", algo_name, train_file)))
        # Second model
        self.models.append(self.addSubModel(SmartPredict("_SmartPredict", algo_name)))
        # Third model
        self.models.append(self.addSubModel(SmartTransition("_SmartTransition", algo_name)))
        # And connect them
        self.connectPorts(self.models[0].outport, self.models[1].inport)
        self.connectPorts(self.models[1].outport, self.models[2].inport)
        self.connectPorts(self.models[2].outport, self.models[0].inport)


print("Start Simulation")

smartSystem = System(name="SmartAgriculture")
sim = Simulator(smartSystem)
sim.setDSDEVS(True)
sim.setTerminationTime(10.0)
sim.setVerbose(None)
sim.simulate()

print("Simulation terminated with _SmartTraining in state %s" % smartSystem.models[0].state)
print("Simulation terminated with _SmartPredict in state %s" % smartSystem.models[1].state)
print("Simulation terminated with _SmartTransition in state %s" % smartSystem.models[2].state)
#print("Simulation terminated with prediction : %s" % smartSystem.models[2].list_prediction)
