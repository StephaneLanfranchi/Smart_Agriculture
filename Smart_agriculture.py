from pypdevs.DEVS import CoupledDEVS
from pypdevs.simulator import Simulator
from Smart_predict_class import SmartPredict
from Smart_transition_class import SmartTransition
from Smart_training_class import SmartTraining


class System(CoupledDEVS):
    def __init__(self, name=None):
        CoupledDEVS.__init__(self, name)

        # Load data file
        train_file = './data/weather_ajaccio.csv'

        self.models = []
        # First model
        self.models.append(self.addSubModel(SmartTraining("_SmartTraining", train_file)))
        # Second model
        self.models.append(self.addSubModel(SmartPredict("_SmartPredict")))
        # Third model
        self.models.append(self.addSubModel(SmartTransition("_SmartTransition")))
        # Connect ouport to inport
        self.connectPorts(self.models[0].outport, self.models[1].inport)
        self.connectPorts(self.models[1].outport, self.models[2].inport)
        self.connectPorts(self.models[2].outport, self.models[0].inport)



print("Start Simulation")

smartSystem = System(name="SmartAgriculture")
sim = Simulator(smartSystem)
sim.setDSDEVS(True)

# Définie le temps d'éxécution
sim.setTerminationTime(8)
sim.setVerbose(None)
sim.simulate()

print("Simulation terminated with _SmartTraining in state %s" % smartSystem.models[0].state)
print("Simulation terminated with _SmartPredict in state %s" % smartSystem.models[1].state)
print("Simulation terminated with _SmartTransition in state %s" % smartSystem.models[2].state)
