class StateError(Exception):
    def __init__(self):
        Exception.__init__(self, "Unknown statement")
