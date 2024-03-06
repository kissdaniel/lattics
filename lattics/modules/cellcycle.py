"""LattiCS module containing cell cycle models.
"""

class CellCycleBase:
    def __init__(self):
        self._agent = None            

    def attach_agent(self, agent):
        self._agent = agent
        self._agent.initialize_status_flag('division_flag', False)

    def update(self, dt):
        pass

    def reset(self):
        pass

class FixedLengthIncrementalModel(CellCycleBase):
    def __init__(self):
        super().__init__()
        self._current_state = 0
        self._length = None

    @property
    def length(self):
        """The length (time duration) of the cell cycle."""
        return self._length
    
    @length.setter
    def length(self, value):
        self._length = value

    def update(self, dt):
        self._current_state += dt
        if self.length <= self._current_state:
            self._agent.set_status_flag('division_flag', True)
    
    def reset(self):
        self.current_state = 0