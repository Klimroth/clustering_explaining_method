import torch
from torch.nn import Parameter, ParameterDict

class Paremeterized_Model():
    def __init__(self, device = 'cpu', dtype = torch.float64):
        self.device = device
        self.dtype = dtype
        self.parameters = ParameterDict()
        self._constrained = {}

    def add_param(self, name, init, constrained):
        self.parameters[name] = Parameter(init.clone().to(self.device).to(self.dtype))
        self._constrained[name] = constrained
        
    def constrained(self, name, indices = None):
        constrained = self._constrained[name](self.parameters[name])
        if indices is None:
            return constrained
        else:
            return constrained[indices]
        
    def to_device(self, device):
        self.parameters = self.parameters.to(device)
        self.device = device