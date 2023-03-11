import collections
import torch
import matplotlib.pyplot as plt

class FrameStack():
    
    def __init__(self, num):
        self.buffer = collections.deque(maxlen=num)
        self.buffer.append(torch.zeros((1, 84, 84),))
        self.buffer.append(torch.zeros((1, 84, 84),))
        self.buffer.append(torch.zeros((1, 84, 84),))
        self.buffer.append(torch.zeros((1, 84, 84),))
        self.max_size = num
        
    def put(self, state):
        self.buffer.append(state)
        
    def CanCat(self):
        if len(self.buffer) == self.max_size:
            return True
        else:
            return False
    def Get(self):
        result = torch.stack(list(self.buffer), dim = 1).squeeze(0)
        return result