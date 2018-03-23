import random
import numpy as np


class ExperienceBuffer():
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    
    def add(self, experience):
        total_size = len(self.buffer) + len(experience) 
        if total_size >= self.buffer_size:
            self.buffer[0:total_size - self.buffer_size] = []
        self.buffer.extend(experience)


    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])