'''DataLoader.py

Defines class that represents a loadable stream of data that can be loaded
from a folder of numpy arrays created from the `FileNode` class defined in
the `driverless` repository.
'''

from perc22a.data.utils import DataInstance

import numpy as np
import os
import cv2
import time

class DataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.instances = os.listdir(data_dir)
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        filename = f"instance-{index}.npz"
        filepath = os.path.join(self.data_dir, filename)

        di = DataInstance()
        di.load(filepath)
        return di