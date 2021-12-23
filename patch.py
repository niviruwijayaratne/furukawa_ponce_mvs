import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

class Patch:
    def __init__(self, center, direction, ref_im_idx):
        self.center = center
        self.normal = direction
        self.rp = ref_im_idx
        self.tp  = []
        self.sp = None
        self.mu = 5
        self.grid = None
        self.d = -self.center.dot(self.normal)
    
    def construct_patch(self):
        d = -self.center.dot(self.normal)
        x_center, y_center = self.center[:-1]
        x, y = np.meshgrid(np.arange(x_center - self.mu//2, x_center + self.mu//2 + 1), \
                            np.arange(y_center - self.mu//2, y_center + self.mu//2 + 1))
        z = np.zeros_like(x)
        for y_idx in range(y.shape[0]):
            for x_idx in range(x.shape[0]):
                z[y_idx, x_idx] = (-self.normal[0]*x[y_idx, x_idx] - self.normal[1]*y[y_idx, x_idx] - d)/self.normal[2]

        self.grid = np.dstack([x, y, z])
        

