import numpy as np
import cv2


class Patch:
    def __init__(self, center, direction, ref_im_idx):
        self.center = center
        self.direction = direction
        self.rp = ref_im_idx
    