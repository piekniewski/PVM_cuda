# GPU PVM Implementation
# (C) 2018 Filip Piekniewski All Rights Reserved
# filip@piekniewski.info

import cv2
import numpy as np
import os
import zipfile
import sys
import pvmcuda_pkg.data as data
import pvmcuda_pkg.utils as utils


class SyntheticDataProvider(data.DataProvider):
    _attr = {}
    has_depth = False
    direction = 1
    step = 0
    def __init__(self, dx, dy, blocks_x=None, blocks_y=None, block_size=6, max_length=10000000000, use_segnet=False):
        self.dx = dx
        self.dy = dy
        self.blocks_x = blocks_x
        self.clocks_y = blocks_y
        self.block_size = block_size
        np.random.seed(100)
        self.advance()

    def get_next(self):
        return self.ret_val

    def get_label(self, last_valid=False):
        return np.zeros((self.dx, self.dy), dtype=np.float32)

    def __len__(self):
        return 0

    def advance(self):
        self.ret_val = 1.0 + np.zeros((self.dx, self.dy, 3), dtype=np.float32)
        middle = int(self.dx/2)
        self.ret_val[self.step*self.direction+middle-1:self.step*self.direction+middle+1, :] = 0
    #    self.ret_val[self.step*self.direction+middle-1::4, :] = 0
    #    self.ret_val[self.step*self.direction+middle-1::-4, :] = 0
        if self.step == middle:
            self.step = 0
            self.direction = np.random.randint(0, 2)
            if self.direction == 0:
                self.direction = -1
        self.step += 1

    def describe(self):
        return "Sytnthetic crap 1"

    def decode_label(self, label):
        return np.zeros((self.dx, self.dy, 3), dtype=np.uint8)

    def get_filename(self):
        return ""

    def get_pos(self):
        return self.pos

    def reset_pos(self):
        self.step = 0

    def get_classes(self):
        return []
