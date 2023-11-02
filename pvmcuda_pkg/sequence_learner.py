# GPU PVM Implementation
# (C) 2017 Filip Piekniewski All Rights Reserved
# filip@piekniewski.info
import pvmcuda_pkg.gpu_routines as gpu_routines
import pycuda.gpuarray as gpuarray
import numpy as np
import sys
import os
import cv2
import pycuda.autoinit
import argparse
import json
import time
import pvmcuda_pkg.disp as disp
import pycuda.driver as cuda
import pickle
import zipfile
import random
import datetime
import pvmcuda_pkg.utils as utils

np.set_printoptions(threshold=np.inf, linewidth=200, precision=6)

# Specs:
# units with their sizes
# set of block mappings for input in the form [id] -> [id]
# where id=-1 is a special block for input array
# set of block mappings for context in the form [id] -> [id]
# sequence history length - how many frames to use for training
# sequence predicted length - how many frames to predict
# sequence interval - time shift from input to prediction
#
# %t+3% [ t+3 ][ t+2 ][ context t+2 ] -> Activation(t+3)
# %t+2% [ t+2 ][ t+1 ][ context t+1 ] -> Activation(t+2)
# %t+1% [ t+1 ][ t+0 ][ context t+0 ] -> Activation(t+1)
# %t+0% [ t-0 ][ t-1 ][ context t-1 ] -> Activation(t+0)
#
# Asscociation occurs between Activation(t) and Activation(t+seq_interval)
# For learning [ t-0 ][ t-1 ] -> [ t+2 ][ t+1 ] seq_interval=2, seq_len=2

# The needs to be reimplemented
def get_surround(xy, x_size=10, y_size=10, radius=1, exclude_self=True):
    """
    Returns the indices of elements on the grid that are within square radius
    of the given xy

      radius = 1:

        0 1 0
        1 1 1
        0 1 0

      radius = 1.5:

        1 1 1
        1 1 1
        1 1 1

      radius = 2

        0 0 1 0 0
        0 1 1 1 0
        1 1 1 1 1
        0 1 1 1 0
        0 0 1 0 0

    Setting exclude_self to True removes the center unit
    """
    laterals = []
    for dx in range(-int(radius), int(radius)+1, 1):
        for dy in range(-int(radius), int(radius)+1, 1):
            if dx**2 + dy**2 > radius**2:
                continue
            if (xy[0]+dx >= 0) and (xy[0]+dx < x_size) and (xy[1]+dy >= 0) and (xy[1]+dy < y_size):
                if not (exclude_self and dx == 0 and dy == 0):
                    laterals.append((xy[0]+dx, xy[1]+dy))
    return laterals

# This needs to be reimplemented
def get_fan_in(xy=(0, 0), dim_x_l=10, dim_y_l=10, dim_x_u=9, dim_y_u=9, block_x=2, block_y=2, radius=2.0):
    """
    Selects a block_x x block_y subsquare in the underlying layers lying directly below the unit in the
    upper layer. Selects units within radius in that block.

      e.g. block_x=2, block_y=2 radius=2

        1 1
        1 1

      e.g. block_x=3, block_y=3 radius=2

        1 1 1
        1 1 1
        1 1 1

      e.g. block_x=3, block_y=3 radius=1

        0 1 0
        1 1 1
        0 1 0

    """
    x = xy[0]
    y = xy[1]
    if dim_x_u > 1:
        factor_x = ((dim_x_l-1)-(block_x-1))/(1.0*(dim_x_u-1))
    else:
        factor_x = ((dim_x_l-1)-(block_x))/2.0
    if dim_y_u > 1:
        factor_y = ((dim_y_l-1)-(block_y-1))/(1.0*(dim_y_u-1))
    else:
        factor_y = ((dim_y_l-1)-(block_y))/2.0
    results = []
    if dim_x_u > 1 and dim_y_u > 1:
        for xx in range(block_x):
            for yy in range(block_y):
                if (xx-(block_x-1)*0.5)**2 + (yy-(block_y-1)*0.5)**2 > radius**2:
                    continue
                results.append((int((factor_x*(x))+xx), int((factor_y*(y))+yy)))
        return results
    elif dim_x_u == 1 and dim_y_u > 1:
        for xx in range(block_x):
            for yy in range(block_y):
                if (xx-(block_x-1)*0.5)**2 + (yy-(block_y-1)*0.5)**2 > radius**2:
                    continue
                results.append((int((dim_x_l-block_x)/2.0+xx), int((factor_y*(y)+yy))))
        return results
    elif dim_x_u > 1 and dim_y_u == 1:
        for xx in range(block_x):
            for yy in range(block_y):
                if (xx-(block_x-1)*0.5)**2 + (yy-(block_y-1)*0.5)**2 > radius**2:
                    continue
                results.append((int((factor_x*(x)+xx)), int((dim_y_l-block_y)/2.0+yy)))
        return results
    elif dim_x_u == 1 and dim_y_u == 1:
        for xx in range(block_x):
            for yy in range(block_y):
                if (xx-(block_x-1)*0.5)**2 + (yy-(block_y-1)*0.5)**2 > radius**2:
                    continue
                results.append((int((dim_x_l-block_x)/2.0+xx), int((dim_y_l-block_y)/2.0+yy)))
        return results


def append_unique(L, el):
    if el not in L:
        L.append(el)
    return L


def extend_unique(L, L1):
    for el in L1:
        if el not in L:
            L.append(el)
    return L


class PVM_object():
    def __init__(self, specs, name="noname"):
        """
        This for now assumes a 3 layer perceptron as the PVM unit.
        :param specs:
        """
        self.device = pycuda.driver.Device(0).name()
        if specs is not None:
            self.specs = specs
            self.name = name
            self.uniq_id = "%08x" % random.getrandbits(32)
            self.time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            self.poly = utils.check_if_enabled("polynomial", self.specs)
            self.total_weights = 0
            self.total_input_mem = 0
            self.total_repr_mem = 0
            self.total_primary_projections = 0
            self.total_context_projections = 0
            self.base_t = 0
            self.current_frame = 0
            self.sequence_length = 3
            self.sequence_buff_length = 4
            self.step = 0
            self.generate_graph()
            self.generate_memory()
            self.generate_memory_ptrs()
            self.generate_flow_ptrs()
            self.create_mem_gpu()

    def generate_graph(self):
        specs = self.specs
        self.sequence_interval = 2
        self.graph = []
        self.layer_ptrs = []
        self.learning_rate = 0.001
        self.momentum = 0.9
        total_weights = 0
        total_units = 0
        total_input_mem = 0
        total_repr_mem = 0
        total_primary_projections = 0
        total_context_projections = 0
        i = 0
        self.input_channels = 3
        if "input_channels" in list(specs.keys()):
            self.input_channels = int(specs['input_channels'])
        for layer in range(len(specs['layer_shapes'])):
            if layer == 0:
                self.layer_ptrs.append(0)
            else:
                self.layer_ptrs.append(self.layer_ptrs[-1]+int(specs['layer_shapes'][layer-1]) * int(specs['layer_shapes'][layer-1]))
            for x in range(int(specs['layer_shapes'][layer])):
                for y in range(int(specs['layer_shapes'][layer])):
                    block = {}
                    if layer == 0:
                        block['base_input_offset'] = self.sequence_interval * int(specs['input_block_size']) * int(specs['input_block_size']) * self.input_channels
                        block['output_dim'] = block['base_input_offset']
                        block['base_input_size'] = int(specs['input_block_size']) * int(specs['input_block_size']) * self.input_channels
                        block['xs'] = [x]
                        block['ys'] = [y]
                    else:
                        block['base_input_offset'] = 0
                        block['base_input_size'] = 0
                        block['output_dim'] = 0
                        block['xs'] = []
                        block['ys'] = []

                    total_units += 1
                    block['id'] = i
                    i += 1
                    block['type'] = "hidden"
                    block['size'] = int(specs['hidden_block_size']) * int(specs['hidden_block_size'])
                    block['input_dim'] = 0
                    block['layer'] = layer
                    block['grid_pos'] = (x, y)
                    block['image_pos'] = None
                    block['primary_destinations'] = []
                    block['context_destinations'] = []
                    block['primary_sources'] = []
                    block['context_sources'] = []
                    block['context_dim'] = 0
                    block['base_context_offset'] = block['size'] + 1
                    block['sources'] = []
                    block['mem'] = None
                    block['gpu_mem'] = None
                    block['running_input_ptr'] = block['base_input_offset']
                    block['ptr'] = None
                    block['gpu_ptr'] = None
                    block['primary_dst_ptr'] = []
                    block['context_dst_ptr'] = []
                    self.graph.append(block)
        for block in self.graph:
            if block['type'] == 'hidden':
                # Connect surround
                sur = get_surround(xy=block['grid_pos'],
                                   x_size=int(specs['layer_shapes'][block['layer']]),
                                   y_size=int(specs['layer_shapes'][block['layer']]),
                                   radius=int(specs['lateral_radius']),
                                   exclude_self=specs["context_exclude_self"] == '1')

                for sublock in self.graph:
                    if sublock['layer'] == block['layer'] and sublock['grid_pos'] in sur:
                        sublock['context_destinations'].append(block['id'])
                        block['context_sources'].append(sublock['id'])
                        total_context_projections += 1
                # Connect forward path
                if block['layer'] > 0:
                    fan_in = get_fan_in(xy=block['grid_pos'],
                                        dim_x_l=int(specs['layer_shapes'][block['layer'] - 1]),
                                        dim_y_l=int(specs['layer_shapes'][block['layer'] - 1]),
                                        dim_x_u=int(specs['layer_shapes'][block['layer']]),
                                        dim_y_u=int(specs['layer_shapes'][block['layer']]),
                                        block_x=int(specs['fan_in_square_size']),
                                        block_y=int(specs['fan_in_square_size']),
                                        radius=float(specs['fan_in_radius']))
                    for sublock in self.graph:
                        if sublock['layer'] == block['layer'] - 1 and sublock['grid_pos'] in fan_in:
                            sublock['primary_destinations'].append(block['id'])
                            block['primary_sources'].append(sublock['id'])
                            block['base_input_size'] += sublock['size']
                            block['xs'] = extend_unique(block['xs'], sublock['xs'])
                            block['ys'] = extend_unique(block['ys'], sublock['ys'])
                            total_primary_projections += 1

        # Reverse connect feedback
        for block in self.graph:
            if block['type'] == 'hidden':
                for dest in block['primary_destinations']:
                    self.graph[dest]['context_destinations'].append(block['id'])
                    block["context_sources"].append(dest)
                    total_context_projections += 1
        if utils.check_if_enabled('send_context_two_layers_back', self.specs):
            for block in self.graph:
                if block['type'] == 'hidden':
                    for dest in block['primary_destinations']:
                        for dest1 in self.graph[dest]['primary_destinations']:
                            self.graph[dest1]['context_destinations'].append(block['id'])
                            block["context_sources"].append(dest1)
                            total_context_projections += 1
        if utils.check_if_enabled("last_layer_context_to_all", self.specs):
            for block in self.graph:
                if block['layer'] == len(specs['layer_shapes']) - 1:  # Last layer
                    for ublock in self.graph:
                        if ublock['id'] not in block['context_destinations'] and ublock['type'] == 'hidden':
                            block['context_destinations'].append(ublock['id'])
                            ublock["context_sources"].append(block['id'])
                            total_context_projections += 1
        for block in self.graph:
            if block['type'] == 'hidden':
                block['input_dim'] = block['base_input_offset']
                for sb in block['primary_sources']:
                    block['input_dim'] += self.sequence_interval * self.graph[sb]['size']
                    block['output_dim'] += self.sequence_interval * self.graph[sb]['size']
                for sb in block['context_sources']:
                    block['input_dim'] += self.graph[sb]['size']
                    block['context_dim'] += self.graph[sb]['size']
        for block in self.graph:
            if block['type'] == 'hidden':
                if utils.check_if_enabled("feed_context_in_complex_layer", self.specs):
                    total_weights += (block['input_dim'] + 1) * block['size'] + (block['size'] + 1 + block["context_dim"]) * block['output_dim']
                    total_repr_mem += block['size'] + 1 + block['context_dim']
                else:
                    total_weights += (block['input_dim'] + 1) * block['size'] + (block['size'] + 1) * block['output_dim']
                    total_repr_mem += block['size'] + 1
                total_input_mem += block['input_dim'] + 1
        self.total_units = total_units
        self.total_weights = total_weights
        self.total_input_mem = total_input_mem
        self.total_repr_mem = total_repr_mem
        self.total_context_projections = total_context_projections
        self.total_primary_projections = total_primary_projections
        print("Generated connectivity with %d units and %d weights" % (total_units, total_weights))
        print("Total input mem %d, total representation mem %d" % (total_input_mem, total_repr_mem))
        print("Total primary projections %d, total context projections %d" % (total_primary_projections, total_context_projections))

    def generate_memory(self):
        self.weight_memory_main = 0.03*(np.random.rand(self.total_weights).astype(np.float32) - 0.5)
        self.dweight_memory_0 = np.zeros(shape=(self.total_weights,), dtype=np.float32)
        self.dweight_memory_1 = np.zeros(shape=(self.total_weights,), dtype=np.float32)
        self.list_dweight_memory = [self.dweight_memory_0, self.dweight_memory_1]

        self.weight_memory_cache0 = np.zeros(shape=(self.total_weights,), dtype=np.float32)
        self.weight_memory_cache1 = np.zeros(shape=(self.total_weights,), dtype=np.float32)
        self.buffer_index = 0
        self.input_mem_activation = []
        self.input_mem_delta = []
        self.input_mem_error = []
        # Same structure as input
        self.output_mem_activation = []
        self.output_mem_delta = []
        self.output_mem_error = []
        self.repre_mem_activation = []
        self.repre_mem_delta = []
        self.repre_mem_error = []
        self.beta_input = np.ones(shape=(self.total_input_mem,), dtype=np.float32)
        self.beta_repre = np.ones(shape=(self.total_repr_mem,), dtype=np.float32)
        for i in range(self.sequence_length):
            self.input_mem_activation.append(np.zeros(shape=(self.total_input_mem,), dtype=np.float32) + 1)
            self.input_mem_error.append(np.zeros(shape=(self.total_input_mem,), dtype=np.float32))
            self.input_mem_delta.append(np.zeros(shape=(self.total_input_mem,), dtype=np.float32))
            # Add the output memory
            self.output_mem_activation.append(np.zeros(shape=(self.total_input_mem,), dtype=np.float32) + 1)
            self.output_mem_error.append(np.zeros(shape=(self.total_input_mem,), dtype=np.float32))
            self.output_mem_delta.append(np.zeros(shape=(self.total_input_mem,), dtype=np.float32))

        for i in range(self.sequence_length):
            self.repre_mem_activation.append(np.zeros(shape=(self.total_repr_mem,), dtype=np.float32) + 1)
            self.repre_mem_error.append(np.zeros(shape=(self.total_repr_mem,), dtype=np.float32))
            self.repre_mem_delta.append(np.zeros(shape=(self.total_repr_mem,), dtype=np.float32))

    def generate_memory_ptrs(self):
        self.weight_ptr0 = np.zeros(shape=(self.total_units), dtype=np.int32)
        self.weight_ptr1 = np.zeros(shape=(self.total_units), dtype=np.int32)
        self.shape0_L0 = np.zeros(shape=(self.total_units), dtype=np.int32)
        self.shape1_L0 = np.zeros(shape=(self.total_units), dtype=np.int32)
        self.shape0_L1 = np.zeros(shape=(self.total_units), dtype=np.int32)
        self.shape1_L1 = np.zeros(shape=(self.total_units), dtype=np.int32)
        self.input_ptr = np.zeros(shape=(self.total_units), dtype=np.int32)
        self.repr_ptr = np.zeros(shape=(self.total_units), dtype=np.int32)
        self.learning_rate_arr = np.zeros(shape=(self.total_units), dtype=np.float32)
        self.momentum_arr = np.zeros(shape=(self.total_units), dtype=np.float32)
        curr_w0_ptr = 0
        curr_w1_ptr = 0
        curr_i_ptr = 0
        curr_r_ptr = 0
        total_threads_k2_L0 = 0
        total_threads_k2_L1 = 0
        i = 0
        for block in self.graph:
            if block['type'] == 'hidden':
                self.weight_ptr0[i] = curr_w0_ptr
                block["w0_ptr"] = curr_w0_ptr
                curr_w0_ptr += (block['input_dim'] + 1) * block['size']
                total_threads_k2_L0 += (block['input_dim'] + 1)
                self.shape0_L0[block['id']] = block['size']
                self.shape1_L0[block['id']] = (block['input_dim'] + 1)

                self.weight_ptr1[i] = curr_w1_ptr
                self.shape0_L1[block['id']] = block['output_dim']
                if self.specs["feed_context_in_complex_layer"] == "0":
                    total_threads_k2_L1 += (block['size'] + 1)
                    self.shape1_L1[block['id']] = block['size'] + 1
                else:
                    total_threads_k2_L1 += (block['size'] + 1 + block['context_dim'])
                    self.shape1_L1[block['id']] = block['size'] + 1 + block['context_dim']


                block["w1_ptr"] = curr_w1_ptr
                if self.specs["feed_context_in_complex_layer"] == "0":
                    curr_w1_ptr += (block['size'] + 1) * block['output_dim']
                else:
                    curr_w1_ptr += (block['size'] + 1 + block['context_dim']) * block['output_dim']
                self.input_ptr[i] = curr_i_ptr
                block["i_ptr"] = curr_i_ptr
                curr_i_ptr += block['input_dim'] + 1
                self.repr_ptr[i] = curr_r_ptr
                block["r_ptr"] = curr_r_ptr
                if self.specs["feed_context_in_complex_layer"] == "0":
                    curr_r_ptr += (block['size'] + 1)
                else:
                    curr_r_ptr += (block['size'] + 1 + block['context_dim'])
                i += 1
        self.weight_ptr1[:] += curr_w0_ptr
        self.obj_id_k2_L0 = np.zeros(shape=(total_threads_k2_L0,), dtype=np.int32)
        self.row_id_k2_L0 = np.zeros(shape=(total_threads_k2_L0,), dtype=np.int32)
        self.obj_id_k2_L1 = np.zeros(shape=(total_threads_k2_L1,), dtype=np.int32)
        self.row_id_k2_L1 = np.zeros(shape=(total_threads_k2_L1,), dtype=np.int32)
        self.col_id_k2_L1 = np.zeros(shape=(total_threads_k2_L1,), dtype=np.int32)
        thread_id_L0 = 0
        thread_id_L1 = 0

        for block in self.graph:
            if block['type'] == 'hidden':
                block["w1_ptr"] += curr_w0_ptr
                for i in range(block['input_dim']+1):
                    self.obj_id_k2_L0[thread_id_L0] = block['id']
                    self.row_id_k2_L0[thread_id_L0] = i
                    thread_id_L0 += 1
                if self.specs["feed_context_in_complex_layer"] == "0":
                    for i in range(block['size'] + 1):
                        self.obj_id_k2_L1[thread_id_L1] = block['id']
                        self.row_id_k2_L1[thread_id_L1] = i
                        thread_id_L1 += 1
                else:
                    for i in range(block['size'] + 1 + block['context_dim']):
                        self.obj_id_k2_L1[thread_id_L1] = block['id']
                        self.row_id_k2_L1[thread_id_L1] = i
                        thread_id_L1 += 1

        self.cuda_block_size = 128
        self.cuda_grid_L0 = total_threads_k2_L0 // self.cuda_block_size + 1
        self.cuda_total_threads_k2_L0 = total_threads_k2_L0
        self.cuda_grid_L1 = total_threads_k2_L1 // self.cuda_block_size + 1
        self.cuda_total_threads_k2_L1 = total_threads_k2_L1

    def generate_flow_ptrs(self):
        """
        Flow pointers tell which block of memory to copy where.
        :return:
        """
        # For now this only supports sequence of length 2
        self.flow_ptr_from = (np.zeros(shape=(self.total_primary_projections+self.total_context_projections,), dtype=np.int32))
        self.flow_ptr_to = (np.zeros(shape=(self.total_primary_projections+self.total_context_projections,), dtype=np.int32))
        self.flow_ptr_repr_to = (np.zeros(shape=(self.total_context_projections,), dtype=np.int32))
        self.flow_ptr_repr_from = (np.zeros(shape=(self.total_context_projections,), dtype=np.int32))
        self.flow_ptr_repr_size = (np.zeros(shape=(self.total_context_projections,), dtype=np.int32))
        self.flow_ptr_size = (np.zeros(shape=(self.total_primary_projections+self.total_context_projections,), dtype=np.int32))
        self.flow_ptr_input_shift_from = (np.zeros(shape=(self.total_units,), dtype=np.int32))
        self.flow_ptr_input_shift_to = (np.zeros(shape=(self.total_units,), dtype=np.int32))
        self.flow_ptr_input_shift_size = (np.zeros(shape=(self.total_units,), dtype=np.int32))
        self.flow_ptr_input_frame = (np.zeros(shape=(int(self.specs['layer_shapes'][0])*int(self.specs['layer_shapes'][0]),), dtype=np.int32))
        self.flow_ptr_input_frame_size = (np.zeros(shape=(int(self.specs['layer_shapes'][0])*int(self.specs['layer_shapes'][0]),), dtype=np.int32))
        i = 0
        for block in self.graph:
            if block['type'] == 'hidden':
                primary_src_shift = 0
                for primary_source in block['primary_sources']:
                    self.flow_ptr_size[i] = block['size']
                    self.flow_ptr_from[i] = self.repr_ptr[primary_source]
                    self.flow_ptr_to[i] = block["i_ptr"] + block['running_input_ptr']
                    primary_src_shift += self.graph[primary_source]['size']
                    block['running_input_ptr'] += self.graph[primary_source]['size']
                    i += 1
                block['running_input_ptr'] += primary_src_shift * (self.sequence_interval-1)
                for context_source in block['context_sources']:
                    self.flow_ptr_size[i] = block['size']
                    self.flow_ptr_from[i] = self.repr_ptr[context_source]
                    self.flow_ptr_to[i] = block["i_ptr"] + block['running_input_ptr']
                    block['running_input_ptr'] += self.graph[context_source]['size']
                    i += 1
                self.flow_ptr_input_shift_from[block['id']] = block['i_ptr']
                self.flow_ptr_input_shift_to[block['id']] = block['i_ptr'] + block['base_input_size']
                self.flow_ptr_input_shift_size[block['id']] = block['base_input_size'] * (self.sequence_interval - 1)
        i = 0
        for block in self.graph:
            if block['type'] == 'hidden':
                running_ptr = 0
                for context_source in block['context_sources']:
                    self.flow_ptr_repr_size[i] = self.graph[context_source]['size']
                    self.flow_ptr_repr_from[i] = self.repr_ptr[context_source]
                    self.flow_ptr_repr_to[i] = block["r_ptr"] + block['size'] + 1 + running_ptr
                    running_ptr += self.graph[context_source]['size']
                    i += 1

        for i in range(int(self.specs['layer_shapes'][0])*int(self.specs['layer_shapes'][0])):
            self.flow_ptr_input_frame[i] = self.graph[i]['i_ptr']
            self.flow_ptr_input_frame_size[i] = self.graph[i]['base_input_size']

    def create_mem_gpu(self):
        self.weight_memory_main_gpu = gpuarray.to_gpu(self.weight_memory_main)  # The main weight matrix
        self.dweight_memory_gpu = [gpuarray.to_gpu(self.list_dweight_memory[0]),
                                   gpuarray.to_gpu(self.list_dweight_memory[1])]  # weight cache for updates and momenta (2x arrays)
        self.weight_memory_cache0_gpu = gpuarray.to_gpu(self.weight_memory_cache0)
        self.input_mem_activation_gpu = [gpuarray.to_gpu(x) for x in self.input_mem_activation]  # x input seq, Input memory
        self.input_mem_delta_gpu = [gpuarray.to_gpu(x) for x in self.input_mem_delta]  # x input seq, Delta memory
        self.input_mem_error_gpu = [gpuarray.to_gpu(x) for x in self.input_mem_error]  # x input sex, error memory
        self.output_mem_activation_gpu = [gpuarray.to_gpu(x) for x in self.output_mem_activation]  # x input seq, Input memory
        self.output_mem_delta_gpu = [gpuarray.to_gpu(x) for x in self.output_mem_delta]  # x input seq, Delta memory
        self.output_mem_error_gpu = [gpuarray.to_gpu(x) for x in self.output_mem_error]  # x input sex, error memory
        self.repre_mem_activation_gpu = [gpuarray.to_gpu(x) for x in self.repre_mem_activation]  # x input seq, representation memory
        self.repre_mem_delta_gpu = [gpuarray.to_gpu(x) for x in self.repre_mem_delta]  # x input seq, representation memory
        self.repre_mem_error_gpu = [gpuarray.to_gpu(x) for x in self.repre_mem_error]  # x input seq, representation memory
        self.flow_ptr_from_gpu = gpuarray.to_gpu(self.flow_ptr_from)
        self.flow_ptr_to_gpu = gpuarray.to_gpu(self.flow_ptr_to)
        self.flow_ptr_size_gpu = gpuarray.to_gpu(self.flow_ptr_size)
        self.flow_ptr_repr_from_gpu = gpuarray.to_gpu(self.flow_ptr_repr_from)
        self.flow_ptr_repr_to_gpu = gpuarray.to_gpu(self.flow_ptr_repr_to)
        self.flow_ptr_repr_size_gpu = gpuarray.to_gpu(self.flow_ptr_repr_size)
        self.flow_ptr_input_shift_from_gpu = gpuarray.to_gpu(self.flow_ptr_input_shift_from)
        self.flow_ptr_input_shift_to_gpu = gpuarray.to_gpu(self.flow_ptr_input_shift_to)
        self.flow_ptr_input_shift_size_gpu = gpuarray.to_gpu(self.flow_ptr_input_shift_size)
        self.flow_ptr_input_frame_gpu = gpuarray.to_gpu(self.flow_ptr_input_frame)
        self.weight_ptr0_gpu = gpuarray.to_gpu(self.weight_ptr0)
        self.weight_ptr1_gpu = gpuarray.to_gpu(self.weight_ptr1)
        self.input_ptr_gpu = gpuarray.to_gpu(self.input_ptr)
        self.repr_ptr_gpu = gpuarray.to_gpu(self.repr_ptr)
        self.shape0_L0_gpu = gpuarray.to_gpu(self.shape0_L0)
        self.shape1_L0_gpu = gpuarray.to_gpu(self.shape1_L0)
        self.shape0_L1_gpu = gpuarray.to_gpu(self.shape0_L1)
        self.shape1_L1_gpu = gpuarray.to_gpu(self.shape1_L1)
        self.obj_id_k2_L0_gpu = gpuarray.to_gpu(self.obj_id_k2_L0)
        self.row_id_k2_L0_gpu = gpuarray.to_gpu(self.row_id_k2_L0)
        self.obj_id_k2_L1_gpu = gpuarray.to_gpu(self.obj_id_k2_L1)
        self.row_id_k2_L1_gpu = gpuarray.to_gpu(self.row_id_k2_L1)
        self.beta_input_gpu = gpuarray.to_gpu(self.beta_input)
        self.beta_repre_gpu = gpuarray.to_gpu(self.beta_repre)
        self.learning_rate_arr_gpu = gpuarray.to_gpu(self.learning_rate_arr)
        self.momentum_arr_gpu = gpuarray.to_gpu(self.momentum_arr)

    def update_learning_rate(self, override_rate=None):
        if self.step % int(self.specs['delay_each_layer_learning']) == 0:
            if self.step / int(self.specs['delay_each_layer_learning']) < len(self.specs['layer_shapes']):
                layer_to_enable = int(self.step / int(self.specs['delay_each_layer_learning']))
                begin_idx = 0
                end_idx = 0
                for l in range(layer_to_enable):
                    begin_idx += int(self.specs['layer_shapes'][l])**2
                end_idx=begin_idx + int(self.specs['layer_shapes'][layer_to_enable])**2
                print("Enabling layer %d" % layer_to_enable)
                print("Begin idx %d end idx %d" % (begin_idx, end_idx))
                self.learning_rate_arr[begin_idx:end_idx] = float(self.specs['initial_learning_rate'])
                self.learning_rate = float(self.specs['initial_learning_rate'])
                self.momentum_arr[begin_idx:end_idx] = float(self.specs['momentum'])
        if self.step == int(self.specs['delay_final_learning_rate']):
            self.learning_rate_arr[:] = float(self.specs['final_learning_rate'])
            self.learning_rate = float(self.specs['final_learning_rate'])
            print("Setting final learning rate")
        if 'delay_intermediate_learning_rate' in list(self.specs.keys()) and self.step == int(self.specs['delay_intermediate_learning_rate']):
            self.learning_rate_arr[:] = float(self.specs['intermediate_learning_rate'])
            self.learning_rate = float(self.specs['intermediate_learning_rate'])
            print("Setting intermediate learning rate")
        if override_rate is not None:
            self.learning_rate_arr[:] = override_rate
            self.learning_rate = override_rate
            print("Overriding PVM learning rate to " + str(override_rate))
        self.learning_rate_arr_gpu.set(self.learning_rate_arr)
        self.momentum_arr_gpu.set(self.momentum_arr)
        # if self.step % 1000 == 0:
        #     print ""
        #     print gpuarray.max(self.weight_memory_main_gpu)
        #     print gpuarray.min(self.weight_memory_main_gpu)
        #     #self.regularize(0.99)
        #     gpu_routines.gpu_clip(self.weight_memory_main_gpu,
        #                           np.float32(3.0),
        #                           np.int32(self.total_weights),
        #                           block=(min(self.total_weights, 128), 1, 1),
        #                           grid=(self.total_weights/128 + 1, 1))


    def freeze_learning(self):
        self.learning_rate_saved = self.learning_rate_arr.copy()
        self.momentum_saved = self.momentum_arr.copy()
        self.learning_rate_arr *= 0
        self.momentum_arr *= 0

    def unfreeze_learning(self):
        self.learning_rate_arr[:] = self.learning_rate_saved[:]
        self.momentum_arr[:] = self.momentum_saved[:]

    def get_data_from_gpu(self):
        self.weight_memory_main_gpu.get(self.weight_memory_main)
        self.dweight_memory_gpu[0].get(self.list_dweight_memory[0])
        self.dweight_memory_gpu[1].get(self.list_dweight_memory[1])
        self.weight_memory_cache0_gpu.get(self.weight_memory_cache0)
        list(map(lambda x: x[0].get(x[1]), list(zip(self.input_mem_activation_gpu, self.input_mem_activation))))
        list(map(lambda x: x[0].get(x[1]), list(zip(self.input_mem_delta_gpu, self.input_mem_delta))))
        list(map(lambda x: x[0].get(x[1]), list(zip(self.input_mem_error_gpu, self.input_mem_error))))
        list(map(lambda x: x[0].get(x[1]), list(zip(self.output_mem_activation_gpu, self.output_mem_activation))))
        list(map(lambda x: x[0].get(x[1]), list(zip(self.output_mem_delta_gpu, self.output_mem_delta))))
        list(map(lambda x: x[0].get(x[1]), list(zip(self.output_mem_error_gpu, self.output_mem_error))))
        list(map(lambda x: x[0].get(x[1]), list(zip(self.repre_mem_activation_gpu, self.repre_mem_activation))))
        list(map(lambda x: x[0].get(x[1]), list(zip(self.repre_mem_delta_gpu, self.repre_mem_delta))))
        list(map(lambda x: x[0].get(x[1]), list(zip(self.repre_mem_error_gpu, self.repre_mem_error))))

    def push_input(self, frame):
        """
        Load the frame into the memory
        :param frame:
        :return:
        """
        bs = int(self.specs['input_block_size'])
        for x in range(int(specs['layer_shapes'][0])):
            for y in range(int(specs['layer_shapes'][0])):
                i = x * int(specs['layer_shapes'][0]) + y
                self.input_buf_activation[self.current_frame][self.flow_ptr_input_frame[i]] = \
                    frame[x * bs:x * bs + bs, y * bs:y * bs + bs].flatten()

    def push_input_gpu(self, frame):
        if frame.dtype == np.uint8:
            gpu_frame = gpuarray.to_gpu(frame.astype(np.float32)/255)
        if frame.dtype == np.float32:
            gpu_frame = gpuarray.to_gpu(frame)
        # Shift the inputs
        gpu_routines.gpu_copy_blocks(self.input_mem_activation_gpu[(self.step - 1) % self.sequence_length],
                                     self.input_mem_activation_gpu[(self.step) % self.sequence_length],
                                     self.flow_ptr_input_shift_from_gpu,
                                     self.flow_ptr_input_shift_size_gpu,
                                     self.flow_ptr_input_shift_to_gpu,
                                     np.int32(self.total_units),
                                     block=(min(self.total_units, 128), 1, 1),
                                     grid=(self.total_units//128 + 1, 1))
        self.repre_mem_activation_gpu[(self.step) % self.sequence_length].fill(0)
        # Forward primary and context inputs

        total_projections = self.total_primary_projections + self.total_context_projections
        # Compy inputs and contexts, compress to (0.1, 0.9)
        gpu_routines.gpu_copy_blocks_comp(self.repre_mem_activation_gpu[(self.step - 1) % self.sequence_length],
                                     self.input_mem_activation_gpu[(self.step) % self.sequence_length],
                                     self.flow_ptr_from_gpu,
                                     self.flow_ptr_size_gpu,
                                     self.flow_ptr_to_gpu,
                                     np.int32(total_projections),
                                     block=(min(total_projections, 128), 1, 1),
                                     grid=(total_projections // 128 + 1, 1))
        if utils.check_if_enabled("feed_context_in_complex_layer", self.specs):
            gpu_routines.gpu_copy_blocks(self.repre_mem_activation_gpu[(self.step - 1) % self.sequence_length],
                                         self.repre_mem_activation_gpu[(self.step) % self.sequence_length],
                                         self.flow_ptr_repr_from_gpu,
                                         self.flow_ptr_repr_size_gpu,
                                         self.flow_ptr_repr_to_gpu,
                                         np.int32(self.total_context_projections),
                                         block=(min(self.total_context_projections, 128), 1, 1),
                                         grid=(self.total_context_projections // 128 + 1, 1))

        # Propagate the frame
        frame_patches = int(self.specs['layer_shapes'][0])*int(self.specs['layer_shapes'][0])
        if frame.shape[-1] == 3:
            gpu_routines.gpu_dist_frame(gpu_frame,
                                        self.input_mem_activation_gpu[(self.step) % self.sequence_length],
                                        self.flow_ptr_input_frame_gpu,
                                        np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])),
                                        np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])),
                                        np.int32(int(self.specs['layer_shapes'][0])),
                                        np.int32(int(self.specs['layer_shapes'][0])),
                                        np.int32(int(self.specs['input_block_size'])),
                                        np.int32(int(self.specs['input_block_size'])),
                                        np.int32(0),
                                        np.int32(frame_patches), block=(min(frame_patches, 128), 1, 1),
                                        grid=(frame_patches//128 + 1, 1))
        if frame.shape[-1] == 4:
            gpu_routines.gpu_dist_frame4(gpu_frame,
                                         self.input_mem_activation_gpu[(self.step) % self.sequence_length],
                                         self.flow_ptr_input_frame_gpu,
                                         np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])),
                                         np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])),
                                         np.int32(int(self.specs['layer_shapes'][0])),
                                         np.int32(int(self.specs['layer_shapes'][0])),
                                         np.int32(int(self.specs['input_block_size'])),
                                         np.int32(int(self.specs['input_block_size'])),
                                         np.int32(0),
                                         np.int32(frame_patches), block=(min(frame_patches, 128), 1, 1),
                                         grid=(frame_patches//128 + 1, 1))

    def pop_prediction(self, delta_step=0):
        xy = np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size']))
        gpu_frame = gpuarray.to_gpu(np.zeros((xy, xy, self.input_channels), dtype=np.float32))
        frame_patches = int(self.specs['layer_shapes'][0])*int(self.specs['layer_shapes'][0])
        if self.input_channels == 3:
            gpu_routines.gpu_collect_frame(gpu_frame,
                                           self.output_mem_activation_gpu[(self.step+delta_step) % self.sequence_length],
                                           self.flow_ptr_input_frame_gpu,
                                           np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])),
                                           np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])),
                                           np.int32(int(self.specs['layer_shapes'][0])),
                                           np.int32(int(self.specs['layer_shapes'][0])),
                                           np.int32(int(self.specs['input_block_size'])),
                                           np.int32(int(self.specs['input_block_size'])),
                                           np.int32(0),
                                           np.int32(frame_patches), block=(min(frame_patches, 128), 1, 1),
                                           grid=(frame_patches//128 + 1, 1))
        if self.input_channels == 4:
            gpu_routines.gpu_collect_frame4(gpu_frame,
                                            self.output_mem_activation_gpu[(self.step+delta_step) % self.sequence_length],
                                            self.flow_ptr_input_frame_gpu,
                                            np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])),
                                            np.int32(int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])),
                                            np.int32(int(self.specs['layer_shapes'][0])),
                                            np.int32(int(self.specs['layer_shapes'][0])),
                                            np.int32(int(self.specs['input_block_size'])),
                                            np.int32(int(self.specs['input_block_size'])),
                                            np.int32(0),
                                            np.int32(frame_patches), block=(min(frame_patches, 128), 1, 1),
                                            grid=(frame_patches//128 + 1, 1))

        return gpu_frame.get()

    def pop_layer(self, layer=0):
        frame = np.zeros((int(self.specs['layer_shapes'][layer]) * int(self.specs['hidden_block_size']),
                          int(self.specs['layer_shapes'][layer]) * int(self.specs['hidden_block_size'])), dtype=np.uint32)
        gpu_frame = gpuarray.to_gpu(frame)
        frame_patches = int(self.specs['layer_shapes'][layer])*int(self.specs['layer_shapes'][layer])
        gpu_routines.gpu_collect_activ(gpu_frame,
                                       self.repre_mem_activation_gpu[(self.step) % self.sequence_length],
                                       self.repr_ptr_gpu,
                                       np.int32(int(self.specs['layer_shapes'][layer]) * int(self.specs['hidden_block_size'])),
                                       np.int32(int(self.specs['layer_shapes'][layer]) * int(self.specs['hidden_block_size'])),
                                       np.int32(int(self.specs['layer_shapes'][layer])),
                                       np.int32(int(self.specs['layer_shapes'][layer])),
                                       np.int32(int(self.specs['hidden_block_size'])),
                                       np.int32(int(self.specs['hidden_block_size'])),
                                       np.int32(self.layer_ptrs[layer]),
                                       np.int32(frame_patches), block=(min(frame_patches, 128), 1, 1),
                                       grid=(frame_patches//128 + 1, 1))
        frame[:] = gpu_frame.get()
        return frame.astype(np.uint8)

    def pop_pred(self, frame):
        """
        Extract the frame prediction back from the memory
        :param frame:
        :return:
        """
        # not current frame, but some in the future
        bs = int(self.specs['input_block_size'])
        for x in range(int(specs['layer_shapes'][0])):
            for y in range(int(specs['layer_shapes'][0])):
                i = x * int(specs['layer_shapes'][0]) + y
                shape = frame[x * bs:x * bs + bs, y * bs:y * bs + bs].shape
                frame[x * bs:x * bs + bs, y * bs:y * bs + bs] = self.input_buf_activation[self.current_frame][self.flow_ptr_input_frame[i]].reshape(shape)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def forward_gpu(self):
        """
        Forward pass of the perceptron algorithm
        :return:
        """
        current_step = self.step % self.sequence_length
        gpu_routines.gpu_dot_fast_set_bias(self.weight_memory_main_gpu,
                                           self.input_mem_activation_gpu[current_step],
                                           self.repre_mem_activation_gpu[current_step], # result
                                           self.weight_ptr0_gpu,
                                           self.input_ptr_gpu,
                                           self.repr_ptr_gpu,
                                           self.shape0_L0_gpu,  # Weight shape
                                           self.shape1_L0_gpu,  # Weight shape
                                           self.obj_id_k2_L0_gpu,  # each weight needs an id
                                           self.row_id_k2_L0_gpu,  #
                                           np.int32(self.cuda_total_threads_k2_L0),
                                           block=(self.cuda_block_size, 1, 1),
                                           grid=(self.cuda_grid_L0, 1))
        if not self.poly:
            gpu_routines.gpu_sigmoid_fast(self.repre_mem_activation_gpu[current_step],
                                          self.repr_ptr_gpu,
                                          self.beta_repre_gpu,
                                          self.shape0_L0_gpu,
                                          np.int32(self.total_units), block=(512, 1, 1), grid=(self.total_units // 512 + 1, 1))
        else:
            gpu_routines.gpu_sigmoid_poly_fast(self.repre_mem_activation_gpu[current_step],
                                               self.repr_ptr_gpu,
                                               self.beta_repre_gpu,
                                               self.shape0_L0_gpu,
                                               np.int32(self.total_units), block=(512, 1, 1), grid=(self.total_units // 512 + 1, 1))

        # Output container needs to be zeroed
        self.output_mem_activation_gpu[current_step] *= 0
        gpu_routines.gpu_dot_fast_set_bias(self.weight_memory_main_gpu,
                                           self.repre_mem_activation_gpu[current_step],
                                           self.output_mem_activation_gpu[current_step],
                                           self.weight_ptr1_gpu,
                                           self.repr_ptr_gpu,
                                           self.input_ptr_gpu,
                                           self.shape0_L1_gpu,  # Weight shape
                                           self.shape1_L1_gpu,  # Weight shape
                                           self.obj_id_k2_L1_gpu,  # each weight needs an id
                                           self.row_id_k2_L1_gpu,  #
                                           np.int32(self.cuda_total_threads_k2_L1),
                                           block=(self.cuda_block_size, 1, 1),
                                           grid=(self.cuda_grid_L1, 1))

        if not self.poly:
            gpu_routines.gpu_sigmoid_fast(self.output_mem_activation_gpu[current_step],
                                          self.input_ptr_gpu,
                                          self.beta_input_gpu,
                                          self.shape0_L1_gpu,
                                          np.int32(self.total_units), block=(512, 1, 1), grid=(self.total_units // 512 + 1, 1))
        else:
            gpu_routines.gpu_sigmoid_poly_fast(self.output_mem_activation_gpu[current_step],
                                               self.input_ptr_gpu,
                                               self.beta_input_gpu,
                                               self.shape0_L1_gpu,
                                               np.int32(self.total_units), block=(512, 1, 1), grid=(self.total_units // 512 + 1, 1))



    def backward_gpu(self):
        """
        Backpropagation and weight updates

        :return:
        """
        activation_step = (self.step - self.sequence_interval) % self.sequence_length
        current_step = self.step % self.sequence_length

        self.output_mem_error_gpu[activation_step].fill(0)
        self.output_mem_error_gpu[activation_step] -= self.output_mem_activation_gpu[activation_step]
        self.output_mem_error_gpu[activation_step] += self.input_mem_activation_gpu[current_step]
        if utils.check_if_enabled("opt_abs_diff", self.specs):
            gpu_routines.gpu_sgn(self.output_mem_error_gpu[activation_step],
                                 np.int32(self.total_units), block=(512, 1, 1),
                                 grid=(self.total_units // 512 + 1, 1)
                                 )
        # Derivative of the output layer

        if not self.poly:
            gpu_routines.gpu_sigmoid_der_mul(self.output_mem_activation_gpu[activation_step],  # Activation
                                             self.output_mem_error_gpu[activation_step],  # Error
                                             self.output_mem_delta_gpu[activation_step],  # delta -> result
                                             self.input_ptr_gpu,  # Activ ptr
                                             self.input_ptr_gpu,  # error ptr
                                             self.input_ptr_gpu,  # delta ptr
                                             self.shape0_L1_gpu,    # shape
                                             np.int32(self.total_units), block=(512, 1, 1),
                                             grid=(self.total_units // 512 + 1, 1))
        else:
            gpu_routines.gpu_sigmoid_poly_der_mul(self.output_mem_activation_gpu[activation_step],  # Activation
                                                  self.output_mem_error_gpu[activation_step],  # Error
                                                  self.output_mem_delta_gpu[activation_step],  # delta -> result
                                                  self.input_ptr_gpu,  # Activ ptr
                                                  self.input_ptr_gpu,  # error ptr
                                                  self.input_ptr_gpu,  # delta ptr
                                                  self.shape0_L1_gpu,    # shape
                                                  np.int32(self.total_units), block=(512, 1, 1),
                                                  grid=(self.total_units // 512 + 1, 1))

        # Backpropagate to the representation layer
        gpu_routines.gpu_dot_transpose_fast(self.weight_memory_main_gpu,  # Matrix mem
                                            self.weight_memory_cache0_gpu,  # Buffer -> result
                                            self.output_mem_delta_gpu[activation_step],  # Delta mem
                                            self.weight_ptr1_gpu,  # Weight ptr
                                            self.input_ptr_gpu,  # Delta ptr
                                            self.shape0_L1_gpu,  # Weight shape
                                            self.shape1_L1_gpu,  # Weight shape
                                            self.obj_id_k2_L1_gpu,  # each weight needs an id
                                            self.row_id_k2_L1_gpu,  # row id
                                            np.int32(self.cuda_total_threads_k2_L1),
                                            block=(self.cuda_block_size, 1, 1),
                                            grid=(self.cuda_grid_L1, 1))
        self.repre_mem_error_gpu[activation_step].fill(0)
        gpu_routines.gpu_sum_dot_transpose(self.weight_memory_cache0_gpu,  # Buffer
                                           self.repre_mem_error_gpu[activation_step],  #
                                           self.weight_ptr1_gpu,  #
                                           self.repr_ptr_gpu,  #
                                           self.shape0_L1_gpu,  # Weight shape
                                           self.shape1_L1_gpu,  # Weight shape
                                           self.obj_id_k2_L1_gpu,  # each weight needs an id
                                           self.row_id_k2_L1_gpu,  # row id
                                           np.int32(self.cuda_total_threads_k2_L1),
                                           block=(self.cuda_block_size, 1, 1),
                                           grid=(self.cuda_grid_L1, 1))
        # O1=self.output_mem_delta_gpu[activation_step].get()[:self.shape0_L1[0]]
        # if np.mean(O1)<0:
        #     W1 = self.weight_memory_main_gpu.get()[
        #          self.weight_ptr1[0]:self.weight_ptr1[0] + self.shape0_L1[0] * self.shape1_L1[0]].reshape(
        #         (self.shape0_L1[0], self.shape1_L1[0]))
        #     print np.mean(O1)
        #     print W1.shape
        #     R1 = self.repre_mem_error_gpu[activation_step].get()[:self.shape1_L1[0]]
        #     R2 = np.dot(W1.T, O1)
        #     print np.mean(R1)
        if not self.poly:
            # Derivative in the representation layer
            gpu_routines.gpu_sigmoid_der_mul(self.repre_mem_activation_gpu[activation_step],  # Activation
                                             self.repre_mem_error_gpu[activation_step],  # Error
                                             self.repre_mem_delta_gpu[activation_step],  # delta -> result
                                             self.repr_ptr_gpu,  # Activ ptr
                                             self.repr_ptr_gpu,  # error ptr
                                             self.repr_ptr_gpu,  # delta ptr
                                             self.shape1_L1_gpu,  # shape
                                             np.int32(self.total_units), block=(512, 1, 1),
                                             grid=(self.total_units // 512 + 1, 1))
        else:
            gpu_routines.gpu_sigmoid_poly_der_mul(self.repre_mem_activation_gpu[activation_step],  # Activation
                                                  self.repre_mem_error_gpu[activation_step],  # Error
                                                  self.repre_mem_delta_gpu[activation_step],  # delta -> result
                                                  self.repr_ptr_gpu,  # Activ ptr
                                                  self.repr_ptr_gpu,  # error ptr
                                                  self.repr_ptr_gpu,  # delta ptr
                                                  self.shape1_L1_gpu,  # shape
                                                  np.int32(self.total_units), block=(512, 1, 1),
                                                  grid=(self.total_units // 512 + 1, 1))
        # Now for the weights.
        gpu_routines.gpu_generalized_outer_fast3(self.output_mem_delta_gpu[activation_step],  # Upper layer delta
                                                 self.repre_mem_activation_gpu[activation_step],  # Lower layer activity
                                                 self.dweight_memory_gpu[(self.buffer_index+1) % 2],  # Momentum matrix
                                                 self.dweight_memory_gpu[self.buffer_index],  # Weight change matrix (result)
                                                 self.input_ptr_gpu,  # Ptr to the upper delta
                                                 self.repr_ptr_gpu,  # Ptr to the input
                                                 self.weight_ptr1_gpu,  # Weight ptr for mom matrix
                                                 self.weight_ptr1_gpu,  # Weight ptr for result
                                                 self.shape0_L1_gpu,  # Weight shape
                                                 self.shape1_L1_gpu,  # Weight shape
                                                 self.learning_rate_arr_gpu,  # Learning rate
                                                 self.momentum_arr_gpu,  # Momentum
                                                 self.obj_id_k2_L1_gpu,  # each weight needs an id
                                                 self.row_id_k2_L1_gpu,  # row id
                                                 np.int32(self.cuda_total_threads_k2_L1),
                                                 block=(self.cuda_block_size, 1, 1),
                                                 grid=(self.cuda_grid_L1, 1))

        gpu_routines.gpu_generalized_outer_fast3(self.repre_mem_delta_gpu[activation_step],  # Upper layer delta
                                                 self.input_mem_activation_gpu[activation_step],  # Lower layer activity
                                                 self.dweight_memory_gpu[(self.buffer_index+1) % 2],  # Momentum matrix
                                                 self.dweight_memory_gpu[self.buffer_index],  # Weight change matrix (result)
                                                 self.repr_ptr_gpu,  # Ptr to the upper delta
                                                 self.input_ptr_gpu,  # Ptr to the input
                                                 self.weight_ptr0_gpu,  # Weight ptr for mom matrix
                                                 self.weight_ptr0_gpu,  # Weight ptr for result
                                                 self.shape0_L0_gpu,  # Weight shape 0
                                                 self.shape1_L0_gpu,  # Weight shape 1
                                                 self.learning_rate_arr_gpu,  # Learning rate
                                                 self.momentum_arr_gpu,  # Momentum
                                                 self.obj_id_k2_L0_gpu,  # each weight needs an id
                                                 self.row_id_k2_L0_gpu,  # row id
                                                 np.int32(self.cuda_total_threads_k2_L0),
                                                 block=(self.cuda_block_size, 1, 1),
                                                 grid=(self.cuda_grid_L0, 1))

        self.weight_memory_main_gpu += self.dweight_memory_gpu[self.buffer_index]
        # Flip buffers
        self.buffer_index = (self.buffer_index + 1) % 2


    def regularize(self, alpha):
        """
        Not necessary
        :param alpha:
        :return:
        """
        self.weight_memory_main_gpu *= alpha


    def drop_buf(self, myzip, filename, buf):
        print("Saving " + filename + " "*10 +"\r", end=' ')
        sys.stdout.flush()
        zipi = zipfile.ZipInfo()
        zipi.filename = filename
        zipi.date_time = time.localtime()[:6]
        zipi.compress_type = zipfile.ZIP_DEFLATED
        zipi.external_attr = 0o777 << 16
        myzip.writestr(zipi, buf)

    def save(self, outfile):
        self.get_data_from_gpu()
        myzip = zipfile.ZipFile(outfile, "a", allowZip64=True)
        folder = "pvm"
        for k in list(self.__dict__.keys()):
            if type(self.__dict__[k]) == np.ndarray:
                self.drop_buf(myzip, folder+"/"+k+".npy", memoryview(self.__dict__[k]))
                metainfo = [self.__dict__[k].shape, self.__dict__[k].dtype]
                GS = pickle.dumps(metainfo)
                self.drop_buf(myzip, folder + "/" + k + ".npy.meta", GS)
            elif "_gpu" not in k and not k.startswith("list_"):
                GS = pickle.dumps(self.__dict__[k])
                self.drop_buf(myzip, folder+"/"+k+".pkl", GS)
        myzip.close()

    def load(self, filename):
        myzip = zipfile.ZipFile(filename, "r")
        for (i, element) in enumerate(myzip.namelist()):
            if element.startswith('pvm'):
                print(("[ %d/%d ] Extracting " % (i, len(myzip.namelist()))) + element + " "*10 + "\r", end=' ')
                sys.stdout.flush()
                if element.endswith(".npy"):
                    handle = myzip.open(element)
                    metahandle = myzip.open(element+".meta")
                    metainfo = pickle.load(metahandle)
                    buf = handle.read()
                    npo = np.frombuffer(bytearray(buf), dtype=metainfo[1])
                    npo.reshape(metainfo[0])
                    npo.setflags(write=True)
                    obj_name=os.path.basename(element)[:-4]
                    self.__dict__[obj_name] = npo
                    print("Loaded numpy object " + obj_name + "\r", end=' ')
                    print("Dtype " + str(metainfo[1]) + " shape" + str(metainfo[0]) + " "*10 + "\r", end=' ')
                elif element.endswith(".pkl"):
                    handle = myzip.open(element)
                    obj = pickle.load(handle)
                    obj_name = os.path.basename(element)[:-4]
                    self.__dict__[obj_name] = obj
                    print("Loaded object " + obj_name + " "*10 + "\r", end=' ')
        self.list_dweight_memory = [self.dweight_memory_0, self.dweight_memory_1]
        self.list_weight_memory_cache = [self.weight_memory_cache0, self.weight_memory_cache1]
        if not "flow_ptr_repr_from" in list(self.__dict__.keys()):
            self.flow_ptr_repr_from = np.zeros(10, dtype=np.int32)
        if not "flow_ptr_repr_to" in list(self.__dict__.keys()):
            self.flow_ptr_repr_to = np.zeros(10, dtype=np.int32)
        if not "flow_ptr_repr_size" in list(self.__dict__.keys()):
            self.flow_ptr_repr_size = np.zeros(10, dtype=np.int32)
        self.learning_rate = self.learning_rate_arr[0]
        self.create_mem_gpu()
        self.device = pycuda.driver.Device(0).name()


    def get_input_shape(self):
        s1 = int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])
        s2 = int(self.specs['layer_shapes'][0]) * int(self.specs['input_block_size'])
        s3 = 3
        return (s1, s2, s3)

    def display_unit(self, i, delta_step=0):
        # Debugging mess, this should one day be converted into unit tests
        import pprint
        current_step = (self.step + delta_step) % self.sequence_length
        print("Unit %d" % i)
        print(self.graph[i])
        print("Weight shape 1 L0")
        print(self.shape1_L0)
        print("Weight shape 0 L0")
        print(self.shape0_L0)
        print("Weight shape 1 L1")
        print(self.shape1_L1)
        print("Weight shape 0 L1")
        print(self.shape0_L1)
        print("Inputs")
        print(self.input_mem_activation_gpu[current_step].get()[self.input_ptr[i]:self.input_ptr[i]+self.shape1_L0[i]])
        print("Representations")
        print(self.repre_mem_activation_gpu[current_step].get()[self.repr_ptr[i]:self.repr_ptr[i]+self.shape1_L1[i]])
        print("Outputs")
        print(self.output_mem_activation_gpu[current_step].get()[self.input_ptr[i]:self.input_ptr[i]+self.shape0_L1[i]])
        # print "Output error"
        # print self.output_mem_error_gpu[current_step].get()[self.input_ptr[i]:self.input_ptr[i]+self.shape0_L1[i]]
        # print "Output deltas"
        # print self.output_mem_delta_gpu[current_step].get()[self.input_ptr[i]:self.input_ptr[i]+self.shape0_L1[i]]
        #delta=self.output_mem_delta_gpu[current_step].get()[self.input_ptr[i]:self.input_ptr[i]+self.shape0_L1[i]]
        # print "Expected deltas"
        # O = self.output_mem_activation_gpu[current_step].get()[self.input_ptr[i]:self.input_ptr[i]+self.shape0_L1[i]]
        # E = self.output_mem_error_gpu[current_step].get()[self.input_ptr[i]:self.input_ptr[i]+self.shape0_L1[i]]
        # D = O*(1-O)*E
        # print "Output"
        # print O
        # print "Error"
        # print E
        # # print "Representation error"
        # print self.repre_mem_error_gpu[current_step].get()[self.repr_ptr[i]:self.repr_ptr[i]+self.shape1_L1[i]]
        # WW = self.weight_memory_main_gpu.get()[self.weight_ptr1[i]:self.weight_ptr1[i]+self.shape0_L1[i]*self.shape1_L1[i]]
        # W1 = WW.reshape(self.shape0_L1[i], self.shape1_L1[i])
        # # print W1.shape
        # print "Expected error"
        # print np.dot(W1.T, D)
        # print "Difference"
        # print self.repre_mem_error_gpu[current_step].get()[self.repr_ptr[i]:self.repr_ptr[i]+self.shape1_L1[i]] - np.dot(W1.T, D)
        # Iw = self.weight_memory_cache0_gpu.get()[self.weight_ptr1[i]:self.weight_ptr1[i]+self.shape0_L1[i]*self.shape1_L1[i]]
        # Iw = Iw.reshape(self.shape0_L1[i], self.shape1_L1[i])
        # eIw = (W1.T * D).T
        # # print "Weigths"
        # pprint.pprint(W1[:10, :10])
        # print "Vector"
        # pprint.pprint(D)
        # pprint.pprint(delta)
        # print "intermediate matrix"
        # print Iw[:10, :10]
        # print "Expected "
        # print eIw[:10, :10]
        # print "Difference"
        # print Iw[:10, :10] - eIw[:10, :10]
        # assert np.allclose(np.dot(W1.T, D), np.sum(eIw, axis=0))
        # Er = np.dot(W1.T, D)
        # Ar = self.repre_mem_activation_gpu[current_step].get()[self.repr_ptr[i]:self.repr_ptr[i]+self.shape1_L1[i]]
        # Dr = Er * Ar * (1-Ar)
        # print "Expected deltas at reprezentation"
        # print Dr
        # print self.repre_mem_delta_gpu[current_step].get()[self.repr_ptr[i]:self.repr_ptr[i]+self.shape1_L1[i]]
        # WW = self.weight_memory_main_gpu.get()[self.weight_ptr0[i]:self.weight_ptr0[i]+self.shape0_L0[i]*self.shape1_L0[i]]
        # W0 = WW.reshape(self.shape0_L0[i], self.shape1_L0[i])
        # DW = self.dweight_memory_gpu[self.buffer_index].get()[self.weight_ptr0[i]:self.weight_ptr0[i]+self.shape0_L0[i]*self.shape1_L0[i]]
        # DW0 = DW.reshape(self.shape0_L0[i], self.shape1_L0[i])
        # I = self.input_mem_activation_gpu[current_step].get()[self.input_ptr[i]:self.input_ptr[i]+self.shape1_L0[i]]
        # DWE = np.outer(I, Dr[:-1]).T * self.learning_rate_arr[i]
        # print DW0.shape
        # print DWE.shape
        # print DW0[:10,:10]
        # print DWE[:10,:10]
        # DW = self.dweight_memory_gpu[self.buffer_index].get()[self.weight_ptr1[i]:self.weight_ptr1[i]+self.shape0_L1[i]*self.shape1_L1[i]]
        # DW1 = DW.reshape(self.shape0_L1[i], self.shape1_L1[i])
        # DWE = np.outer(Ar, D[:-1]).T * self.learning_rate_arr[i]
        # print D
        # print Ar
        # print DW1[:10,:10]
        # print DWE[:10,:10]



if __name__ == "__main__":
    # print "Hello"
    # for devicenum in range(cuda.Device.count()):
    #     device = cuda.Device(devicenum)
    #     attrs = device.get_attributes()
    #
    #     # Beyond this point is just pretty printing
    #     print("\n===Attributes for device %d" % devicenum)
    #     for (key, value) in attrs.iteritems():
    #         print("%s:%s" % (str(key), str(value)))
    #
    #dev = drv.Device(0)
    #dev.make_context()
    sys.stdout.flush()
    parser = argparse.ArgumentParser(description="Description",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-L", "--load", help="Load a pre tgrained model", type=str, default="")
    parser.add_argument("-S", "--spec", help="Specification file name (file in .json format)", type=str, default="")
    parser.add_argument("-D", "--Display", help="Set display onn/off", action="store_true")
    parser.add_argument("-b", "--snapshot", help="Run some experiments", action="store_true")
    parser.add_argument("-e", "--experiment", help="Drop a snapshot", action="store_true")
    parser.add_argument('-O', '--options', type=json.loads,
                        help="Option dictionary (as described above) given in json form '{\"key1\": \"value1\"}\'.",
                        default='{}')
    args = parser.parse_args()
    print(args.spec)
    if args.spec != "" or args.load != "":
        if args.spec != "":
            name = os.path.splitext(os.path.basename(args.spec))[0]
            specs = json.load(open(args.spec, "r"))
            S = PVM_object(specs=specs, name=name)
        if args.load != "":
            S = PVM_object(specs=None)
            S.load(args.load)
        xx = int(S.specs['layer_shapes'][0]) * int(S.specs['input_block_size'])
        frames = utils.load_video("./IMG_4107.MOV", xx, xx)
        frame = 0*np.ones(shape=(xx, xx, 3), dtype=np.uint8)
        cv2.circle(frame, (30, 30), 10, (255, 255, 255), -1)
        frame[:, ::2, :] = 255
        D = disp.Display(width=900, height=400)
        pframe = frame.copy()
        S.push_input_gpu(frame)
        S.push_input_gpu(frame)
        #S.poly = True
        t0 = time.time()
        t1 = time.time()
        for x in range(100000000):
            S.update_learning_rate()
            frame = frames[x % len(frames)]
            if not args.experiment:
                S.push_input_gpu(frame)
            else:
                if x < 1000:
                    S.push_input_gpu(frame)
                else:
                     frame1 = frame.copy()
                     pfram1 = (pframe/255.0).astype(np.float32)
                #     frame1[50:70, 50:70] = 0.5
                #     #frame1[50:70, 50:70] = pfram1[50:70, 50:70]
                #     frame1[40:80, 40:80] = pfram1[40:80, 40:80]
                     #frame1[10:110, 10:110] = pfram1[10:110, 10:110]
                     frame1[:] = pfram1[:] # Full dream mode
                     S.push_input_gpu(frame1)
                     S.learning_rate = 0

            S.forward_gpu()
            S.pop_prediction(pframe)
            S.backward_gpu()
            # S.display_unit(50, delta_step=-S.sequence_interval)
            #S.display_unit(600)
            #if x % 1000 == 0:
            #    S.regularize(alpha=0.999)
            if args.Display or (args.snapshot and x % len(frames) == 100):
                activ0 = S.pop_layer(layer=0)
                activ1 = S.pop_layer(layer=1)
                activ2 = S.pop_layer(layer=2)
                activ3 = S.pop_layer(layer=3)
                activ4 = S.pop_layer(layer=4)
                activ5 = S.pop_layer(layer=5)
                activ6 = S.pop_layer(layer=6)
                activ7 = S.pop_layer(layer=7)
                activ8 = S.pop_layer(layer=8)
                D.place_rgb_float(10, 10, frame)
                D.place_rgb(10, 200, pframe)
                D.place_gray(150, 10, activ0)
                D.place_gray(330, 10, activ1)
                D.place_gray(480, 10, activ2)
                D.place_gray(600, 10, activ3)
                D.place_gray(700, 10, activ4)
                D.place_gray(770, 10, activ5)
                D.place_gray(820, 10, activ6)
                D.place_gray(855, 10, activ7)
                D.place_gray(875, 10, activ8)
                if args.experiment and x>990:
                    D.write("dream_%09d.png" % (S.step))
                if args.Display:
                    D.show("Activity")
                    k = cv2.waitKey(1) & 0xFF
                if args.snapshot and x % len(frames) == 100:
                    D.write("pngs/%s_pvm_state_%09d.png" % (S.uniq_id, S.step))
            else:
                k = 0
            #activ9 = S.pop_layer(layer=9)
            S.step += 1
            # cv2.imshow("Input", frame)
            # cv2.imshow("Pred", pframe)
            # cv2.imshow("Activity0", activ0)
            # cv2.imshow("Activity1", activ1)
            # cv2.imshow("Activity2", activ2)
            # cv2.imshow("Activity3", activ3)
            # cv2.imshow("Activity4", activ4)
            # cv2.imshow("Activity5", activ5)
            # cv2.imshow("Activity6", activ6)
            # cv2.imshow("Activity7", activ7)
            # cv2.imshow("Activity8", activ8)
            #cv2.imshow("Activity9", activ9)
            if k == ord('s') or (S.step > 1 and S.step % 500000 == 0):
                S.save("./Sim_%s_%s_%09d.zip" % (S.name, S.uniq_id, S.step))
            if k == ord('l'):
                S = PVM_object(specs=None)
                S.load("./dump.zip")
            if k == ord('Q'):
                exit(0)
        pycuda.driver.stop_profiler()
        exit()
        S.get_data_from_gpu()
        for x in range(100):
            print("#" * 80)
            S.push_input_gpu(frame)
            S.forward_gpu()
            print("Input mem")
            print(S.input_mem_activation_gpu[S.step % S.sequence_length].get()[:10])
            print("Representation mem")
            print(S.repre_mem_activation_gpu[S.step % S.sequence_length].get()[:10])
            print("Output mem")
            print(S.output_mem_activation_gpu[S.step % S.sequence_length].get()[:10])
            # S.forward_gpu()
            S.backward_gpu()
            print("Error mem")
            print(S.output_mem_error_gpu[(S.step - S.sequence_interval) % S.sequence_length].get()[:10])
            print("Delta mem")
            print(S.output_mem_delta_gpu[(S.step - S.sequence_interval) % S.sequence_length].get()[:10])
            print("Error mem repr")
            print(S.repre_mem_error_gpu[(S.step - S.sequence_interval) % S.sequence_length].get()[:10])
            print("Delta mem repr")
            print(S.repre_mem_delta_gpu[(S.step - S.sequence_interval) % S.sequence_length].get()[:10])
            print("Error mem input")
            print(S.input_mem_error_gpu[(S.step - S.sequence_interval) % S.sequence_length].get()[:10])
            print("Delta mem input")
            print(S.input_mem_delta_gpu[(S.step - S.sequence_interval) % S.sequence_length].get()[:10])

            S.step += 1
