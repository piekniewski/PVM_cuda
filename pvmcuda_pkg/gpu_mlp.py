# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info
import pvmcuda_pkg.gpu_routines as gpu_routines
import pycuda.gpuarray as gpuarray
import numpy as np
import pprint
import sys
import os
import cv2
import time
import pycuda.autoinit
import zipfile
import pickle
from .synthetic_data import SyntheticDataProvider


def ndarray_catch(original_format_func):
    def _format(*argv):
        myobj = argv[1]
        if issubclass(type(myobj), np.ndarray):
            array_text = 'array(dtype:' + str(myobj.dtype)
            array_text +='; shape:' + str(myobj.shape) + ')'
            argv = list(argv)
            argv[1] = array_text
        return original_format_func(*argv)
    return _format

pprint.PrettyPrinter._format = ndarray_catch(pprint.PrettyPrinter._format)


class MLP_collection(object):
    def __init__(self, specs=None, learning_rate=0.1, momentum=0.5):
        self.layerwise_objects = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        if specs is not None:
            self.layerwise_weight_mem_req = np.zeros((10,), dtype=np.int32)
            self.layerwise_weight_objects = np.zeros((10,), dtype=np.int32)
            self.layerwise_output_mem_req = np.zeros((10,), dtype=np.int32)
            self.flip_buf = False
            self.propagate_all_the_way = False
            self.input_layer_req = 0
            for current_layer in range(0, 10):
                for spec in specs:
                    for (i, layer) in enumerate(spec):
                        if i == current_layer:
                            if i < len(spec)-1:
                                self.layerwise_weight_mem_req[i] += (spec[i]+1) * (spec[i+1])
                                self.layerwise_weight_objects[i] += 1
                                self.layerwise_output_mem_req[i] += (spec[i + 1] + 1) # For the bias unit
                            if i == 0:
                                self.input_layer_req += (spec[i] + 1)  # For the bias
            # Thread num per block very much affects the performance
            # but it is uneasy to set it right once and for all.
            # Below is rough heuristics of what works best,
            # will need to be tuned more systematically
            # ( may also depend on the GPU)
            self.total_objects = max(self.layerwise_weight_objects)
            if self.total_objects > 8000:
                thread_num = 256
            elif self.total_objects > 3000:
                thread_num = 196
            else:
                thread_num = 128
            self.threads = int(min(self.total_objects, thread_num))
            self.total_objects_a = np.zeros(shape=(1,), dtype=np.int32)
            self.total_objects_a[0] = self.total_objects
            self.grid_size = int(self.total_objects // thread_num + 1)
            max_layers = 0
            for i in range(10):
                if self.layerwise_output_mem_req[i] == 0:
                    break
                max_layers += 1
            self.max_layers = max_layers
            for current_layer in range(max_layers):
                L1_weight_mem = np.zeros(shape=(self.layerwise_weight_mem_req[current_layer],), dtype=np.float32)
                L1_weight_mem_buf = np.zeros(shape=(self.layerwise_weight_mem_req[current_layer],), dtype=np.float32)
                buf_weight_mem = memoryview(L1_weight_mem)
                L1_we_ptr_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]+1,), dtype=np.int32)
                buf_we_ptr_mem = memoryview(L1_we_ptr_mem)

                L1_dweight_mem = np.zeros(shape=(self.layerwise_weight_mem_req[current_layer],), dtype=np.float32)
                buf_dweight_mem = memoryview(L1_dweight_mem)

                L1_mweight_mem = np.zeros(shape=(self.layerwise_weight_mem_req[current_layer],), dtype=np.float32)
                buf_mweight_mem = memoryview(L1_mweight_mem)

                L1_shape0_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]), dtype=np.int32)
                buf_shape0_mem = memoryview(L1_shape0_mem)
                L1_shape1_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]), dtype=np.int32)
                buf_shape1_mem = memoryview(L1_shape1_mem)
                beta_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]), dtype=np.float32) + 1
                buf_beta_mem = memoryview(beta_mem)

                learning_rate_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]), dtype=np.float32) + learning_rate
                buf_learning_rate_mem = memoryview(learning_rate_mem)

                momentum_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]), dtype=np.float32) + momentum
                buf_momentum_mem = memoryview(momentum_mem)


                L1_output_mem = np.zeros(shape=(self.layerwise_output_mem_req[current_layer],), dtype=np.float32)
                buf_output_mem = memoryview(L1_output_mem)
                L1_ou_ptr_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]+1,), dtype=np.int32)
                buf_ou_ptr_mem = memoryview(L1_ou_ptr_mem)

                L1_delta_mem = np.zeros(shape=(self.layerwise_output_mem_req[current_layer],), dtype=np.float32)
                buf_delta_mem = memoryview(L1_delta_mem)
                L1_de_ptr_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]+1,), dtype=np.int32)
                buf_de_ptr_mem = memoryview(L1_de_ptr_mem)

                L1_error_mem = np.zeros(shape=(self.layerwise_output_mem_req[current_layer],), dtype=np.float32)
                buf_error_mem = memoryview(L1_error_mem)
                L1_er_ptr_mem = np.zeros(shape=(self.layerwise_weight_objects[current_layer]+1,), dtype=np.int32)
                buf_er_ptr_mem = memoryview(L1_er_ptr_mem)

                for (j, spec) in enumerate(specs):
                    for (i, layer) in enumerate(spec):
                        if i == current_layer and current_layer < (max_layers):
                            L1_shape0_mem[j] = spec[i + 1]
                            L1_shape1_mem[j] = spec[i] + 1 # for bias
                weight_obj = []
                dweight_obj = []
                mweight_obj = []
                input_vec = []
                delta_vec = []
                error_vec = []
                for i in range(len(specs)):
                    mat = np.frombuffer(buf_weight_mem, dtype=np.float32, count=L1_shape0_mem[i] * L1_shape1_mem[i], offset=L1_we_ptr_mem[i] * 4).reshape(L1_shape0_mem[i], L1_shape1_mem[i])
                    dmat = np.frombuffer(buf_dweight_mem, dtype=np.float32, count=L1_shape0_mem[i] * L1_shape1_mem[i], offset=L1_we_ptr_mem[i] * 4).reshape(L1_shape0_mem[i], L1_shape1_mem[i])
                    mmat = np.frombuffer(buf_mweight_mem, dtype=np.float32, count=L1_shape0_mem[i] * L1_shape1_mem[i], offset=L1_we_ptr_mem[i] * 4).reshape(L1_shape0_mem[i], L1_shape1_mem[i])
                    mat[:] = (np.random.rand(mat.shape[0], mat.shape[1])-0.5)*2/np.sqrt(mat.shape[0]-0.9)
                    weight_obj.append(mat)
                    dweight_obj.append(dmat)
                    mweight_obj.append(mmat)
                    L1_we_ptr_mem[i+1] = L1_we_ptr_mem[i] + L1_shape0_mem[i] * L1_shape1_mem[i]

                    vec = np.frombuffer(buf_output_mem, dtype=np.float32, count=L1_shape0_mem[i]+1, offset=4 * L1_ou_ptr_mem[i])
                    vec[-1] = 1.0
                    L1_ou_ptr_mem[i+1] = L1_ou_ptr_mem[i] + L1_shape0_mem[i] + 1
                    input_vec.append(vec)
                    vec = np.frombuffer(buf_delta_mem, dtype=np.float32, count=L1_shape0_mem[i] + 1, offset=4 * L1_de_ptr_mem[i])
                    L1_de_ptr_mem[i+1] = L1_de_ptr_mem[i] + L1_shape0_mem[i] + 1
                    delta_vec.append(vec)
                    vec = np.frombuffer(buf_error_mem, dtype=np.float32, count=L1_shape0_mem[i] + 1, offset=4 * L1_er_ptr_mem[i])
                    L1_er_ptr_mem[i + 1] = L1_er_ptr_mem[i] + L1_shape0_mem[i] + 1
                    error_vec.append(vec)

                total_threads_k2 = 0
                for mat in weight_obj:
                    total_threads_k2 += mat.shape[1]

                obj_id_k2 = np.zeros(shape=(total_threads_k2,), dtype=np.int32)
                row_id = np.zeros(shape=(total_threads_k2,), dtype=np.int32)
                thread_id = 0
                for (i, mat) in enumerate(weight_obj):
                    for l in range(mat.shape[1]):
                        obj_id_k2[thread_id] = i
                        row_id[thread_id] = l
                        thread_id += 1
                block_size = 128
                grid_k2 = total_threads_k2 // block_size + 1

                self.layerwise_objects[current_layer]['layer'] = current_layer
                self.layerwise_objects[current_layer]['weight_mem'] = L1_weight_mem
                self.layerwise_objects[current_layer]['weight_mem_buf'] = L1_weight_mem_buf
                self.layerwise_objects[current_layer]['dweight_mem'] = L1_dweight_mem
                self.layerwise_objects[current_layer]['mweight_mem'] = L1_mweight_mem
                self.layerwise_objects[current_layer]['weight_ptr'] = L1_we_ptr_mem
                self.layerwise_objects[current_layer]['shape0'] = L1_shape0_mem
                self.layerwise_objects[current_layer]['shape1'] = L1_shape1_mem
                self.layerwise_objects[current_layer]['weight_obj'] = weight_obj
                self.layerwise_objects[current_layer]['dweight_obj'] = dweight_obj
                self.layerwise_objects[current_layer]['mweight_obj'] = mweight_obj
                self.layerwise_objects[current_layer]["weight_buf"] = buf_weight_mem
                self.layerwise_objects[current_layer]["we_ptr_buf"] = buf_we_ptr_mem
                self.layerwise_objects[current_layer]["shape0_buf"] = buf_shape0_mem
                self.layerwise_objects[current_layer]["shape1_buf"] = buf_shape1_mem
                self.layerwise_objects[current_layer]["obj_id"] = obj_id_k2
                self.layerwise_objects[current_layer]["row_id"] = row_id
                self.layerwise_objects[current_layer]["block_size"] = block_size
                self.layerwise_objects[current_layer]["grid"] = grid_k2
                self.layerwise_objects[current_layer]["total_threads"] = total_threads_k2
                self.layerwise_objects[current_layer]["beta_mem"] = beta_mem
                self.layerwise_objects[current_layer]["beta_buf"] = buf_beta_mem
                self.layerwise_objects[current_layer]["learning_rate_mem"] = learning_rate_mem
                self.layerwise_objects[current_layer]["learning_rate_buf"] = buf_learning_rate_mem
                self.layerwise_objects[current_layer]["momentum_mem"] = momentum_mem
                self.layerwise_objects[current_layer]["momentum_buf"] = buf_momentum_mem
                self.layerwise_objects[current_layer + 1]['input_obj'] = input_vec
                self.layerwise_objects[current_layer + 1]['delta_obj'] = delta_vec
                self.layerwise_objects[current_layer + 1]['error_obj'] = error_vec
                self.layerwise_objects[current_layer + 1]['delta_mem'] = L1_delta_mem
                self.layerwise_objects[current_layer + 1]["buf_delta"] = buf_delta_mem
                self.layerwise_objects[current_layer + 1]["delta_ptr"] = L1_de_ptr_mem
                self.layerwise_objects[current_layer + 1]["buf_de_ptr"] = buf_de_ptr_mem
                self.layerwise_objects[current_layer + 1]['input_mem'] = L1_output_mem
                self.layerwise_objects[current_layer + 1]["buf_input"] = buf_output_mem
                self.layerwise_objects[current_layer + 1]["input_ptr"] = L1_ou_ptr_mem
                self.layerwise_objects[current_layer + 1]["buf_in_ptr"] = buf_ou_ptr_mem
                self.layerwise_objects[current_layer + 1]['error_mem'] = L1_error_mem
                self.layerwise_objects[current_layer + 1]['error_ptr'] = L1_er_ptr_mem
                self.layerwise_objects[current_layer + 1]['buf_error'] = buf_error_mem
                self.layerwise_objects[current_layer + 1]['buf_er_ptr'] = buf_er_ptr_mem
            # create the input layer
            input_vec = []
            delta_vec = []
            error_vec = []
            L1_input_mem = np.zeros(shape=(self.input_layer_req,), dtype=np.float32)
            buf_input_mem = memoryview(L1_input_mem)
            L1_in_ptr_mem = np.zeros(shape=(len(specs)+1,), dtype=np.int32)
            buf_in_ptr_mem = memoryview(L1_in_ptr_mem)
            L1_delta_mem = np.zeros(shape=(self.input_layer_req,), dtype=np.float32)
            buf_delta_mem = memoryview(L1_delta_mem)
            L1_de_ptr_mem = np.zeros(shape=(len(specs)+1,), dtype=np.int32)
            buf_de_ptr_mem = memoryview(L1_de_ptr_mem)
            L1_error_mem = np.zeros(shape=(self.input_layer_req,), dtype=np.float32)
            buf_error_mem = memoryview(L1_error_mem)
            L1_er_ptr_mem = np.zeros(shape=(len(specs)+1,), dtype=np.int32)
            buf_er_ptr_mem = memoryview(L1_er_ptr_mem)

            for i in range(len(specs)):
                shape1 = self.layerwise_objects[0]['shape1'][i]
                vec = np.frombuffer(buf_input_mem, dtype=np.float32, count=shape1, offset=4 * L1_in_ptr_mem[i])
                vec[-1] = 1.0
                L1_in_ptr_mem[i + 1] = L1_in_ptr_mem[i] + shape1
                input_vec.append(vec)
                vec = np.frombuffer(buf_delta_mem, dtype=np.float32, count=shape1, offset=4 * L1_de_ptr_mem[i])
                L1_de_ptr_mem[i + 1] = L1_de_ptr_mem[i] + shape1
                delta_vec.append(vec)
                vec = np.frombuffer(buf_error_mem, dtype=np.float32, count=shape1, offset=4 * L1_er_ptr_mem[i])
                L1_er_ptr_mem[i + 1] = L1_er_ptr_mem[i] + shape1
                error_vec.append(vec)

            self.layerwise_objects[0]['input_obj'] = input_vec
            self.layerwise_objects[0]['delta_obj'] = delta_vec
            self.layerwise_objects[0]['error_obj'] = error_vec
            self.layerwise_objects[0]['input_mem'] = L1_input_mem
            self.layerwise_objects[0]['input_ptr'] = L1_in_ptr_mem
            self.layerwise_objects[0]['buf_input'] = buf_input_mem
            self.layerwise_objects[0]['buf_in_ptr'] = buf_in_ptr_mem
            self.layerwise_objects[0]['delta_mem'] = L1_delta_mem
            self.layerwise_objects[0]['delta_ptr'] = L1_de_ptr_mem
            self.layerwise_objects[0]['buf_delta'] = buf_delta_mem
            self.layerwise_objects[0]['buf_de_ptr'] = buf_de_ptr_mem
            self.layerwise_objects[0]['error_mem'] = L1_error_mem
            self.layerwise_objects[0]['error_ptr'] = L1_er_ptr_mem
            self.layerwise_objects[0]['buf_error'] = buf_error_mem
            self.layerwise_objects[0]['buf_er_ptr'] = buf_er_ptr_mem

    def generate_gpu_mem(self):
        for i in range(self.max_layers):
            self.layerwise_objects[i]['gpu_weight_mem'] = gpuarray.to_gpu(self.layerwise_objects[i]['weight_mem'])
            self.layerwise_objects[i]['gpu_weight_mem_buf'] = gpuarray.to_gpu(self.layerwise_objects[i]['weight_mem_buf'])
            self.layerwise_objects[i]['gpu_dweight_mem'] = gpuarray.to_gpu(self.layerwise_objects[i]['dweight_mem'])
            self.layerwise_objects[i]['gpu_mweight_mem'] = gpuarray.to_gpu(self.layerwise_objects[i]['mweight_mem'])
            self.layerwise_objects[i]['gpu_weight_ptr'] = gpuarray.to_gpu(self.layerwise_objects[i]['weight_ptr'])
            self.layerwise_objects[i]['gpu_shape0'] = gpuarray.to_gpu(self.layerwise_objects[i]['shape0'])
            self.layerwise_objects[i]['gpu_shape1'] = gpuarray.to_gpu(self.layerwise_objects[i]['shape1'])
            self.layerwise_objects[i]['gpu_beta'] = gpuarray.to_gpu(self.layerwise_objects[i]['beta_mem'])
            self.layerwise_objects[i]['gpu_obj_id'] = gpuarray.to_gpu(self.layerwise_objects[i]['obj_id'])
            self.layerwise_objects[i]['gpu_row_id'] = gpuarray.to_gpu(self.layerwise_objects[i]['row_id'])
            self.layerwise_objects[i]['gpu_learning_rate'] = gpuarray.to_gpu(self.layerwise_objects[i]['learning_rate_mem'])
            self.layerwise_objects[i]['gpu_momentum'] = gpuarray.to_gpu(self.layerwise_objects[i]['momentum_mem'])
            self.layerwise_objects[i]['gpu_input_mem'] = gpuarray.to_gpu(self.layerwise_objects[i]['input_mem'])
            self.layerwise_objects[i]['gpu_input_ptr'] = gpuarray.to_gpu(self.layerwise_objects[i]['input_ptr'])
            self.layerwise_objects[i]['gpu_delta_mem'] = gpuarray.to_gpu(self.layerwise_objects[i]['delta_mem'])
            self.layerwise_objects[i]['gpu_delta_ptr'] = gpuarray.to_gpu(self.layerwise_objects[i]['delta_ptr'])
            self.layerwise_objects[i]['gpu_error_mem'] = gpuarray.to_gpu(self.layerwise_objects[i]['error_mem'])
            self.layerwise_objects[i]['gpu_error_ptr'] = gpuarray.to_gpu(self.layerwise_objects[i]['error_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_shape1'] = gpuarray.to_gpu(self.layerwise_objects[self.max_layers-1]['shape0'])
        self.layerwise_objects[self.max_layers]['gpu_input_mem'] = gpuarray.to_gpu(self.layerwise_objects[self.max_layers]['input_mem'])
        self.layerwise_objects[self.max_layers]['gpu_input_ptr'] = gpuarray.to_gpu(self.layerwise_objects[self.max_layers]['input_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_delta_mem'] = gpuarray.to_gpu(self.layerwise_objects[self.max_layers]['delta_mem'])
        self.layerwise_objects[self.max_layers]['gpu_delta_ptr'] = gpuarray.to_gpu(self.layerwise_objects[self.max_layers]['delta_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_error_mem'] = gpuarray.to_gpu(self.layerwise_objects[self.max_layers]['error_mem'])
        self.layerwise_objects[self.max_layers]['gpu_error_ptr'] = gpuarray.to_gpu(self.layerwise_objects[self.max_layers]['error_ptr'])
        self.total_objects_gpu = gpuarray.to_gpu(self.total_objects_a)

    def set_gpu_error(self):
        self.layerwise_objects[self.max_layers]['gpu_error_mem'].set(self.layerwise_objects[self.max_layers]['error_mem'])

    def set_gpu_input(self):
        self.layerwise_objects[0]['gpu_input_mem'].set(self.layerwise_objects[0]['input_mem'])



    def set_gpu_mem(self):
        for i in range(self.max_layers):
            self.layerwise_objects[i]['gpu_weight_mem'].set(self.layerwise_objects[i]['weight_mem'])
            self.layerwise_objects[i]['gpu_weight_mem_buf'].set(self.layerwise_objects[i]['weight_mem_buf'])
            self.layerwise_objects[i]['gpu_dweight_mem'].set(self.layerwise_objects[i]['dweight_mem'])
            self.layerwise_objects[i]['gpu_mweight_mem'].set(self.layerwise_objects[i]['mweight_mem'])
            self.layerwise_objects[i]['gpu_weight_ptr'].set(self.layerwise_objects[i]['weight_ptr'])
            self.layerwise_objects[i]['gpu_shape0'].set(self.layerwise_objects[i]['shape0'])
            self.layerwise_objects[i]['gpu_shape1'].set(self.layerwise_objects[i]['shape1'])
            self.layerwise_objects[i]['gpu_beta'].set(self.layerwise_objects[i]['beta_mem'])
            self.layerwise_objects[i]['gpu_obj_id'].set(self.layerwise_objects[i]['obj_id'])
            self.layerwise_objects[i]['gpu_row_id'].set(self.layerwise_objects[i]['row_id'])
            self.layerwise_objects[i]['gpu_learning_rate'].set(self.layerwise_objects[i]['learning_rate_mem'])
            self.layerwise_objects[i]['gpu_momentum'].set(self.layerwise_objects[i]['momentum_mem'])
            self.layerwise_objects[i]['gpu_input_mem'].set(self.layerwise_objects[i]['input_mem'])
            self.layerwise_objects[i]['gpu_input_ptr'].set(self.layerwise_objects[i]['input_ptr'])
            self.layerwise_objects[i]['gpu_delta_mem'].set(self.layerwise_objects[i]['delta_mem'])
            self.layerwise_objects[i]['gpu_delta_ptr'].set(self.layerwise_objects[i]['delta_ptr'])
            self.layerwise_objects[i]['gpu_error_mem'].set(self.layerwise_objects[i]['error_mem'])
            self.layerwise_objects[i]['gpu_error_ptr'].set(self.layerwise_objects[i]['error_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_shape1'].set(self.layerwise_objects[self.max_layers-1]['shape0'])
        self.layerwise_objects[self.max_layers]['gpu_input_mem'].set(self.layerwise_objects[self.max_layers]['input_mem'])
        self.layerwise_objects[self.max_layers]['gpu_input_ptr'].set(self.layerwise_objects[self.max_layers]['input_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_delta_mem'].set(self.layerwise_objects[self.max_layers]['delta_mem'])
        self.layerwise_objects[self.max_layers]['gpu_delta_ptr'].set(self.layerwise_objects[self.max_layers]['delta_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_error_mem'].set(self.layerwise_objects[self.max_layers]['error_mem'])
        self.layerwise_objects[self.max_layers]['gpu_error_ptr'].set(self.layerwise_objects[self.max_layers]['error_ptr'])

    def get_gpu_mem(self):
        for i in range(self.max_layers):
            self.layerwise_objects[i]['gpu_weight_mem'].get(self.layerwise_objects[i]['weight_mem'])
            self.layerwise_objects[i]['gpu_weight_mem_buf'].get(self.layerwise_objects[i]['weight_mem_buf'])
            self.layerwise_objects[i]['gpu_dweight_mem'].get(self.layerwise_objects[i]['dweight_mem'])
            self.layerwise_objects[i]['gpu_mweight_mem'].get(self.layerwise_objects[i]['mweight_mem'])
            self.layerwise_objects[i]['gpu_weight_ptr'].get(self.layerwise_objects[i]['weight_ptr'])
            self.layerwise_objects[i]['gpu_shape0'].get(self.layerwise_objects[i]['shape0'])
            self.layerwise_objects[i]['gpu_shape1'].get(self.layerwise_objects[i]['shape1'])
            self.layerwise_objects[i]['gpu_beta'].get(self.layerwise_objects[i]['beta_mem'])
            self.layerwise_objects[i]['gpu_obj_id'].get(self.layerwise_objects[i]['obj_id'])
            self.layerwise_objects[i]['gpu_row_id'].get(self.layerwise_objects[i]['row_id'])
            self.layerwise_objects[i]['gpu_learning_rate'].get(self.layerwise_objects[i]['learning_rate_mem'])
            self.layerwise_objects[i]['gpu_momentum'].get(self.layerwise_objects[i]['momentum_mem'])
            self.layerwise_objects[i]['gpu_input_mem'].get(self.layerwise_objects[i]['input_mem'])
            self.layerwise_objects[i]['gpu_input_ptr'].get(self.layerwise_objects[i]['input_ptr'])
            self.layerwise_objects[i]['gpu_delta_mem'].get(self.layerwise_objects[i]['delta_mem'])
            self.layerwise_objects[i]['gpu_delta_ptr'].get(self.layerwise_objects[i]['delta_ptr'])
            self.layerwise_objects[i]['gpu_error_mem'].get(self.layerwise_objects[i]['error_mem'])
            self.layerwise_objects[i]['gpu_error_ptr'].get(self.layerwise_objects[i]['error_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_shape1'].get(self.layerwise_objects[self.max_layers-1]['shape0'])
        self.layerwise_objects[self.max_layers]['gpu_input_mem'].get(self.layerwise_objects[self.max_layers]['input_mem'])
        self.layerwise_objects[self.max_layers]['gpu_input_ptr'].get(self.layerwise_objects[self.max_layers]['input_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_delta_mem'].get(self.layerwise_objects[self.max_layers]['delta_mem'])
        self.layerwise_objects[self.max_layers]['gpu_delta_ptr'].get(self.layerwise_objects[self.max_layers]['delta_ptr'])
        self.layerwise_objects[self.max_layers]['gpu_error_mem'].get(self.layerwise_objects[self.max_layers]['error_mem'])
        self.layerwise_objects[self.max_layers]['gpu_error_ptr'].get(self.layerwise_objects[self.max_layers]['error_ptr'])

    def forward(self, poly=False):
        if not poly:
            for i in range(self.max_layers):
                gpu_routines.py_dot_sigmoid(self.layerwise_objects[i]['weight_obj'],
                                            self.layerwise_objects[i]['input_obj'],
                                            self.layerwise_objects[i+1]['input_obj'],
                                            self.layerwise_objects[i]['beta_mem'],
                                            self.layerwise_weight_objects[i])
        else:
            for i in range(self.max_layers):
                gpu_routines.py_dot_sigmoid_poly(self.layerwise_objects[i]['weight_obj'],
                                                 self.layerwise_objects[i]['input_obj'],
                                                 self.layerwise_objects[i+1]['input_obj'],
                                                 self.layerwise_weight_objects[i])


    def forward_gpu(self, poly=False, set_input=False):
        if set_input:
            self.set_gpu_input()
        for i in range(self.max_layers):
            self.layerwise_objects[i + 1]['gpu_input_mem'].fill(0)
            gpu_routines.gpu_dot_fast_set_bias(self.layerwise_objects[i]['gpu_weight_mem'],
                                               self.layerwise_objects[i]['gpu_input_mem'],
                                               self.layerwise_objects[i + 1]['gpu_input_mem'],
                                               self.layerwise_objects[i]['gpu_weight_ptr'],
                                               self.layerwise_objects[i]['gpu_input_ptr'],
                                               self.layerwise_objects[i + 1]['gpu_input_ptr'],
                                               self.layerwise_objects[i]['gpu_shape0'],
                                               self.layerwise_objects[i]['gpu_shape1'],
                                               self.layerwise_objects[i]['gpu_obj_id'],
                                               self.layerwise_objects[i]['gpu_row_id'],
                                               np.int32(self.layerwise_objects[i]['total_threads']),
                                               block=(self.layerwise_objects[i]['block_size'], 1, 1),
                                               grid=(self.layerwise_objects[i]['grid'], 1))
            if not poly:
                gpu_routines.gpu_sigmoid_fast(self.layerwise_objects[i+1]['gpu_input_mem'],
                                              self.layerwise_objects[i + 1]['gpu_input_ptr'],
                                              self.layerwise_objects[i]['gpu_beta'],
                                              self.layerwise_objects[i]['gpu_shape0'],
                                              np.int32(self.total_objects), block=(512, 1, 1), grid=(self.grid_size, 1))
            else:
                gpu_routines.gpu_sigmoid_poly_fast(self.layerwise_objects[i + 1]['gpu_input_mem'],
                                                   self.layerwise_objects[i + 1]['gpu_input_ptr'],
                                                   self.layerwise_objects[i]['gpu_beta'],
                                                   self.layerwise_objects[i]['gpu_shape0'],
                                                   np.int32(self.total_objects), block=(512, 1, 1), grid=(self.grid_size, 1))
        self.layerwise_objects[self.max_layers]['gpu_input_mem'].get(self.layerwise_objects[self.max_layers]['input_mem'])
        self.layerwise_objects[self.max_layers]['gpu_delta_mem'].get(self.layerwise_objects[self.max_layers]['delta_mem'])
        self.layerwise_objects[self.max_layers]['gpu_error_mem'].get(self.layerwise_objects[self.max_layers]['error_mem'])


    def backward(self, poly=False):
        for i in range(self.max_layers, -1, -1):
            if not poly:
                gpu_routines.py_sigmoid_der_mul(self.layerwise_objects[i]['input_obj'],
                                                self.layerwise_objects[i]['error_obj'],
                                                self.layerwise_objects[i]['delta_obj'],
                                                self.total_objects)
            else:
                gpu_routines.py_sigmoid_poly_der_mul(self.layerwise_objects[i]['input_obj'],
                                                     self.layerwise_objects[i]['error_obj'],
                                                     self.layerwise_objects[i]['delta_obj'],
                                                     self.total_objects)

            if i > 0:
                gpu_routines.py_dot_transpose(self.layerwise_objects[i - 1]['weight_obj'],
                                              self.layerwise_objects[i]['delta_obj'],
                                              self.layerwise_objects[i - 1]['error_obj'],
                                              self.total_objects)
                if not self.flip_buf:
                    gpu_routines.py_generalized_outer(self.layerwise_objects[i]['delta_obj'],
                                                       self.layerwise_objects[i - 1]['input_obj'],
                                                       self.layerwise_objects[i - 1]['mweight_obj'],
                                                       self.layerwise_objects[i - 1]['dweight_obj'],
                                                       self.layerwise_objects[i - 1]['learning_rate_mem'],
                                                       self.layerwise_objects[i - 1]['momentum_mem'],
                                                       self.total_objects)
                    self.layerwise_objects[i - 1]['weight_mem'][:] += self.layerwise_objects[i - 1]['dweight_mem'][:]
                else:
                    gpu_routines.py_generalized_outer(self.layerwise_objects[i]['delta_obj'],
                                                       self.layerwise_objects[i - 1]['input_obj'],
                                                       self.layerwise_objects[i - 1]['dweight_obj'],
                                                       self.layerwise_objects[i - 1]['mweight_obj'],
                                                       self.layerwise_objects[i - 1]['learning_rate_mem'],
                                                       self.layerwise_objects[i - 1]['momentum_mem'],
                                                       self.total_objects)
                    self.layerwise_objects[i - 1]['weight_mem'][:] += self.layerwise_objects[i - 1]['mweight_mem'][:]
        self.flip_buf = not self.flip_buf


    def backward_gpu(self, poly=False, set_error=False, propagate_all_way=False, propagate_only=False):
        if set_error:
            self.set_gpu_error()
        for i in range(self.max_layers, -1, -1):

            if not poly:
                gpu_routines.gpu_sigmoid_der_mul(self.layerwise_objects[i]['gpu_input_mem'],
                                                 self.layerwise_objects[i]['gpu_error_mem'],
                                                 self.layerwise_objects[i]['gpu_delta_mem'],
                                                 self.layerwise_objects[i]['gpu_input_ptr'],
                                                 self.layerwise_objects[i]['gpu_error_ptr'],
                                                 self.layerwise_objects[i]['gpu_delta_ptr'],
                                                 self.layerwise_objects[i]['gpu_shape1'],
                                                 np.int32(self.total_objects), block=(self.threads, 1, 1), grid=(self.grid_size, 1))
            else:
                gpu_routines.gpu_sigmoid_poly_der_mul(self.layerwise_objects[i]['gpu_input_mem'],
                                                      self.layerwise_objects[i]['gpu_error_mem'],
                                                      self.layerwise_objects[i]['gpu_delta_mem'],
                                                      self.layerwise_objects[i]['gpu_input_ptr'],
                                                      self.layerwise_objects[i]['gpu_error_ptr'],
                                                      self.layerwise_objects[i]['gpu_delta_ptr'],
                                                      self.layerwise_objects[i]['gpu_shape1'],
                                                      np.int32(self.total_objects), block=(self.threads, 1, 1), grid=(self.grid_size, 1))
            if i > 0:
                if i > 1 or i == 1 and (self.propagate_all_the_way or propagate_all_way):
                    self.layerwise_objects[i - 1]['gpu_error_mem'].fill(0)
                    gpu_routines.gpu_dot_transpose_fast(self.layerwise_objects[i - 1]['gpu_weight_mem'],
                                                        self.layerwise_objects[i - 1]['gpu_weight_mem_buf'],
                                                        self.layerwise_objects[i]['gpu_delta_mem'],
                                                        self.layerwise_objects[i - 1]['gpu_weight_ptr'],
                                                        self.layerwise_objects[i]['gpu_delta_ptr'],
                                                        self.layerwise_objects[i - 1]['gpu_shape0'],
                                                        self.layerwise_objects[i - 1]['gpu_shape1'],
                                                        self.layerwise_objects[i - 1]['gpu_obj_id'],
                                                        self.layerwise_objects[i - 1]['gpu_row_id'],
                                                        np.int32(self.layerwise_objects[i - 1]['total_threads']),
                                                        block=(self.layerwise_objects[i - 1]['block_size'], 1, 1),
                                                        grid=(self.layerwise_objects[i - 1]['grid'], 1))
                    gpu_routines.gpu_sum_dot_transpose(self.layerwise_objects[i - 1]['gpu_weight_mem_buf'],
                                                       self.layerwise_objects[i - 1]['gpu_error_mem'],
                                                       self.layerwise_objects[i - 1]['gpu_weight_ptr'],
                                                       self.layerwise_objects[i - 1]['gpu_error_ptr'],
                                                       self.layerwise_objects[i - 1]['gpu_shape0'],
                                                       self.layerwise_objects[i - 1]['gpu_shape1'],
                                                       self.layerwise_objects[i - 1]['gpu_obj_id'],
                                                       self.layerwise_objects[i - 1]['gpu_row_id'],
                                                       np.int32(self.layerwise_objects[i - 1]['total_threads']),
                                                       block=(self.layerwise_objects[i - 1]['block_size'], 1, 1),
                                                       grid=(self.layerwise_objects[i - 1]['grid'], 1))
                if not self.flip_buf:
                    gpu_routines.gpu_generalized_outer_fast(self.layerwise_objects[i]['gpu_delta_mem'],
                                                            self.layerwise_objects[i - 1]['gpu_input_mem'],
                                                            self.layerwise_objects[i - 1]['gpu_mweight_mem'],
                                                            self.layerwise_objects[i - 1]['gpu_dweight_mem'],
                                                            self.layerwise_objects[i]['gpu_delta_ptr'],
                                                            self.layerwise_objects[i - 1]['gpu_input_ptr'],
                                                            self.layerwise_objects[i - 1]['gpu_weight_ptr'],
                                                            self.layerwise_objects[i - 1]['gpu_weight_ptr'],
                                                            self.layerwise_objects[i - 1]['gpu_shape0'],
                                                            self.layerwise_objects[i - 1]['gpu_shape1'],
                                                            self.layerwise_objects[i - 1]['gpu_learning_rate'],
                                                            self.layerwise_objects[i - 1]['gpu_momentum'],
                                                            self.layerwise_objects[i - 1]['gpu_obj_id'],
                                                            self.layerwise_objects[i - 1]['gpu_row_id'],
                                                            np.int32(self.layerwise_objects[i - 1]['total_threads']),
                                                            block=(self.layerwise_objects[i - 1]['block_size'], 1, 1),
                                                            grid=(self.layerwise_objects[i - 1]['grid'], 1))
                    self.layerwise_objects[i - 1]['gpu_weight_mem'] += (self.layerwise_objects[i - 1]['gpu_dweight_mem'])
                else:
                    gpu_routines.gpu_generalized_outer_fast(self.layerwise_objects[i]['gpu_delta_mem'],
                                                            self.layerwise_objects[i - 1]['gpu_input_mem'],
                                                            self.layerwise_objects[i - 1]['gpu_dweight_mem'],
                                                            self.layerwise_objects[i - 1]['gpu_mweight_mem'],
                                                            self.layerwise_objects[i]['gpu_delta_ptr'],
                                                            self.layerwise_objects[i - 1]['gpu_input_ptr'],
                                                            self.layerwise_objects[i - 1]['gpu_weight_ptr'],
                                                            self.layerwise_objects[i - 1]['gpu_weight_ptr'],
                                                            self.layerwise_objects[i - 1]['gpu_shape0'],
                                                            self.layerwise_objects[i - 1]['gpu_shape1'],
                                                            self.layerwise_objects[i - 1]['gpu_learning_rate'],
                                                            self.layerwise_objects[i - 1]['gpu_momentum'],
                                                            self.layerwise_objects[i - 1]['gpu_obj_id'],
                                                            self.layerwise_objects[i - 1]['gpu_row_id'],
                                                            np.int32(self.layerwise_objects[i - 1]['total_threads']),
                                                            block=(self.layerwise_objects[i - 1]['block_size'], 1, 1),
                                                            grid=(self.layerwise_objects[i - 1]['grid'], 1))
                    if not propagate_only:
                        self.layerwise_objects[i - 1]['gpu_weight_mem'] += (self.layerwise_objects[i - 1]['gpu_mweight_mem'])
        self.flip_buf = not self.flip_buf

    def set_learning_rate(self, rate):
        print("Setting MLP learning rate to " + str(rate))
        for i in range(self.max_layers):
            self.layerwise_objects[i]['learning_rate_mem'][:] = rate
            self.layerwise_objects[i]['gpu_learning_rate'].set(self.layerwise_objects[i]['learning_rate_mem'])

    def drop_buf(self, myzip, filename, buf):
        print("Saving " + filename + " "*10  +"\r", end=' ')
        sys.stdout.flush()
        zipi = zipfile.ZipInfo()
        zipi.filename = filename
        zipi.date_time = time.localtime()[:6]
        zipi.compress_type = zipfile.ZIP_DEFLATED
        zipi.external_attr = 0o777 << 16
        myzip.writestr(zipi, buf)

    def save(self, outfile):
        self.get_gpu_mem()
        myzip = zipfile.ZipFile(outfile, "a", allowZip64=True)
        folder = "mlp"
        for k in list(self.__dict__.keys()):
            if k == "layerwise_objects":
                for layer in range(len(self.__dict__[k])):
                    for obj in list(self.__dict__[k][layer].keys()):
                        if not obj.startswith("gpu_"):
                            sys.stdout.flush()
                            if type(self.__dict__[k][layer][obj]) == np.ndarray:
                                self.drop_buf(myzip, folder + "/objects/" + obj + "_%02d.npy" % layer, memoryview(self.__dict__[k][layer][obj]))
                                metainfo = [self.__dict__[k][layer][obj].shape, self.__dict__[k][layer][obj].dtype]
                                GS = pickle.dumps(metainfo)
                                self.drop_buf(myzip, folder + "/objects/" + obj + "_%02d.npy.meta" % layer, GS)
                            elif "buf_" not in obj and "_buf" not in obj and not type(self.__dict__[k][layer][obj]) == list:
                                GS = pickle.dumps(self.__dict__[k][layer][obj])
                                self.drop_buf(myzip, folder + "/objects/" + obj + "_%02d.pkl" % layer, GS)
            else:
                 GS = pickle.dumps(self.__dict__[k])
                 self.drop_buf(myzip, folder+"/"+k+".pkl", GS)
        myzip.close()

    def load(self, filename):
        myzip = zipfile.ZipFile(filename, "r")
        for (i, element) in enumerate(myzip.namelist()):
            if element.startswith('mlp'):
                print(("[ %d/%d ] Extracting " % (i, len(myzip.namelist()))) + element + " "*10 + "\r", end=' ')
                sys.stdout.flush()
                if "/objects/" in element:
                    if element.endswith(".npy"):
                        handle = myzip.open(element)
                        metahandle = myzip.open(element+".meta")
                        metainfo = pickle.load(metahandle)
                        buf = handle.read()
                        npo = np.frombuffer(bytearray(buf), dtype=metainfo[1])
                        npo.reshape(metainfo[0])
                        npo.setflags(write=True)
                        obj_name = os.path.basename(element)[:-7]
                        obj_layer = int(os.path.basename(element)[-6:-4])
                        self.__dict__["layerwise_objects"][obj_layer][obj_name] = npo
                        print("Loaded numpy object " + obj_name + " in layer "+ str(obj_layer) + "\r", end=' ')
                        print("Dtype " + str(metainfo[1]) + " shape" + str(metainfo[0]) + " "*10 + "\r", end=' ')
                    elif element.endswith(".pkl"):
                        handle = myzip.open(element)
                        obj = pickle.load(handle)
                        obj_name = os.path.basename(element)[:-7]
                        obj_layer = int(os.path.basename(element)[-6:-4])
                        self.__dict__["layerwise_objects"][obj_layer][obj_name] = obj
                        print("Loaded object " + obj_name + " in layer " + str(obj_layer) + " "*10  + "\r", end=' ')
                else:
                    handle = myzip.open(element)
                    obj = pickle.load(handle)
                    obj_name = os.path.basename(element)[:-4]
                    self.__dict__[obj_name] = obj
                    print("Loaded object " + obj_name + " in layer " + " " * 10 + "\r", end=' ')
        self.generate_gpu_mem()
        if not "propagate_all_the_way" in list(self.__dict__.keys()):
            self.propagate_all_the_way = False
#        self.list_dweight_memory = [self.dweight_memory_0, self.dweight_memory_1]
#        self.list_weight_memory_cache = [self.weight_memory_cache0, self.weight_memory_cache1]
#        self.create_mem_gpu()
#        self.device = pycuda.driver.Device(0).name()



def create_specs():
    specs = []
    specs.append([50, 30, 40])
    specs.append([55, 35, 45])
    specs.append([45, 65, 35])
    M = MLP_collection(specs)
    M.generate_gpu_mem()
    for s in range(len(specs)):
        M.layerwise_objects[0]['input_obj'][s] += 0.5
    M.set_gpu_mem()
    M.get_gpu_mem()
    M.forward(poly=False)
    print((M.layerwise_objects[2]['input_obj']))
    M.forward_gpu(poly=False)
    M.get_gpu_mem()
    print((M.layerwise_objects[2]['input_obj']))
    for o in M.layerwise_objects[2]['error_obj']:
        o[:] += 0.5
    #for o in M.layerwise_objects[2]['delta_obj']:
    #    o[:] += 0.5
    print(M.layerwise_objects[2]['error_obj'])
    print("#" * 100)
    print((M.layerwise_objects[2]['delta_obj']))
    print((M.layerwise_objects[1]['delta_obj']))
    print((M.layerwise_objects[0]['delta_obj']))
    print("#" * 100)
    M.backward()
    print(M.layerwise_objects[2]['error_obj'])
    print(M.layerwise_objects[2]['input_obj'])
    print((M.layerwise_objects[2]['delta_obj']))
    print((M.layerwise_objects[1]['delta_obj']))
    print((M.layerwise_objects[0]['delta_obj']))


def print_mlp(M):
    print("#" * 100)
    print("Activ0", M.layerwise_objects[0]['input_obj'])
    print("Delta0", M.layerwise_objects[0]['delta_obj'])
    print("Error0", M.layerwise_objects[0]['error_obj'])
    print("Activ1", M.layerwise_objects[1]['input_obj'])
    print("Delta1", M.layerwise_objects[1]['delta_obj'])
    print("Error1", M.layerwise_objects[1]['error_obj'])
    print("Activ2", M.layerwise_objects[2]['input_obj'])
    print("Delta2", M.layerwise_objects[2]['delta_obj'])
    print("Error2", M.layerwise_objects[2]['error_obj'])
    print("Weight0", M.layerwise_objects[0]['weight_obj'])
    print("Weight1", M.layerwise_objects[1]['weight_obj'])


def test_single_MLP():
    specs = []
    specs.append([2, 4, 1])
    M = MLP_collection(specs)
    print(M.layerwise_objects[0]['input_obj'][0][-1])
    M.generate_gpu_mem()
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output = [[0], [1], [1], [0]]
    poly=False
    GPU=True
    M.set_gpu_mem()
    for step in range(100000):
        s = np.random.randint(0, 4)
        M.layerwise_objects[0]['input_obj'][0][:2] = inputs[s]

        if not GPU:
            M.forward()
            error = (output[s] - M.layerwise_objects[2]['input_obj'][0])
            M.layerwise_objects[2]['error_obj'][0][:] = error
            M.backward(poly=poly)
            print_mlp(M)

        if GPU:
            M.forward_gpu(poly=poly)
            error = (output[s] - M.layerwise_objects[2]['input_obj'][0])
            M.layerwise_objects[2]['error_obj'][0][:] = error
            M.backward_gpu(poly=poly)
            M.get_gpu_mem()

        if True and step % 100 == 0:
            print("#" * 80)
            print(M.layerwise_objects[0]['input_obj'][0][:])
            print(M.layerwise_objects[2]['input_obj'][0][:])
            print(output[s])
            print("Error", error)


def test_1000_MLPs():
    specs = []
    for i in range(1000):
        specs.append([2, 4, 1])

    M = MLP_collection(specs)
    M.generate_gpu_mem()
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output = [[0], [1], [1], [0]]
    poly = False
    GPU = False
    M.set_gpu_mem()
    for step in range(100000):
        s = np.random.randint(0, 4)
        for i in range(1000):
            M.layerwise_objects[0]['input_obj'][i][:2] = inputs[s]

        if not GPU:
            M.forward()
            gerror=0
            for i in range(1000):
                error = (output[s] - M.layerwise_objects[2]['input_obj'][i])
                M.layerwise_objects[2]['error_obj'][i][:] = error
                gerror+=error
            M.backward(poly=poly)

        if GPU:
            M.forward_gpu(poly=poly)
            gerror=0
            for i in range(1000):
                error = (output[s] - M.layerwise_objects[2]['input_obj'][i])
                M.layerwise_objects[2]['error_obj'][i][:] = error
                gerror+=error
            M.backward_gpu(poly=poly)
            M.get_gpu_mem()

        if True and step % 100 == 0:
            print("#" * 80)
            print("Error", gerror)

def load_video(path, dx, dy):
    cap = cv2.VideoCapture(os.path.expanduser(path))
    while not cap.isOpened():
        cap = cv2.VideoCapture(path)
        cv2.waitKey(1000)
        print("Wait for the header")

    frames = []
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            #cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            framer = cv2.resize(frame, dsize=(dx*5, dy*5), interpolation=cv2.INTER_CUBIC)
            framer = framer.astype(np.float32)/255
            frames.append(framer)
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(frames)==100:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
    return frames

def test_PVM_like_layer():
    specs = []
    specs1 = []
    dx = 20
    dy = 20
    DP = SyntheticDataProvider(dx=dx*5,dy=dy*5)
    adversarial=True
    for i in range(dx*dy):
        specs.append([150, 49, 150])
        specs1.append([150, 70, 2])
    frames = load_video("./IMG_4107.MOV", dx, dy)

    frames = []
    for i in range(100):
        frames.append(DP.get_next())
        DP.advance()

    cv2.waitKey(100)
    M = MLP_collection(specs, learning_rate=0.001)
    M1 = MLP_collection(specs, learning_rate=0.003)
    M.generate_gpu_mem()
    M1.generate_gpu_mem()
    poly = False
    GPU = True
    M.set_gpu_mem()
    pframe = np.zeros_like(frames[0])
    gpu_pframe = gpuarray.to_gpu(pframe)
    gpu_frame0 = gpuarray.to_gpu(frames[0])
    gpu_frame1 = gpuarray.to_gpu(frames[1])
    gpu_frame2 = gpuarray.to_gpu(frames[2])
    gpu_frame3 = gpuarray.to_gpu(frames[3])


    # gpu_prev_input
    t0 = time.time()
    for step in range(10000000):
        if not GPU:
            for x in range(dx):
                for y in range(dy):
                    i = x * dy + y
                    M.layerwise_objects[0]['input_obj'][i][:75] = frames[step % len(frames)][x*5:x*5+5, y*5:y*5+5].flatten()
                    M.layerwise_objects[0]['input_obj'][i][75:150] = frames[(step-1) % len(frames)][x * 5:x * 5 + 5, y * 5:y * 5 + 5].flatten()
        else:
            frame = frames[step % len(frames)]
            gpu_routines.gpu_dist_frame(gpu_frame0,
                                        M.layerwise_objects[0]['gpu_input_mem'],
                                        M.layerwise_objects[0]['gpu_input_ptr'],
                                        np.int32(dx * 5),
                                        np.int32(dy * 5),
                                        np.int32(dx),
                                        np.int32(dy),
                                        np.int32(5),
                                        np.int32(5),
                                        np.int32(0),
                                        np.int32(M.total_objects), block=(M.threads, 1, 1),
                                        grid=(M.grid_size, 1))
            gpu_routines.gpu_dist_frame(gpu_frame1,
                                        M.layerwise_objects[0]['gpu_input_mem'],
                                        M.layerwise_objects[0]['gpu_input_ptr'],
                                        np.int32(frame.shape[0]),
                                        np.int32(frame.shape[1]),
                                        np.int32(dx),
                                        np.int32(dy),
                                        np.int32(5),
                                        np.int32(5),
                                        np.int32(75),
                                        np.int32(M.total_objects), block=(M.threads, 1, 1),
                                        grid=(M.grid_size, 1))
            if adversarial:
                gpu_routines.gpu_dist_frame(gpu_frame0,
                                            M1.layerwise_objects[0]['gpu_input_mem'],
                                            M1.layerwise_objects[0]['gpu_input_ptr'],
                                            np.int32(dx * 5),
                                            np.int32(dy * 5),
                                            np.int32(dx),
                                            np.int32(dy),
                                            np.int32(5),
                                            np.int32(5),
                                            np.int32(0),
                                            np.int32(M1.total_objects), block=(M1.threads, 1, 1),
                                            grid=(M1.grid_size, 1))
                gpu_routines.gpu_dist_frame(gpu_frame1,
                                            M1.layerwise_objects[0]['gpu_input_mem'],
                                            M1.layerwise_objects[0]['gpu_input_ptr'],
                                            np.int32(frame.shape[0]),
                                            np.int32(frame.shape[1]),
                                            np.int32(dx),
                                            np.int32(dy),
                                            np.int32(5),
                                            np.int32(5),
                                            np.int32(75),
                                            np.int32(M1.total_objects), block=(M1.threads, 1, 1),
                                            grid=(M1.grid_size, 1))



            gpu_frame0.set_async(frames[(step + 1) % len(frames)])
            gpu_frame1.set_async(frames[(step + 2) % len(frames)])

        if not GPU:
            M.forward()
            gerror=0
            for x in range(dx):
                for y in range(dy):
                    i = x*dy + y
                    ou1 = frames[step % len(frames)][x*5:x*5+5, y*5:y*5+5].flatten()
                    ou2 = frames[(step + 1) % len(frames)][x*5:x*5+5, y*5:y*5+5].flatten()
                    output = np.concatenate((ou1, ou2))
                    error = (output - M.layerwise_objects[2]['input_obj'][i][:150])
                    M.layerwise_objects[2]['error_obj'][i][:150] = error
                    gerror += error
            M.backward(poly=poly)

        if GPU:
            M.forward_gpu(poly=poly)
            M1.forward_gpu(poly=poly)
            gerror=0
            gpu_routines.gpu_calc_error_frame(gpu_frame2,
                                              M.layerwise_objects[2]['gpu_input_mem'],
                                              M.layerwise_objects[2]['gpu_input_ptr'],
                                              M.layerwise_objects[2]['gpu_error_mem'],
                                              M.layerwise_objects[2]['gpu_error_ptr'],
                                              np.int32(frame.shape[0]),
                                              np.int32(frame.shape[1]),
                                              np.int32(dx),
                                              np.int32(dy),
                                              np.int32(5),
                                              np.int32(5),
                                              np.int32(0),
                                              np.int32(M.total_objects),
                                              block=(M.threads, 1, 1),
                                              grid=(M.grid_size, 1))
            gpu_routines.gpu_calc_error_frame(gpu_frame3,
                                              M.layerwise_objects[2]['gpu_input_mem'],
                                              M.layerwise_objects[2]['gpu_input_ptr'],
                                              M.layerwise_objects[2]['gpu_error_mem'],
                                              M.layerwise_objects[2]['gpu_error_ptr'],
                                              np.int32(frame.shape[0]),
                                              np.int32(frame.shape[1]),
                                              np.int32(dx),
                                              np.int32(dy),
                                              np.int32(5),
                                              np.int32(5),
                                              np.int32(75),
                                              np.int32(M.total_objects),
                                              block=(M.threads, 1, 1),
                                              grid=(M.grid_size, 1))
            if adversarial:
                # Associate real input with 0-hot
                gpu_routines.gpu_set_one_hot_error(M1.layerwise_objects[2]['gpu_input_mem'],
                                                   M1.layerwise_objects[2]['gpu_input_ptr'],
                                                   M1.layerwise_objects[2]['gpu_error_mem'],
                                                   M1.layerwise_objects[2]['gpu_error_ptr'],
                                                   np.int32(0),
                                                   np.int32(2),
                                                   np.int32(M1.total_objects),
                                                   block=(M.threads, 1, 1),
                                                   grid=(M.grid_size, 1))
                #  Train real input
                M1.backward_gpu(poly=poly)
                # Copy the fake prediction
                gpu_routines.gpu_copy_blocks_fixed(M.layerwise_objects[2]['gpu_input_mem'],
                                                   M1.layerwise_objects[0]['gpu_input_mem'],
                                                   M.layerwise_objects[0]['gpu_input_ptr'],
                                                   M1.layerwise_objects[0]['gpu_input_ptr'],
                                                   np.int32(150),
                                                   np.float32(1.0),
                                                   np.int32(M.total_objects),
                                                   block=(M.threads, 1, 1),
                                                   grid=(M.grid_size, 1)
                                                   )
                # Set the output to fake
                gpu_routines.gpu_set_one_hot_error(M1.layerwise_objects[2]['gpu_input_mem'],
                                                   M1.layerwise_objects[2]['gpu_input_ptr'],
                                                   M1.layerwise_objects[2]['gpu_error_mem'],
                                                   M1.layerwise_objects[2]['gpu_error_ptr'],
                                                   np.int32(1),
                                                   np.int32(2),
                                                   np.int32(M1.total_objects),
                                                   block=(M.threads, 1, 1),
                                                   grid=(M.grid_size, 1))
                # Train the fake
                M1.backward_gpu(poly=poly)
                # Set the output to true again
                gpu_routines.gpu_set_one_hot_error(M1.layerwise_objects[2]['gpu_input_mem'],
                                                   M1.layerwise_objects[2]['gpu_input_ptr'],
                                                   M1.layerwise_objects[2]['gpu_error_mem'],
                                                   M1.layerwise_objects[2]['gpu_error_ptr'],
                                                   np.int32(0),
                                                   np.int32(2),
                                                   np.int32(M1.total_objects),
                                                   block=(M.threads, 1, 1),
                                                   grid=(M.grid_size, 1))
                M1.backward_gpu(poly=poly, propagate_all_way=True, propagate_only=True)

            gpu_frame2.set_async(frames[(step + 3) % len(frames)])
            gpu_frame3.set_async(frames[(step + 4) % len(frames)])
        else:
            for x in range(dx):
                for y in range(dy):
                    i = x*dy + y
                    ou1 = frames[step % len(frames)][x*5:x*5+5, y*5:y*5+5].flatten()
                    ou2 = frames[(step + 1) % len(frames)][x*5:x*5+5, y*5:y*5+5].flatten()
                    output = np.concatenate((ou1, ou2))
                    error = (output - M.layerwise_objects[2]['input_obj'][i][:150])
                    M.layerwise_objects[2]['error_obj'][i][:150] = error
                    gerror += error


        M.backward_gpu(poly=poly)
        if adversarial:
            #print M1.layerwise_objects[0]['gpu_error_mem'].get()
            # Copy the fake prediction
            gpu_routines.gpu_copy_blocks_fixed(M1.layerwise_objects[0]['gpu_error_mem'],
                                               M.layerwise_objects[2]['gpu_error_mem'],
                                               M1.layerwise_objects[0]['gpu_error_ptr'],
                                               M.layerwise_objects[2]['gpu_error_ptr'],
                                               np.int32(150),
                                               np.float32(100),
                                               np.int32(M.total_objects),
                                               block=(M.threads, 1, 1),
                                               grid=(M.grid_size, 1)
                                               )
            M.backward_gpu(poly=poly)


        if True: # and step % 100 == 50:
            t1 = time.time()
            print("Step %d, %2.2f fps " % (step, step/(t1-t0)))
            if GPU:
                gpu_routines.gpu_collect_frame(gpu_pframe,
                                               M.layerwise_objects[2]['gpu_input_mem'],
                                               M.layerwise_objects[0]['gpu_input_ptr'],
                                               np.int32(dx * 5),
                                               np.int32(dy * 5),
                                               np.int32(dx),
                                               np.int32(dy),
                                               np.int32(5),
                                               np.int32(5),
                                               np.int32(0),
                                               np.int32(M.total_objects), block=(M.threads, 1, 1),
                                               grid=(M.grid_size, 1))
                aframe = (gpu_pframe.get() * 255).astype(np.uint8)
                if step % 10000 == 50:
                    if adversarial:
                        cv2.imwrite("adv_prediction_%05d.png" % step, aframe)
                    else:
                        cv2.imwrite("1prediction_%05d.png" % step, aframe)

            else:
                for x in range(dx):
                    for y in range(dy):
                        i = x * dy + y
                        pframe[x * 5:x * 5 + 5, y * 5:y * 5 + 5] = M.layerwise_objects[2]['input_obj'][i][:75].reshape(5, 5, 3)
                aframe = (pframe*255).astype(np.uint8)
            cv2.imshow("Predicted", aframe)
            cv2.imshow("Frame", (frames[step % len(frames)]*255).astype(np.uint8))
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
    pycuda.driver.stop_profiler()
    cv2.waitKey(0)

def create_pvm(specs):
    layer_shapes = specs['layer_shapes']
    total_units = 0


if __name__ == "__main__":
    pycuda.tools.clear_context_caches()
    test_PVM_like_layer()
    #test_1000_MLPs()
    #test_single_MLP()
    #create_specs()
