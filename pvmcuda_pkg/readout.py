# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info
import numpy as np
import pvmcuda_pkg.gpu_mlp as gpu_mlp
import pvmcuda_pkg.gpu_routines as gpu_routines
import pycuda.gpuarray as gpuarray
import sys
import zipfile
import time
import pickle
import os
import pvmcuda_pkg.utils as utils


class Readout(object):
    def __init__(self, PVMObject = None, representation_size=100, heatmap_block_size=None):
        if PVMObject is not None:
            self.readout_layer = 2
            if heatmap_block_size is not None:
                self.heatmap_block_size = heatmap_block_size
            else:
                self.heatmap_block_size = int(PVMObject.specs['input_block_size'])
            self.blocks_x = int(PVMObject.specs['layer_shapes'][0])
            self.blocks_y = int(PVMObject.specs['layer_shapes'][0])
            self.PVMObject = PVMObject
            self.shape = int(self.blocks_x*self.heatmap_block_size)
            sizes = np.zeros(int(PVMObject.specs['layer_shapes'][0]) * int(PVMObject.specs['layer_shapes'][0]), dtype=np.int)
            blocks = 0
            i = 0
            for x in range(int(PVMObject.specs['layer_shapes'][0])):
                for y in range(int(PVMObject.specs['layer_shapes'][0])):
                    for block in PVMObject.graph:
                        if x in block['xs'] and y in block['ys']:
                            sizes[i] += block['size']
                            blocks += 1
                    i += 1
            self.total_units = int(PVMObject.specs['layer_shapes'][0])**2
            self.ptrs_from = np.zeros(blocks, dtype=np.int32)
            self.ptrs_to = np.zeros(blocks, dtype=np.int32)
            self.qnt_from = np.zeros(blocks, dtype=np.int32)
            self.sizes = sizes
            i = 0
            blocks = 0
            running_ptr = 0
            mlp_specs = []
            for x in range(int(PVMObject.specs['layer_shapes'][0])):
                for y in range(int(PVMObject.specs['layer_shapes'][0])):
                    for block in PVMObject.graph:
                        if x in block['xs'] and y in block['ys']:
                            self.ptrs_from[blocks] = PVMObject.repr_ptr[block['id']]
                            self.ptrs_to[blocks] = running_ptr
                            self.qnt_from[blocks] = block['size']
                            running_ptr += block['size']
                            blocks += 1
                    running_ptr += 1 # For bias unit
                    if utils.check_if_enabled("4_layer_readout", self.PVMObject.specs):
                        mlp_specs.append([sizes[i], representation_size, representation_size, self.heatmap_block_size**2])
                        self.readout_layer = 3
                    else:
                        mlp_specs.append([sizes[i], representation_size, self.heatmap_block_size**2])
                    i += 1
            learning_rate = float(0.01)
            momentum = float(self.PVMObject.specs["momentum"])
            self.mlp = gpu_mlp.MLP_collection(specs=mlp_specs, learning_rate=learning_rate, momentum=momentum)
            self.mlp.generate_gpu_mem()
            self.mlp.set_gpu_mem()
            self.total_blocks = blocks
            self.create_gpu_mem()

    def create_gpu_mem(self):
        self.ptrs_from_gpu = gpuarray.to_gpu(self.ptrs_from)
        self.ptrs_to_gpu = gpuarray.to_gpu(self.ptrs_to)
        self.sizes_gpu = gpuarray.to_gpu(self.qnt_from)

    def copy_data(self):
        gpu_routines.gpu_copy_blocks(self.PVMObject.repre_mem_activation_gpu[(self.PVMObject.step) % self.PVMObject.sequence_length],
                                     self.mlp.layerwise_objects[0]['gpu_input_mem'],
                                     self.ptrs_from_gpu,
                                     self.sizes_gpu,
                                     self.ptrs_to_gpu,
                                     np.int32(self.total_blocks),
                                     block=(min(self.total_blocks, 128), 1, 1),
                                     grid=(self.total_blocks // 128 + 1, 1))

    def forward(self):
        self.mlp.forward_gpu()

    def train(self, label):
        label_gpu = gpuarray.to_gpu(label)
        if utils.check_if_enabled("opt_abs_diff", self.PVMObject.specs):
            gpu_routines.gpu_calc_abs_diff_error_frame_1ch(label_gpu,
                                                           self.mlp.layerwise_objects[self.readout_layer]['gpu_input_mem'],
                                                           self.mlp.layerwise_objects[self.readout_layer]['gpu_input_ptr'],
                                                           self.mlp.layerwise_objects[self.readout_layer]['gpu_error_mem'],
                                                           self.mlp.layerwise_objects[self.readout_layer]['gpu_error_ptr'],
                                                           np.int32(label.shape[0]),
                                                           np.int32(label.shape[1]),
                                                           np.int32(int(self.PVMObject.specs['layer_shapes'][0])),
                                                           np.int32(int(self.PVMObject.specs['layer_shapes'][0])),
                                                           np.int32(self.heatmap_block_size),
                                                           np.int32(self.heatmap_block_size),
                                                           np.int32(0),
                                                           np.int32(self.total_units),
                                                           block=(min(self.total_units, 128), 1, 1),
                                                           grid=(self.total_units // 128 + 1, 1))
        else:
            gpu_routines.gpu_calc_error_frame_1ch(label_gpu,
                                                  self.mlp.layerwise_objects[self.readout_layer]['gpu_input_mem'],
                                                  self.mlp.layerwise_objects[self.readout_layer]['gpu_input_ptr'],
                                                  self.mlp.layerwise_objects[self.readout_layer]['gpu_error_mem'],
                                                  self.mlp.layerwise_objects[self.readout_layer]['gpu_error_ptr'],
                                                  np.int32(label.shape[0]),
                                                  np.int32(label.shape[1]),
                                                  np.int32(int(self.PVMObject.specs['layer_shapes'][0])),
                                                  np.int32(int(self.PVMObject.specs['layer_shapes'][0])),
                                                  np.int32(self.heatmap_block_size),
                                                  np.int32(self.heatmap_block_size),
                                                  np.int32(0),
                                                  np.int32(self.total_units),
                                                  block=(min(self.total_units, 128), 1, 1),
                                                  grid=(self.total_units // 128 + 1, 1))
        self.mlp.backward_gpu()

    def get_heatmap(self):
        frame = np.zeros((self.shape, self.shape), dtype=np.float32)
        gpu_pframe = gpuarray.to_gpu(frame)
        gpu_routines.gpu_collect_frame_1ch(gpu_pframe,
                                           self.mlp.layerwise_objects[self.readout_layer]['gpu_input_mem'],
                                           self.mlp.layerwise_objects[self.readout_layer]['gpu_input_ptr'],
                                           np.int32(self.shape),
                                           np.int32(self.shape),
                                           np.int32(int(self.PVMObject.specs['layer_shapes'][0])),
                                           np.int32(int(self.PVMObject.specs['layer_shapes'][0])),
                                           np.int32(self.heatmap_block_size),
                                           np.int32(self.heatmap_block_size),
                                           np.int32(0),
                                           np.int32(self.total_units),
                                           block=(min(self.total_units, 128), 1, 1),
                                           grid=(self.total_units // 128 + 1, 1))
        gpu_pframe.get(frame)
        return frame

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
        myzip = zipfile.ZipFile(outfile, "a", allowZip64=True)
        folder = "readout"
        for k in list(self.__dict__.keys()):
            if not k.endswith("_gpu") and k != "mlp" and k!="PVMObject":
                if type(self.__dict__[k]) == np.ndarray:
                    self.drop_buf(myzip, folder+"/"+k+".npy", memoryview(self.__dict__[k]))
                    metainfo = [self.__dict__[k].shape, self.__dict__[k].dtype]
                    GS = pickle.dumps(metainfo)
                    self.drop_buf(myzip, folder + "/" + k + ".npy.meta", GS)
                elif "_gpu" not in k and not k.startswith("list_"):
                    GS = pickle.dumps(self.__dict__[k])
                    self.drop_buf(myzip, folder+"/"+k+".pkl", GS)
        myzip.close()
        self.mlp.save(outfile)

    def load(self, filename):
        myzip = zipfile.ZipFile(filename, "r")
        loaded = False
        self.readout_layer = 2
        for (i, element) in enumerate(myzip.namelist()):
            if element.startswith('readout'):
                loaded = True
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
        myzip.close()
        if not loaded:
            return False
        if "heatmap_block_size" not in list(self.__dict__.keys()):
            self.heatmap_block_size = None
        self.mlp = gpu_mlp.MLP_collection()
        self.mlp.load(filename)
        self.mlp.generate_gpu_mem()
        self.mlp.set_gpu_mem()
        self.create_gpu_mem()
        return True

    def set_pvm(self, PVMObject):
        self.PVMObject = PVMObject
        if self.heatmap_block_size is None:
            self.heatmap_block_size = int(PVMObject.specs['input_block_size'])
        if "readout_multiplier" in list(self.PVMObject.specs.keys()):
            mul = float(self.PVMObject.specs["readout_multiplier"])
        else:
            mul = 10
        self.mlp.set_learning_rate(mul * float(self.PVMObject.learning_rate))

    def update_learning_rate(self, override_rate=None):
        if "readout_multiplier" in list(self.PVMObject.specs.keys()):
            mul = float(self.PVMObject.specs["readout_multiplier"])
        else:
            mul = 10
        if self.PVMObject.step == int(self.PVMObject.specs['delay_final_learning_rate']):
            self.mlp.set_learning_rate(mul * float(self.PVMObject.specs['final_learning_rate']))
            print("Setting final readout learning rate")
        if 'delay_intermediate_learning_rate' in list(self.PVMObject.specs.keys()) and self.PVMObject.step == int(self.PVMObject.specs['delay_intermediate_learning_rate']):
            self.mlp.set_learning_rate(mul * float(self.PVMObject.specs['intermediate_learning_rate']))
            print("Setting intermediate readout learning rate")
        if override_rate is not None:
            self.mlp.set_learning_rate(override_rate)
            print("Overriding Readout learning rate to " + str(override_rate))


