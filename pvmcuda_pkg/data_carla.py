# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info


import cv2
import numpy as np
import os
import zipfile
import sys
from .data import DataProvider
from .utils import VideoRecorder

class CarlaVidSingleDataProvider(DataProvider):
    has_depth = True
    classes = [
        [[0, 0, 0],   "None"],
        [[70, 70, 70],  "Buildings"],
        [[190, 153, 153],  "Fences"],
        [[72, 0, 90],  "Other"],
        [[220, 20, 60],  "Pedestrians"],
        [[153, 153, 153],  "Poles"],
        [[157, 234, 50],  "RoadLines"],
        [[128, 64, 128],  "Roads"],
        [[244, 35, 232],  "Sidewalks"],
        [[107, 142, 35],  "Vegetation"],
        [[0, 0, 255],  "Vehicles"],
        [[102, 102, 156],  "Walls"],
        [[220, 220, 0],  "TrafficSigns"],
    ]

    segnet_classes = [
        [[128, 128, 128], "Sky"],
        [[128, 0, 0], "Building"],
        [[192,192, 128], "Pole"],
        [[255, 69, 0], "Road Marking"],
        [[128, 64, 128], "Road"],
        [[60, 40, 222], "Pavement"],
        [[128, 128, 0], "Tree"],
        [[192, 128, 128], "Sign Symbol"],
        [[64, 64, 128], "Fence"],
        [[64, 0, 128], "Vehicle"],
        [[64, 64, 0], "Pedestrian"],
        [[0, 128, 192], "Bike"],
        [[0, 0, 0], "Void"]
    ]

    camvid_to_segnet_map = {
        "Road": "Road",
        "RoadShoulder": "Road",
        "Sky": "Sky",
        "Sidewalk": "Pavement",
        "Pedestrian": "Pedestrian",
        "Archway": "Sky",
        "Car": "Vehicle",
        "Tree": "Tree",
        "VegetationMisc": "Tree",
        "Truck_Bus": "Vehicle",
        "Train": "Vehicle",
        "Wall": "Building",
        "Fence": "Fence",
        "CartLuggagePram": "Vehicle",
        "MotorcycleScooter": "Vehicle",
        "OtherMoving" : "Vehicle",
        "Child": "Pedestrian",
        "Column_Pole": "Pole",
        "SignSymbol": "Sign Symbol",
        "TrafficLight": "Sign Symbol",
        "Bicyclist": "Bike",
        "LaneMkgsDriv": "Road Marking",
        "LaneMkgsNonDriv": "Road Marking",
        "Misc_Text": "Sign Symbol",
        "SUVPickupTruck": "Vehicle",
        "TrafficCone": "Sign Symbol",
        "Bridge": "Building",
        "Tunnel": "Building",
        "ParkingBlock": "Pavement",
        "Void": "Void",
        "Animal": "Void",
        "Building": "Building"
    }

    def __init__(self, path, dx, dy, blocks_x=None, blocks_y=None, block_size=6, max_length=10000000000, use_segnet=False):
        self.file_path = os.path.expanduser(path)
        self.use_segnet = use_segnet
        myzip = zipfile.ZipFile(os.path.expanduser(self.file_path), "r")
        self.frames_dict = {}
        self.depth_dict = {}
        if use_segnet:
            self.create_camvid_segnet_mapping()
            self.classes_arr = np.array([x[0] for x in self.segnet_classes])
        else:
            self.classes_arr = np.array([x[0] for x in self.classes])
        self.filenames = []
        self.labels_dict = {}
        self.pos_tran = []
        self.blocks_x = blocks_x
        self.blocks_y = blocks_y
        self.block_size = block_size
        self.labels_cnt = 0
        self.shape = (300, 300)
        for (i, element) in enumerate(myzip.namelist()):
            if i % 30 == 0:
                print(("[ %d/%d ] Extracting " % (i, len(myzip.namelist()))) + element + " "*10 + "\r", end=' ')
            sys.stdout.flush()
            if element.endswith("png"):
                handle = myzip.open(element)
                buf = handle.read()
                img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
                if "view/" in element:
                    nr = int(os.path.basename(element)[-9:-4])
                    framer = cv2.resize(img, dsize=(dx, dy), interpolation=cv2.INTER_CUBIC)
                    framer = framer.astype(np.float32) / 255
                    self.frames_dict[nr] = framer
                    self.filenames.append(os.path.basename(element))
                    self.pos_tran.append(nr)
                if "depth/" in element:
                    nr = int(os.path.basename(element)[-9:-4])
                    framer = cv2.resize(img, dsize=(dx, dy), interpolation=cv2.INTER_CUBIC)
                    framer = cv2.cvtColor(framer, cv2.COLOR_BGR2GRAY)
                    framer = framer.astype(np.float32) / 255
                    self.depth_dict[nr] = framer
                    self.pos_tran.append(nr)
                if "label/" in element:
                    nr = int(os.path.basename(element)[-9:-4])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = self.transform_label(img)
                    if self.blocks_x is not None and self.blocks_y is not None:
                        img = cv2.resize(img, dsize=(self.blocks_y, self.blocks_x), interpolation=cv2.INTER_NEAREST)
                        img = self.encode_label(img)
                    self.labels_dict[nr] = img
                    self.labels_cnt += 1
                    self.last_valid_label = self.labels_dict[nr]
        self.pos_tran = sorted(self.pos_tran)
        h, w, _ = self.frames_dict[self.pos_tran[0]].shape
        if dx is not None and dy is not None:
            self.shape = (dx, dy)
        else:
            self.shape = self.frames_dict[self.pos_tran[0]].shape
        self.pos = 0
        myzip.close()

    def create_camvid_segnet_mapping(self):
        self.translate = {}
        for camvid_cl in self.classes:
            name = camvid_cl[1]
            val = camvid_cl[0]
            segnet_name = self.camvid_to_segnet_map[name]
            seg_val = None
            for seg_class in self.segnet_classes:
                if seg_class[1] == segnet_name:
                    seg_val = seg_class[0]
                    break
            self.translate[name] = seg_val

    def transform_label(self, label):
        if self.use_segnet:
            for camvid_cl in self.classes:
                name = camvid_cl[1]
                val = camvid_cl[0]
                label[np.where((label == val).all(axis=2))] = self.translate[name]
        return label

    def get_next(self):
        ret = self.frames_dict[self.pos_tran[self.pos]]
        return ret

    def get_depth(self):
        ret = self.depth_dict[self.pos_tran[self.pos]]
        return ret

    def get_label(self, last_valid=False):
        if self.pos_tran[self.pos] in list(self.labels_dict.keys()):
            self.last_valid_label = self.labels_dict[self.pos_tran[self.pos]]
            return self.labels_dict[self.pos_tran[self.pos]]
        elif last_valid == True:
            return self.last_valid_label
        else:
            return None

    def __len__(self):
        return len(self.pos_tran)

    def advance(self):
        self.pos = (self.pos + 1) % len(self.pos_tran)

    def describe(self):
        return "Zipped labeled Carla frames %s, %d frames, %d labels" % (self.file_path, len(self.pos_tran), self.labels_cnt)

    def encode_label(self, img):
        label = np.zeros((self.blocks_y * self.block_size, self.blocks_x * self.block_size),
                         dtype=np.float32)
        self.cl = self.classes
        if self.use_segnet:
            self.cl = self.segnet_classes
        for (i, c) in enumerate(self.cl):
            xx = i % self.block_size
            yy = i / self.block_size
            label[xx::self.block_size, yy::self.block_size] = np.all(img == c[0], axis=-1).astype(np.float)
        return label


    def decode_label(self, label, blocks_x=None, blocks_y=None, shape=None):
        """
        This label decoding is faster and better then the naive above, since
        it always picks the maximum value in each block
        :param label:
        :param blocks_x:
        :param blocks_y:
        :return:
        """
        if blocks_x is None:
            blocks_x = self.blocks_x
        if blocks_y is None:
            blocks_y = self.blocks_y
        B = data.blockshaped(label, self.block_size, self.block_size).transpose([0, 2, 1])
        C = B.reshape(B.shape[0], B.shape[1]*B.shape[2])
        amx = np.argmax(C, axis=1)
        amx[np.where(amx >= self.classes_arr.shape[0])] = 0
        mx = C[np.indices(amx.shape), amx].flatten()
        decoded1 = self.classes_arr[amx]
        decoded1[np.where(mx < 0.1)] = [0, 0, 0]
        decoded = decoded1.reshape((blocks_y, blocks_x, 3))
        if shape is None:
            shape=self.shape
        return cv2.resize(decoded, dsize=shape, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    def get_filename(self):
        return self.filenames[self.pos]


def threadwrap(func,args,kwargs):
   class res(object): result=None
   def inner(*args,**kwargs):
     res.result=func(*args,**kwargs)
   import threading
   t = threading.Thread(target=inner, args=args, kwargs=kwargs)
   res.thread=t
   return res

def myFun(path, dx, dy, blocks_x=None, blocks_y=None, block_size=6):
  return CarlaVidSingleDataProvider(path, dx=dx, dy=dy, blocks_x=blocks_x, blocks_y=blocks_y, block_size=block_size)

#from guppy import hpy
import gc

class CarlaVideoProvider(DataProvider):
    def __init__(self, path_collection, dx, dy, blocks_x=None, blocks_y=None, block_size=6, max_length=10000000000, augment=0):
        self.pos = 0
        self.has_depth = True
        self.current_sequence = 0
        self.len = 0
        self.augment = augment
        self.dx_per_b = dx / blocks_x
        self.dy_per_b = dy / blocks_y
        self.actual_dx = dx+self.augment * self.dx_per_b
        self.actual_dy = dy+self.augment * self.dy_per_b
        self.dx = dx
        self.dy = dy
        self.blocks_x = blocks_x
        self.blocks_y = blocks_y
        self.block_size = block_size
        self.path_collection = path_collection
        self.sequences = path_collection
        if augment > 0:
            self.augment_vec = np.random.randint(0, self.augment, (2,))
        else:
            self.augment_vec = (0, 0)
        self.sequence = CarlaVidSingleDataProvider(path=self.path_collection[0], dx=self.actual_dx, dy=self.actual_dy, blocks_x=blocks_x+self.augment, blocks_y=blocks_y+self.augment, block_size=block_size)
        self.len = len(self.sequence)
        next_seq = self.path_collection[np.random.randint(0, len(self.path_collection)+1) % len(self.path_collection)]
        self.thread = threadwrap(myFun, [], {"path": next_seq, "dx": self.actual_dx, "dy": self.actual_dy, "blocks_x" :blocks_x+self.augment, "blocks_y": blocks_y+self.augment, "block_size": block_size})
        self.thread.thread.start()

    def __len__(self):
        return self.len

    def get_next(self):
        frame = self.sequence.get_next()
        if self.augment > 0:
            frame = frame[self.augment_vec[0]*self.dx_per_b:self.augment_vec[0]*self.dx_per_b+self.dx,
                          self.augment_vec[1]*self.dy_per_b:self.augment_vec[1]*self.dy_per_b+self.dy]
        return frame

    def get_label(self, last_valid=False):
        label = self.sequence.get_label(last_valid=last_valid)
        if self.augment > 0 and label is not None:
            label = label[self.augment_vec[0]*self.block_size:(self.augment_vec[0] + self.blocks_x)*self.block_size,
                          self.augment_vec[1]*self.block_size:(self.augment_vec[1] + self.blocks_y)*self.block_size]
        return label

    def get_depth(self):
        return self.sequence.get_depth()

    def advance(self):
        self.pos = (self.pos + 1) % self.len
        if self.sequence.pos + 1 == len(self.sequence):
            self.thread.thread.join()
            self.sequence = self.thread.result
            self.len = len(self.sequence)
            next_seq = self.path_collection[np.random.randint(0, len(self.path_collection)+1) % len(self.path_collection)]
            self.thread = threadwrap(myFun, [], {"path": next_seq, "dx": self.actual_dx, "dy": self.actual_dy,
                                                 "blocks_x": self.blocks_x + self.augment,
                                                 "blocks_y": self.blocks_y + self.augment, "block_size": self.block_size})
            self.thread.thread.start()

            if self.augment > 0:
                self.augment_vec = np.random.randint(0, self.augment, (2,))
        else:
            self.sequence.advance()

    def describe(self):
        desc = "Collection of frames: \n"
        desc += self.sequence.describe() + "\n"
        desc += "%d frames together" % self.len
        return desc

    def decode_label(self, label):
        # TODO:
        # This needs to be modified to reflect the right shape with augmentation
        return self.sequence.decode_label(label, blocks_x=self.blocks_x, blocks_y=self.blocks_y, shape=(self.dx, self.dy))

    def reset_pos(self):
        self.sequence.reset_pos()

    def get_classes(self):
        return self.sequence.get_classes()

if __name__ == "__main__":
    C = CarlaVideoProvider([os.path.expanduser("~/carla_data/calra_001_%03d.zip" % i) for i in range(5)], dx=320, dy=200, blocks_x=64, blocks_y=64)
    for i in range(100000):
        img = C.get_next()
        dep = C.get_depth()
        lab = cv2.cvtColor(C.decode_label(C.get_label()), cv2.COLOR_BGR2RGB)
        C.advance()
        # cv2.imshow("Win", img)
        # cv2.imshow("Depth", dep)
        # cv2.imshow("label", lab)
        # d1 = cv2.cvtColor(dep, cv2.COLOR_GRAY2BGR)
        dep = cv2.cvtColor(dep, cv2.COLOR_GRAY2BGR)
        i1 = np.hstack((img, dep))
        i2 = np.hstack(((i1 * 255).astype(np.uint8), lab))
        # video_recorder.record(i2)
        print(data.IOU(lab, np.roll(lab, 10, axis=1), [128, 64, 128]))
        cv2.imshow("All", i2)
        cv2.waitKey(5)

    h = hpy()
    video_recorder = VideoRecorder(rec_filename=os.path.expanduser("~/carla.avi"), fps=30)
    Z = CarlaVidSingleDataProvider(os.path.expanduser("~/calra_001_%03d.zip" % 0), dx=320, dy=240)
    for i in range(120):

        Z1 = threadwrap(myFun, [os.path.expanduser("~/carla_00_%03d.zip" % ((i + 1) % 13))], {"dx": 320, "dy": 240})
        Z1.thread.start()
        for i in range(len(Z)):
            img = Z.get_next()
            dep = Z.get_depth()
            lab = cv2.cvtColor(Z.get_label(), cv2.COLOR_BGR2RGB)
            Z.advance()
            #cv2.imshow("Win", img)
            #cv2.imshow("Depth", dep)
            #cv2.imshow("label", lab)
            #d1 = cv2.cvtColor(dep, cv2.COLOR_GRAY2BGR)
            i1 = np.hstack((img, dep[:, :, np.newaxis]))
            i2 = np.hstack(((i1*255).astype(np.uint8), lab))
            # video_recorder.record(i2)
            cv2.imshow("All", i2)
            cv2.waitKey(5)
        Z1.thread.join()
        del Z
        Z = Z1.result
        del Z1
        gc.collect()
