# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info
import cv2
import numpy as np
import os
import zipfile
import sys


class DataProvider(object):
    _attr = {}
    has_depth = False
    def __init__(self):
        raise NotImplementedError

    def get_next(self):
        raise NotImplementedError

    def get_label(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def advance(self):
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError

    def decode_label(self, label):
        return label

    def set_attr(self, attr, value):
        self._attr[attr] = value

    def get_attr(self, attr):
        if attr in self._attr:
            return self._attr[attr]
        else:
            return None

    def get_filename(self):
        return None

    def get_pos(self):
        return self.pos

    def reset_pos(self):
        self.pos = 0

    def get_classes(self):
        return self.classes


class MovieDataProvider(DataProvider):
    def __init__(self, path, dx, dy, max_length=10000000000):
        self.pos = 0
        self.file_path = os.path.expanduser(path)
        cap = cv2.VideoCapture(self.file_path)
        while not cap.isOpened():
            cap = cv2.VideoCapture(path)
            cv2.waitKey(1000)
            print("Wait for the header")
        self.frames = []
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = cap.read()
            if flag:
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                framer = cv2.resize(frame, dsize=(dx, dy), interpolation=cv2.INTER_CUBIC)
                framer = cv2.flip(framer, 0)
                framer = framer.astype(np.float32) / 255
                self.frames.append(framer)
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT) or len(self.frames) == max_length:
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break

    def get_next(self):
        ret = self.frames[self.pos]
        return ret

    def get_label(self, last_valid=False):
        return None

    def __len__(self):
        return len(self.frames)

    def advance(self):
        self.pos = (self.pos + 1) % len(self.frames)

    def describe(self):
        return "Movie file %s, %d frames, unlabeled" % (self.file_path, len(self.frames))



# Use the class below to load old PVM data.
# Fist convert the data by using pvm's export to zip like below:
# for file in ~/PVM_data/PVM_data/* ; do python export_to_zip.py  -i $file -o ~/old_PVMdata/`basename $file`.zip; done;
# Next you can read those zips
class ZipDataProvider(DataProvider):
    """


    """
    def __init__(self, path, dx=None, dy=None, max_length=10000000000):
        self.file_path = os.path.expanduser(path)
        myzip = zipfile.ZipFile(os.path.expanduser(self.file_path), "r")
        self.frames_dict = {}
        for (i, element) in enumerate(myzip.namelist()):
            print(("[ %d/%d ] Extracting " % (i, len(myzip.namelist()))) + element + " "*10 + "\r", end=' ')
            sys.stdout.flush()
            if element.endswith("jpg"):
                handle = myzip.open(element)
                buf = handle.read()
                img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
                nr = int(os.path.basename(element)[:4])
                self.frames_dict[nr] = img
            if element.endswith("txt"):
                handle = myzip.open(element)
                labels = handle.readlines()
        h, w, _ = self.frames_dict[0].shape
        if dx is not None and dy is not None:
            self.shape = (dx, dy)
        else:
            self.shape = self.frames_dict[0].shape
            dx = w
            dy = h
        factor_h = dy / float(h)
        factor_w = dx / float(w)
        self.frames = []
        self.labels = []
        for i in range(len(self.frames_dict)):
            frame = self.frames_dict[i]
            framer = cv2.resize(frame, dsize=(dx, dy), interpolation=cv2.INTER_CUBIC)
            framer = framer.astype(np.float32) / 255
            self.frames.append(framer)
            label = [int(x) for x in labels[i].split(b',')]
            if label[0] == -1:
                self.labels.append(label)
            else:
                self.labels.append([int(label[0]*factor_w), int(label[1]*factor_h), int(label[2]*factor_w), int(label[3]*factor_h)])
        self.pos = 0

    def get_next(self):
        ret = self.frames[self.pos]
        return ret

    def get_label(self, last_valid=False):
        ret = np.zeros(self.shape, dtype=np.float32)
        label = self.labels[self.pos]
        if label[0] < 0:
            return ret
        cv2.rectangle(ret, pt1=tuple(label[0:2]), pt2=tuple(np.array(label[0:2])+np.array(label[2:4])), color=1.0, thickness=-1)
        return ret

    def __len__(self):
        return len(self.frames)

    def advance(self):
        self.pos = (self.pos + 1) % len(self.frames)

    def describe(self):
        return "Zipped labeled frames %s, %d frames" % (self.file_path, len(self.frames))

    def decode_label(self, label):
        return label


class ZipCollectionDataProvider(ZipDataProvider):
    def __init__(self, path_collection, dx, dy, max_length=10000000000):
        self.frames = []
        self.labels = []
        self.descriptions = []
        self.shape = (dx, dy)
        for path in path_collection:
            Z = ZipDataProvider(path, dx, dy)
            self.frames.extend(Z.frames)
            self.labels.extend(Z.labels)
            self.descriptions.append(Z.describe())
        self.pos = 0

    def describe(self):
        desc = "Zipped labeled frames collection, %d frames composed of:\n" % (len(self.frames))
        for d in self.descriptions:
            desc += d + "\n"
        return desc

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

class CamVidSingleDataProvider(DataProvider):
    classes = [
        [[64, 128, 64], "Animal"],
        [[192, 0, 128], "Archway"],
        [[0, 128, 192], "Bicyclist"],
        [[0, 128, 64], "Bridge"],
        [[128, 0, 0], "Building"],
        [[64, 0, 128], "Car"],
        [[64, 0, 192], "CartLuggagePram"],
        [[192, 128, 64], "Child"],
        [[192, 192, 128], "Column_Pole"],
        [[64, 64, 128], "Fence"],
        [[128, 0, 192], "LaneMkgsDriv"],
        [[192, 0, 64], "LaneMkgsNonDriv"],
        [[128, 128, 64], "Misc_Text"],
        [[192, 0, 192], "MotorcycleScooter"],
        [[128, 64, 64], "OtherMoving"],
        [[64, 192, 128], "ParkingBlock"],
        [[64, 64, 0], "Pedestrian"],
        [[128, 64, 128], "Road"],
        [[128, 128, 192], "RoadShoulder"],
        [[0, 0, 192], "Sidewalk"],
        [[192, 128, 128], "SignSymbol"],
        [[128, 128, 128], "Sky"],
        [[64, 128, 192], "SUVPickupTruck"],
        [[0, 0, 64], "TrafficCone"],
        [[0, 64, 64], "TrafficLight"],
        [[192, 64, 128], "Train"],
        [[128, 128, 0], "Tree"],
        [[192, 128, 192], "Truck_Bus"],
        [[64, 0, 64], "Tunnel"],
        [[192, 192, 0], "VegetationMisc"],
        [[0, 0, 0], "Void"],
        [[64, 192, 0], "Wall"]
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

    def __init__(self, path, dx, dy, blocks_x=None, blocks_y=None, block_size=6, max_length=10000000000, use_segnet=True):
        self.file_path = os.path.expanduser(path)
        self.use_segnet = use_segnet
        myzip = zipfile.ZipFile(os.path.expanduser(self.file_path), "r")
        self.frames_dict = {}
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
            print(("[ %d/%d ] Extracting " % (i, len(myzip.namelist()))) + element + " "*10 + "\r", end=' ')
            sys.stdout.flush()
            if element.endswith("png"):
                handle = myzip.open(element)
                buf = handle.read()
                img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
                if "video/" in element:
                    nr = int(os.path.basename(element)[-9:-4])
                    framer = cv2.resize(img, dsize=(dx, dy), interpolation=cv2.INTER_CUBIC)
                    framer = framer.astype(np.float32) / 255
                    self.frames_dict[nr] = framer
                    self.filenames.append(os.path.basename(element))
                    self.pos_tran.append(nr)
                if "label/" in element:
                    nr = int(os.path.basename(element)[-11:-6])
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
        return "Zipped labeled Camvid frames %s, %d frames, %d labels" % (self.file_path, len(self.pos_tran), self.labels_cnt)

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

    def decode_label1(self, label, blocks_x=None, blocks_y=None):
        # THis works but is slow and needs to be rewritten.
        # plus it may return a any value that exceeds the threshold
        # not necessairly the max
        if blocks_x is None:
            blocks_x = self.blocks_x
        if blocks_y is None:
            blocks_y = self.blocks_y
        decoded = np.zeros((blocks_y, blocks_x, 3), dtype=np.uint8)
        for (i, c) in enumerate(self.cl):
            xx = i % self.block_size
            yy = i / self.block_size
            decoded[np.where(label[xx::self.block_size, yy::self.block_size] >0.5)] = c[0]
        return cv2.resize(decoded, dsize=self.shape, interpolation=cv2.INTER_NEAREST)

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
        B = blockshaped(label, self.block_size, self.block_size).transpose([0, 2, 1])
        C = B.reshape(B.shape[0], B.shape[1]*B.shape[2])
        amx = np.argmax(C, axis=1)
        amx[np.where(amx >= self.classes_arr.shape[0])] = 0
        mx = C[np.indices(amx.shape), amx].flatten()
        decoded1 = self.classes_arr[amx]
        decoded1[np.where(mx < 0.1)] = [0, 0, 0]
        decoded=decoded1.reshape((blocks_y, blocks_x, 3))
        if shape is None:
            shape=self.shape
        return cv2.resize(decoded, dsize=shape, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    def get_filename(self):
        return self.filenames[self.pos]


class CamVidDataProvider(DataProvider):
    def __init__(self, path_collection, dx, dy, blocks_x=None, blocks_y=None, block_size=6, max_length=10000000000, augment=4):
        self.sequences = []
        self.pos = 0
        self.current_sequence = 0
        self.len = 0
        self.augment = augment
        self.dx_per_b = dx / blocks_x
        self.dy_per_b = dy / blocks_y
        actual_dx = dx+self.augment * self.dx_per_b
        actual_dy = dy+self.augment * self.dy_per_b
        self.dx = dx
        self.dy = dy
        self.blocks_x = blocks_x
        self.blocks_y = blocks_y
        self.block_size = block_size
        if augment > 0:
            self.augment_vec = np.random.randint(0, self.augment, (2,))
        else:
            self.augment_vec = (0, 0)
        for p in path_collection:
            S = CamVidSingleDataProvider(path=p, dx=actual_dx, dy=actual_dy, blocks_x=blocks_x+self.augment, blocks_y=blocks_y+self.augment, block_size=block_size)
            self.sequences.append(S)
            self.len += len(S)

    def __len__(self):
        return self.len

    def get_next(self):
        frame = self.sequences[self.current_sequence].get_next()
        if self.augment>0:
            frame = frame[self.augment_vec[0]*self.dx_per_b:self.augment_vec[0]*self.dx_per_b+self.dx,
                          self.augment_vec[1]*self.dy_per_b:self.augment_vec[1]*self.dy_per_b+self.dy]
        return frame

    def get_label(self, last_valid=False):
        label = self.sequences[self.current_sequence].get_label(last_valid=last_valid)
        if self.augment > 0 and label is not None:
            label = label[self.augment_vec[0]*self.block_size:(self.augment_vec[0] + self.blocks_x)*self.block_size,
                          self.augment_vec[1]*self.block_size:(self.augment_vec[1] + self.blocks_y)*self.block_size]
        return label

    def advance(self):
        self.pos = (self.pos + 1) % self.len
        if self.sequences[self.current_sequence].pos + 1 == len(self.sequences[self.current_sequence]):
            self.current_sequence = (self.current_sequence + 1) % len(self.sequences)
            self.sequences[self.current_sequence].pos = 0
            if self.augment>0:
                self.augment_vec = np.random.randint(0, self.augment, (2,))
        else:
            self.sequences[self.current_sequence].advance()

    def describe(self):
        desc = "Collection of frames: \n"
        for S in self.sequences:
            desc += S.describe() + "\n"
        desc += "%d frames together" % self.len
        return desc

    def decode_label(self, label):
        # TODO:
        # This needs to be modified to reflect the right shape with augmentation
        return self.sequences[0].decode_label(label, blocks_x=self.blocks_x, blocks_y=self.blocks_y, shape=(self.dx, self.dy))


def IOU(img1, img2, rgb):
        A = np.all(img1 == rgb, axis=-1).astype(np.uint8)
        B = np.all(img2 == rgb, axis=-1).astype(np.uint8)
        union = A | B
        inter = A & B
        I1 = np.sum(inter).astype(np.float)/np.prod(A.shape)
        U1 = np.sum(union).astype(np.float)/np.prod(A.shape)
        if U1 == 0:
            return 1.0
        return I1/U1


def IOU_clases(img1, img2, classes, results):
    for (i, cl) in enumerate(classes):
        results[i] = IOU(img1, img2, rgb=cl[0])


if __name__ == "__main__":
    Z = ZipCollectionDataProvider(["~/old_PVMdata/green_ball_bc_office.pkl.zip", "~/old_PVMdata/green_ball_bc_office.pkl.zip"], dx=120, dy=120)
    for i in range(len(Z)):
        img = Z.get_next()
        lab = Z.get_label()
        Z.advance()
        cv2.imshow("Win", img)
        cv2.imshow("label", lab)
        cv2.waitKey(10)
