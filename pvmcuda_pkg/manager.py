# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info
import threading
import socket
import logging
import sys
import time
import pvmcuda_pkg.disp as disp
import cv2
import numpy as np
import os
import pvmcuda_pkg.utils as utils
from .console import DebugConsole
from .data import IOU_clases


class PVMManager(object):
    def __init__(self, PVMObject, DataProvider, ReadoutObject=None, port=9000, snapshot=False, snapdir="pngs", snapoffset=400):
        self.PVMObject = PVMObject
        self.Display = disp.FancyDisplay()
        self.ReadoutObject = ReadoutObject
        self.DataProvider = DataProvider
        self.frame = None
        self.debug_cmds = []
        self.debug_lock = threading.Lock()
        self.port = port
        self.stop_execution = False
        self.t_start = 0
        self.t_previous = 0
        self.t_now = 0
        self.counter = 0
        self._t = threading.Thread(target=self.run_debug_server, args=())
        self._t.start()
        self.setup_display()
        self.fps = 0
        self.do_display = True
        self.mode_dream = False
        self.mode_partial_dream = False
        self.mode_blindspot = False
        self.mode_eval_only = False
        self.mode_pause = False
        self.mode_step = False
        self.mode_debug = False
        self.snapshot = snapshot
        self.snapdir = snapdir
        self.snapoffset = snapoffset
        if snapshot==True:
            utils.make_sure_path_exists(snapdir)
        self.video_recorder = None
        self.frames_to_dump = 0
        self.frames_dumped = 0
        self.evaluate = False

    def setup_display(self):
        input_shape = self.PVMObject.get_input_shape()
        self.Display.add_picture("Input frame", input_shape[0], input_shape[1], 0, 0)
        self.Display.add_picture("Predicted frame", input_shape[0], input_shape[1], 1, 0)
        if self.DataProvider.has_depth:
            self.Display.add_picture("Input depth", input_shape[0], input_shape[1], 0, -1)
            self.Display.add_picture("Predicted depth", input_shape[0], input_shape[1], 1, -1)

        for layer in range(len(self.PVMObject.specs['layer_shapes'])):
            shape = int(self.PVMObject.specs['layer_shapes'][layer]) * int(self.PVMObject.specs['hidden_block_size'])
            self.Display.add_picture("Layer %d" % layer, shape, shape, 0, -1)
        print("Shape")
        print(self.ReadoutObject.shape)
        #self.Display.add_picture("Label", self.ReadoutObject.shape, self.ReadoutObject.shape, 1, -1)
        self.Display.add_picture("Label (ground truth)", input_shape[0], input_shape[1], 1, -1)
        if self.ReadoutObject is not None:
            #self.Display.add_picture("Heatmap",  self.ReadoutObject.shape, self.ReadoutObject.shape, 1, -1)
            self.Display.add_picture("Label predicted", input_shape[0], input_shape[1], 1, -1)
            self.Display.add_picture("Overlap", input_shape[0], input_shape[1], 1, -1)

        self.Display.add_picture("Info", 100, 200, 1, -1)
        self.Display.initialize()

    def run(self, steps=10000):
        self.t_start = time.time()
        for self.counter in range(steps):
            self.process_debug_cmds()
            if self.mode_pause and not self.mode_step:
                time.sleep(0.1)
            else:
                self.mode_step = False
                self.PVMObject.update_learning_rate()
                self.frame = self.DataProvider.get_next()
                if self.DataProvider.has_depth:
                    depth = self.DataProvider.get_depth()
                    if utils.check_if_enabled("ignore_depth", self.PVMObject.specs):
                        depth *= 0
                        #depth += 0.5
                    self.frame = np.concatenate((self.frame, depth[:, :, np.newaxis]), axis=-1)
                self.preprocess_input()
                self.PVMObject.push_input_gpu(utils.compress_tensor(self.frame))
                self.PVMObject.forward_gpu()
                if self.ReadoutObject is not None:
                    self.ReadoutObject.update_learning_rate()
                    self.ReadoutObject.copy_data()
                    self.ReadoutObject.forward()
                    label = self.DataProvider.get_label()
                    if not self.mode_eval_only and label is not None and \
                            (self.PVMObject.step > int(self.PVMObject.specs['delay_intermediate_learning_rate'])):
                        self.ReadoutObject.train(utils.compress_tensor(label))
                if not self.mode_eval_only:
                    self.PVMObject.backward_gpu()
                # self.PVMObject.regularize(0.99)
                self.printfps()
                self.visualize()
                if self.mode_debug:
                    self.PVMObject.display_unit(1)
                    self.PVMObject.display_unit(len(self.PVMObject.graph)/2)
                    self.PVMObject.display_unit(len(self.PVMObject.graph)-1)
                self.PVMObject.step += 1
                self.DataProvider.advance()
                if self.PVMObject.step > 1 and self.PVMObject.step % 100000 == 0:
                    self.save_state()
                if self.ask_stop():
                    break

    def preprocess_input(self):
        """
        Here is where all the dream modes will come in
        :return:
        """
        if self.mode_dream:
            pframe = self.PVMObject.pop_prediction(delta_step=-1)
            self.frame = pframe.copy()
        if self.mode_partial_dream:
            pframe = self.PVMObject.pop_prediction(delta_step=-1)
            pfram1 = (pframe / 255.0).astype(np.float32)
            input_shape = self.PVMObject.get_input_shape()
            spot = int(input_shape[0] / 3)
            self.frame[spot:-spot, spot:-spot] = pfram1[spot:-spot, spot:-spot]
        if self.mode_blindspot:
            input_shape = self.PVMObject.get_input_shape()
            spot = int(input_shape[0]/3)
            self.frame[spot:-spot, spot:-spot] = 0.5


    def visualize(self):
        """
        Here is where the window will be constructed
        :return:
        """
        if self.evaluate:
            label = self.DataProvider.get_label(last_valid=False)
            label_d = self.DataProvider.decode_label(label)
            H = self.ReadoutObject.get_heatmap()
            readout_h = self.DataProvider.decode_label(H)
            R = np.zeros(len(self.DataProvider.get_classes()))
            IOU_clases(label_d, readout_h, self.DataProvider.get_classes(), R)
            self.eval_data += R
            self.eval_frame += 1
            if self.eval_frame == self.max_eval_frame:
                self.evaluate = False
                print("Evalution IOU results")
                for (i, cat) in enumerate(self.DataProvider.get_classes()):
                    print(cat[1], end=' ')
                    print(self.eval_data[i] / self.eval_frame)
                print("----")
                print("Mean", end=' ')
                print(np.mean(self.eval_data/ self.eval_frame))

        if self.do_display or \
                (self.snapshot and (self.PVMObject.step % len(self.DataProvider)) == self.snapoffset) or \
            self.video_recorder is not None or self.frames_to_dump > 0:
            pframe = utils.uncompress_tensor(self.PVMObject.pop_prediction())
            self.Display.place_picture("Input frame", self.frame[:, :, :3])
            self.Display.place_picture("Predicted frame", pframe[:, :, :3])
            if self.DataProvider.has_depth:
                self.Display.place_picture("Input depth", self.frame[:, :, -1])
                self.Display.place_picture("Predicted depth", pframe[:, :, -1])
            label = self.DataProvider.get_label(last_valid=False)
            if label is not None:
                if self.DataProvider.get_attr("test") and self.ReadoutObject is not None:
                    H = self.ReadoutObject.get_heatmap()
                    readout_h = self.DataProvider.decode_label(H)
                    cv2.imwrite("test_img_%05d.png" % self.counter, readout_h)
            label = self.DataProvider.get_label(last_valid=True)
            self.Display.place_picture("Label (ground truth)", self.DataProvider.decode_label(label), flip_bgr=True)
            if self.ReadoutObject is not None:
                #self.Display.place_picture("Heatmap", self.ReadoutObject.get_heatmap())
                H = utils.uncompress_tensor(self.ReadoutObject.get_heatmap())
                heatmap_decoded = self.DataProvider.decode_label(H)
                self.Display.place_picture("Label predicted", heatmap_decoded, flip_bgr=True)
                self.Display.place_picture("Overlap", ((cv2.cvtColor(heatmap_decoded, cv2.COLOR_BGR2RGB)).astype(np.uint8) >> np.uint8(1)) + ((255*self.frame[:, :, :3]).astype(np.uint8) >> np.uint8(1)))
            for layer in range(len(self.PVMObject.specs['layer_shapes'])):
                activ0 = self.PVMObject.pop_layer(layer=layer)
                self.Display.place_picture("Layer %d" % layer, activ0)

            self.Display.place_txt("Info", ["GPU_PVM (c) Filip Piekniewski 2018",
                                            "Step: %d" % self.PVMObject.step,
                                            "PVM_id: %s" % self.PVMObject.uniq_id,
                                            "Create date: %s" % self.PVMObject.time_stamp,
                                            "fps: %2.2f" % self.fps,
                                            "Device: %s" % self.PVMObject.device,
                                            "Training: "+str(not self.mode_eval_only),
                                            "Sequence: "+self.DataProvider.describe()])
            if self.do_display:
                self.Display.show("PVM Display")
                k = cv2.waitKey(1) & 0xFF
                if k == ord('Q'):
                    self.stop_execution = True
                if k == ord('s'):
                    self.save_state()
            if self.snapshot and (self.DataProvider.get_pos() == self.snapoffset):
                self.Display.write(os.path.join(self.snapdir, "%s_pvm_state_%09d.png" % (self.PVMObject.uniq_id, self.PVMObject.step)))
            if self.video_recorder is not None:
                self.video_recorder.record(self.Display.get())
            if self.frames_to_dump > 0:
                cv2.imwrite("PVM_dump_%05d.png" % self.frames_dumped, self.Display.get())
                self.frames_to_dump -= 1
                self.frames_dumped += 1

    def save_state(self):
        self.PVMObject.save("./Sim_%s_%s_%09d.zip" % (self.PVMObject.name, self.PVMObject.uniq_id, self.PVMObject.step))
        if self.ReadoutObject is not None:
            self.ReadoutObject.save(
                "./Sim_%s_%s_%09d.zip" % (self.PVMObject.name, self.PVMObject.uniq_id, self.PVMObject.step))
        if self.PVMObject.step > 100000 and ((self.PVMObject.step - 100000) % 2000000 != 0):
            p_file = "./Sim_%s_%s_%09d.zip" % (self.PVMObject.name, self.PVMObject.uniq_id, self.PVMObject.step - 100000)
            if os.path.exists(p_file):
                print("Removing previously saved file " + p_file)
                os.remove(p_file)

    def printfps(self):
        if self.PVMObject.step % 100 == 0:
            self.t_previous = self.t_now
            self.t_now = time.time()
            elapsed = self.t_now - self.t_start
            self.fps = 100 / (self.t_now - self.t_previous)
            print("%d step, %2.2fs elapsed %2.2f fps, %2.2f inst fps \r" % (
            self.PVMObject.step, elapsed, self.counter / elapsed, self.fps), end=' ')
            sys.stdout.flush()

    def process_debug_cmds(self):
        """
        All the state changing commands regarding execution will be processed here and
        :return:
        """
        self.debug_lock.acquire()
        # do all the debug stuff
        for (i, cmd) in enumerate(self.debug_cmds):
            if cmd == "toggle_display":
                self.do_display = not self.do_display
                if self.do_display == False:
                    cv2.destroyWindow("PVM Display")
                    cv2.destroyAllWindows()
                    for _ in range(10):
                        cv2.waitKey(1)
            if cmd == "freeze_learning":
                self.mode_eval_only = True
                self.PVMObject.freeze_learning()
            if cmd == "unfreeze_learning":
                self.mode_eval_only = False
                self.PVMObject.unfreeze_learning()

            if cmd == "dream":
                self.mode_dream = not self.mode_dream
            if cmd == "pause":
                self.mode_pause = True
            if cmd == "resume":
                self.mode_pause = False
            if cmd == "step":
                self.mode_step = True
            if cmd.startswith("record"):
                self.video_recorder = utils.VideoRecorder(rec_filename=os.path.expanduser(cmd[7:]))
            if cmd == "stop_recording":
                if self.video_recorder is not None:
                    self.video_recorder.finish()
                self.video_recorder = None
            if cmd == "blindspot":
                self.mode_blindspot = not self.mode_blindspot
            if cmd == "debug":
                self.mode_debug = not self.mode_debug
            if cmd == "partial_dream":
                self.mode_partial_dream = not self.mode_partial_dream
            if cmd.startswith("set_pvm_lr"):
                self.PVMObject.update_learning_rate(override_rate=float(cmd[10:]))
            if cmd.startswith("set_readout_lr"):
                self.ReadoutObject.mlp.set_learning_rate(float(cmd[15:]))
            if cmd.startswith("dump_frames"):
                self.frames_to_dump = int(cmd[11:])
                self.frames_dumped = 0
            if cmd.startswith("reset_dataset"):
                self.DataProvider.reset_pos()
            if cmd.startswith("eval"):
                self.max_eval_frame = int(cmd[5:])
                self.eval_frame = 0
                self.eval_data = np.zeros(len(self.DataProvider.get_classes()))
                self.evaluate = True

            self.debug_cmds.pop(i)
        #self.debug_cmds = []
        #print self.debug_cmds
        self.debug_lock.release()

    def run_debug_server(self):
        serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while True:
            try:
                serversock.bind(("", self.port))
                break
            except:
                logging.error("Port %d already taken, trying to bind to port %d" % (self.port, self.port + 1))
                self.port += 1
                continue
        serversock.listen(5)
        logging.info("Listening on port " + str(self.port) + " for debug connections")
        clients = []
        monitor = threading.Thread(target=self._monitor_debug_session, args=())
        monitor.start()
        while 1:
            clientsock, addr = serversock.accept()
            if self.ask_stop():
                logging.info("Exiting debug session")
                sys.stdout.flush()
                break
            logging.info("Accepted a debug connection from " + str(addr))
            clients.append(threading.Thread(target=self._run_debug_session, args=(clientsock,)))
            clients[-1].start()
        monitor.join()

    def _monitor_debug_session(self):
        while 1:
            time.sleep(0.5)
            if self.ask_stop():
                S = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                S.connect(("localhost", self.port))
                S.close()
                logging.info("Closing the server socket")
                break

    def _run_debug_session(self, socket):
        file = socket.makefile(mode="rw")
        print("", file=file)
        print("Predictive Vision Framework version 2.0", file=file)
        print("(C) 2018 Filip Piekniewski, All rights reserved", file=file)
        print("", file=file)
        print("You are connected to a debug shell", file=file)
        print("PVM model: " + self.PVMObject.uniq_id, file=file)
        print("Create date " + self.PVMObject.time_stamp, file=file)
        print("Device " + self.PVMObject.device, file=file)
        print("Dataset " + self.DataProvider.describe(), file=file)
        print("Type 'help' for available commands, 'quit' to exit debug shell", file=file)
        explorer = DebugConsole(stdin=file, stdout=file, cmd_buf=self.debug_cmds, cmd_lock=self.debug_lock)
        explorer.cmdloop()
        logging.info("Ended a remote debug session")
        socket.close()

    def ask_stop(self):
        return self.stop_execution

    def stop(self):
        self.stop_execution = True


if __name__ == "__main__":
    pass
