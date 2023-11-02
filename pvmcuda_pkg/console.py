import numpy as np
import os
import sys
import argparse
import pprint
import cmd
import traceback
import time
import logging


class DebugConsole(cmd.Cmd):
    use_rawinput = False

    def __init__(self,
                 stdin=None,
                 stdout=None,
                 cmd_buf=None,
                 cmd_lock=None):
        if stdin is not None:
            self.stdin = stdin
        else:
            self.stdin=sys.stdin
            self.use_rawinput=True
        if stdout is not None:
            self.stdout = stdout
        else:
            self.stdout=sys.stdout
        cmd.Cmd.__init__(self, stdin=self.stdin, stdout=self.stdout)
        self.cmd_buf = cmd_buf
        self.cmd_lock = cmd_lock
        self.prompt = '\033[1m' + ">" + '\033[0m'

    def do_toggle_display(self, line):
        """
        Enables/disables display window
        :param line:
        :return:
        """
        self.cmd_lock.acquire()
        self.cmd_buf.append("toggle_display")
        self.cmd_lock.release()

    def do_freeze_learning(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("freeze_learning")
        self.cmd_lock.release()

    def do_unfreeze_learning(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("unfreeze_learning")
        self.cmd_lock.release()

    def do_unfreeze_learning(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("unfreeze_learning")
        self.cmd_lock.release()

    def do_toggle_dream(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("dream")
        self.cmd_lock.release()

    def do_pause(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("pause")
        self.cmd_lock.release()

    def do_resume(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("resume")
        self.cmd_lock.release()

    def do_step(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("step")
        self.cmd_lock.release()

    def do_record(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("record "+line)
        self.cmd_lock.release()

    def do_stop_recording(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("stop_recording")
        self.cmd_lock.release()

    def do_toggle_blindspot(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("blindspot")
        self.cmd_lock.release()

    def do_toggle_partial_dream(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("partial_dream")
        self.cmd_lock.release()

    def do_toggle_debug(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("debug")
        self.cmd_lock.release()

    def do_set_pvm_lr(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("set_pvm_lr " + line)
        self.cmd_lock.release()

    def do_set_readout_lr(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("set_readout_lr " + line)
        self.cmd_lock.release()

    def do_dump_frames(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("dump_frames " + line)
        self.cmd_lock.release()

    def do_reset_dataset(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("reset_dataset")
        self.cmd_lock.release()

    def do_eval(self, line):
        self.cmd_lock.acquire()
        self.cmd_buf.append("eval " + line)
        self.cmd_lock.release()


    def do_quit(self, line):
        """
        Exit program
        """
        return True

    def do_EOF(self, line):
        return True