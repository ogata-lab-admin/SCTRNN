# -*- coding:utf-8 -*-

import sys
import os
import datetime
from ctypes import *
from ctypes.util import find_library

libpath = find_library('rnnrunner')
librunner = cdll.LoadLibrary(libpath)

librunner.init_genrand.argtype = c_ulong
librunner.rnn_in_state_from_runner2.restype = POINTER(c_double)
librunner.rnn_c_state_from_runner2.restype = POINTER(c_double)
librunner.rnn_c_inter_state_from_runner2.restype = POINTER(c_double)
librunner.rnn_out_state_from_runner2.restype = POINTER(c_double)
librunner.update_rnn_runner2.argtypes = [c_void_p, POINTER(c_double), c_int, c_double, c_double]


def init_genrand(seed):
    if seed == 0:
        now = datetime.datetime.utcnow()
        seed = ((now.hour * 3600 + now.minute * 60 + now.second) *
                now.microsecond)
    librunner.init_genrand(c_ulong(seed % 4294967295 + 1))


class RNNRunner(object):
    def __init__(self, librunner=librunner):
        self.runner = c_void_p()
        self.librunner = librunner
        self.librunner._new_rnn_runner2(byref(self.runner))
        self.is_initialized = False

    def __del__(self):
        self.free()
        self.librunner._delete_rnn_runner2(self.runner)

    def init(self, file_name, window_length):
        self.free()
        self.librunner.init_rnn_runner2_with_filename(self.runner, file_name,
                window_length)
        self.is_initialized = True

    def free(self):
        if self.is_initialized:
            self.librunner.free_rnn_runner2(self.runner)
        self.is_initialized = False

    def set_time_series_id(self, id=0):
        self.librunner.set_init_state_of_rnn_runner2(self.runner, id)

    def target_num(self):
        return self.librunner.rnn_target_num_from_runner2(self.runner)

    def in_state_size(self):
        return self.librunner.rnn_in_state_size_from_runner2(self.runner)

    def c_state_size(self):
        return self.librunner.rnn_c_state_size_from_runner2(self.runner)

    def out_state_size(self):
        return self.librunner.rnn_out_state_size_from_runner2(self.runner)

    def delay_length(self):
        return self.librunner.rnn_delay_length_from_runner2(self.runner)

    def output_type(self):
        return self.librunner.rnn_output_type_from_runner2(self.runner)

    def update(self, in_state, reg_count, rho_init, momentum):
        if in_state != None:
            x = (c_double * len(in_state))()
            for i in xrange(len(in_state)):
                x[i] = c_double(in_state[i])
        else:
            x = None
        self.librunner.update_rnn_runner2(self.runner, x, reg_count, rho_init,
                momentum)

    def in_state(self, in_state=None):
        x = self.librunner.rnn_in_state_from_runner2(self.runner)
        if in_state != None:
            for i in xrange(len(in_state)):
                x[i] = c_double(in_state[i])
        return [x[i] for i in xrange(self.in_state_size())]

    def c_state(self, c_state=None):
        x = self.librunner.rnn_c_state_from_runner2(self.runner)
        if c_state != None:
            for i in xrange(len(c_state)):
                x[i] = c_double(c_state[i])
        return [x[i] for i in xrange(self.c_state_size())]

    def c_inter_state(self, c_inter_state=None):
        x = self.librunner.rnn_c_inter_state_from_runner2(self.runner)
        if c_inter_state != None:
            for i in xrange(len(c_inter_state)):
                x[i] = c_double(c_inter_state[i])
        return [x[i] for i in xrange(self.c_state_size())]

    def out_state(self, out_state=None):
        x = self.librunner.rnn_out_state_from_runner2(self.runner)
        if out_state != None:
            for i in xrange(len(out_state)):
                x[i] = c_double(out_state[i])
        return [x[i] for i in xrange(self.out_state_size())]

