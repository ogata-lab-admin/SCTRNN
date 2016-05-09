# -*- coding:utf-8 -*-

import sys
import os
import re
import rnn_runner2

def main():
    seed = int(sys.argv[1]) if str.isdigit(sys.argv[1]) else 0
    ignore_index = [int(x) for x in sys.argv[2].split(',') if str.isdigit(x)]
    type = sys.argv[3]
    window_length = int(sys.argv[4]) if str.isdigit(sys.argv[4]) else 10
    reg_count = int(sys.argv[5]) if str.isdigit(sys.argv[5]) else 0
    try:
        rho_init = float(sys.argv[6])
    except ValueError:
        rho_init = 0.001
    try:
        moment = float(sys.argv[7])
    except ValueError:
        moment = 0
    rnn_file = sys.argv[8]
    sequence_file = sys.argv[9]

    rnn_runner2.init_genrand(seed)
    runner = rnn_runner2.RNNRunner()
    runner.init(rnn_file)
    runner.set_time_series_id()

    p = re.compile(r'(^#)|(^$)')
    out_state_queue = []
    for line in open(sequence_file, 'r'):
        if p.match(line) == None:
            input = map(float, line[:-1].split())
            if len(out_state_queue) >= runner.delay_length():
                out_state = out_state_queue.pop(0)
                for i in ignore_index:
                    input[i] = out_state[i]
            runner.update(input, reg_count, rho_init, moment)
            out_state = runner.out_state()
            if type == 'o':
                print '\t'.join([str(x) for x in out_state])
            elif type == 'c':
                c_state = runner.c_state()
                print '\t'.join([str(x) for x in c_state])
            elif type == 'a':
                c_state = runner.c_state()
                print '\t'.join([str(x) for x in out_state + c_state])
            out_state_queue.append(out_state)

if __name__ == '__main__':
    main()

