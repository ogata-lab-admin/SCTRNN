/*
    Copyright (c) 2011, Jun Namikawa <jnamika@gmail.com>

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef RNN_RUNNER2_H
#define RNN_RUNNER2_H

#include "rnn.h"


typedef struct rnn_runner2 {
    int id;
    int delay_length;
    struct recurrent_neural_network rnn;
} rnn_runner2;



int _new_rnn_runner2 (struct rnn_runner2 **runner);

void _delete_rnn_runner2 (struct rnn_runner2 *runner);


void init_rnn_runner2 (
        struct rnn_runner2 *runner,
        FILE *fp,
        int window_length);

void init_rnn_runner2_with_filename (
        struct rnn_runner2 *runner,
        const char *filename,
        int window_length);

void free_rnn_runner2 (struct rnn_runner2 *runner);

void set_init_state_of_rnn_runner2 (
        struct rnn_runner2 *runner,
        int series_id);

void update_rnn_runner2 (
        struct rnn_runner2 *runner,
        double *input,
        int reg_count,
        double rho_init,
        double momentum);


int rnn_in_state_size_from_runner2 (struct rnn_runner2 *runner);
int rnn_c_state_size_from_runner2 (struct rnn_runner2 *runner);
int rnn_out_state_size_from_runner2 (struct rnn_runner2 *runner);
int rnn_delay_length_from_runner2 (struct rnn_runner2 *runner);
int rnn_window_length_from_runner2 (struct rnn_runner2 *runner);
int rnn_output_type_from_runner2 (struct rnn_runner2 *runner);
int rnn_target_num_from_runner2 (struct rnn_runner2 *runner);
double* rnn_in_state_from_runner2 (struct rnn_runner2 *runner);
double* rnn_c_state_from_runner2 (struct rnn_runner2 *runner);
double* rnn_c_inter_state_from_runner2 (struct rnn_runner2 *runner);
double* rnn_out_state_from_runner2 (struct rnn_runner2 *runner);
double* rnn_var_state_from_runner2 (struct rnn_runner2 *runner);
struct rnn_state* rnn_state_from_runner2 (struct rnn_runner2 *runner);

#endif

