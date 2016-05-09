/*
    Copyright (c) 2009-2011, Jun Namikawa <jnamika@gmail.com>

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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>

#include "utils.h"
#include "rnn.h"


#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif


#ifndef FIXED_WEIGHT
#define FIXED_WEIGHT 0
#endif
#ifndef FIXED_THRESHOLD
#define FIXED_THRESHOLD 0
#endif
#ifndef FIXED_TAU
#define FIXED_TAU 1
#endif
#ifndef FIXED_INIT_C_STATE
#define FIXED_INIT_C_STATE 0
#endif
#ifndef FIXED_IN_STATE
#define FIXED_IN_STATE 1
#endif

#ifndef INIT_TAU
#define INIT_TAU 1.0
#endif
#ifndef MIN_VARIANCE
#define MIN_VARIANCE 0.001
#endif

#ifdef ENABLE_ADAPTIVE_LEARNING_RATE

#ifndef MAX_ITERATION_IN_ADAPTIVE_LR
#define MAX_ITERATION_IN_ADAPTIVE_LR 1000
#endif

#ifndef MAX_PERF_INC
#define MAX_PERF_INC 1.1
#endif

#ifndef LR_DEC
#define LR_DEC 0.7
#endif

#ifndef LR_INC
#define LR_INC 1.05
#endif

#endif // ENABLE_ADAPTIVE_LEARNING_RATE



#define foreach(i,c) \
    for (int _c = 0; (c)[_c].begin != -1; _c++) \
        for (int i = (c)[_c].begin, _e = (c)[_c].end; i < _e; i++)

#define foreach_maybe_break(i,c) \
    if ((c)[0].begin != -1) \
        for (int i = (c)[0].begin, e = (c)[0].end, _c = 0; \
            i < e || (e = (c)[++_c].end, i = (c)[_c].begin) != -1; i++)



/******************************************************************************/
/********** Initialization and Free *******************************************/
/******************************************************************************/

void init_rnn_parameters (
        struct rnn_parameters *rnn_p,
        int in_state_size,
        int c_state_size,
        int out_state_size,
        int rep_init_size)
{
    double max_wi, max_wc;

    /*
     * RNN has to contain at least one context neuron and one output neuron.
     * An input neuron is not necessarily required.
     */
    assert(in_state_size >= 0);
    assert(c_state_size >= 1);
    assert(out_state_size >= 1);
    assert(rep_init_size >= 1);

    rnn_p->in_state_size = in_state_size;
    rnn_p->c_state_size = c_state_size;
    rnn_p->out_state_size = out_state_size;
    rnn_p->rep_init_size = rep_init_size;
    rnn_p->output_type = STANDARD_TYPE;
    rnn_p->fixed_weight = FIXED_WEIGHT;
    rnn_p->fixed_threshold = FIXED_THRESHOLD;
    rnn_p->fixed_tau = FIXED_TAU;
    rnn_p->fixed_init_c_state = FIXED_INIT_C_STATE;
    rnn_p->softmax_group_num = 1;
    rnn_p->rep_init_variance = 1;
    rnn_p->prior_strength = 0;

    rnn_parameters_alloc(rnn_p);

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        rnn_p->const_init_c[i] = 0;
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        rnn_p->softmax_group_id[i] = 0;
    }

    max_wi = 1.0 / rnn_p->in_state_size;
    max_wc = 1.0 / rnn_p->c_state_size;

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            rnn_p->weight_ci[i][j] = max_wi * (2*genrand_real1()-1);
        }
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            rnn_p->weight_cc[i][j] = max_wc * (2*genrand_real1()-1);
        }
        rnn_p->threshold_c[i] = 2*genrand_real1()-1;
        rnn_p->tau[i] = INIT_TAU;
        rnn_p->eta[i] = 1.0/rnn_p->tau[i];
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            rnn_p->weight_oc[i][j] = max_wc * (2*genrand_real1()-1);
            rnn_p->weight_vc[i][j] = max_wc * (2*genrand_real1()-1);
        }
        rnn_p->threshold_o[i] = 2*genrand_real1()-1;
        rnn_p->threshold_v[i] = 2*genrand_real1()-1;
    }

    double max_ri = 1.0 / rnn_p->c_state_size;
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            rnn_p->rep_init_c[i][j] = max_ri * (2*genrand_real1()-1);
        }
    }

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        for (int j = 0; j <= rnn_p->in_state_size; j++) {
            rnn_p->connection_ci[i][j].begin = -1;
            rnn_p->connection_ci[i][j].end = -1;
        }
        for (int j = 0; j <= rnn_p->c_state_size; j++) {
            rnn_p->connection_cc[i][j].begin = -1;
            rnn_p->connection_cc[i][j].end = -1;
        }
        rnn_add_connection(rnn_p->in_state_size, rnn_p->connection_ci[i], 0,
                rnn_p->in_state_size);
        rnn_add_connection(rnn_p->c_state_size, rnn_p->connection_cc[i], 0,
                rnn_p->c_state_size);
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        for (int j = 0; j <= rnn_p->c_state_size; j++) {
            rnn_p->connection_oc[i][j].begin = -1;
            rnn_p->connection_oc[i][j].end = -1;
            rnn_p->connection_vc[i][j].begin = -1;
            rnn_p->connection_vc[i][j].end = -1;
        }
        rnn_add_connection(rnn_p->c_state_size, rnn_p->connection_oc[i], 0,
                rnn_p->c_state_size);
        rnn_add_connection(rnn_p->c_state_size, rnn_p->connection_vc[i], 0,
                rnn_p->c_state_size);
    }

    rnn_reset_delta_parameters(rnn_p);
    rnn_reset_prior_distribution(rnn_p);
}


void init_rnn_state (
        struct rnn_state *rnn_s,
        struct rnn_parameters *rnn_p,
        int length,
        const double* const* input,
        const double* const* target)
{
    assert(length > 0);

    rnn_s->rnn_p = rnn_p;
    rnn_s->length = length;

    rnn_state_alloc(rnn_s);

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        //rnn_s->init_c_inter_state[i] = (2*genrand_real1()-1);
        rnn_s->init_c_inter_state[i] = 0;
        rnn_s->init_c_state[i] = tanh(rnn_s->init_c_inter_state[i]);
        rnn_s->delta_init_c_inter_state[i] = 0;
    }
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        rnn_s->gate_init_c[i] = 1.0 / rnn_p->rep_init_size;
        rnn_s->beta_init_c[i] = 0.0;
        rnn_s->delta_beta_init_c[i] = 0.0;
    }
    if (input != NULL) {
        for (int n = 0; n < rnn_s->length; n++) {
            for (int i = 0; i < rnn_p->in_state_size; i++) {
                rnn_s->in_state[n][i] = input[n][i];
            }
        }
    }
    if (target != NULL) {
        for (int n = 0; n < rnn_s->length; n++) {
            for (int i = 0; i < rnn_p->out_state_size; i++) {
                rnn_s->teach_state[n][i] = target[n][i];
            }
        }
    }
}


void init_recurrent_neural_network (
        struct recurrent_neural_network *rnn,
        int in_state_size,
        int c_state_size,
        int out_state_size,
        int rep_init_size)
{
    rnn->series_num = 0;
    rnn->rnn_s = NULL;
    init_rnn_parameters(&rnn->rnn_p, in_state_size, c_state_size,
            out_state_size, rep_init_size);
}



void rnn_add_target (
        struct recurrent_neural_network *rnn,
        int length,
        const double* const* input,
        const double* const* target)
{
    rnn->series_num++;
    REALLOC(rnn->rnn_s, rnn->series_num);
    init_rnn_state(rnn->rnn_s + (rnn->series_num-1), &rnn->rnn_p, length, input,
            target);
}


void rnn_clean_target (struct recurrent_neural_network *rnn)
{
    for (int i = 0; i < rnn->series_num; i++) {
        free_rnn_state(rnn->rnn_s + i);
    }
    FREE(rnn->rnn_s);
    rnn->series_num = 0;
}


void rnn_parameters_alloc (struct rnn_parameters *rnn_p)
{
    const int in_state_size = rnn_p->in_state_size;
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    const int rep_init_size = rnn_p->rep_init_size;

    MALLOC(rnn_p->const_init_c, c_state_size);
    MALLOC(rnn_p->softmax_group_id, out_state_size);

    MALLOC2(rnn_p->weight_ci, c_state_size, in_state_size);
    MALLOC2(rnn_p->weight_cc, c_state_size, c_state_size);
    MALLOC2(rnn_p->weight_oc, out_state_size, c_state_size);
    MALLOC2(rnn_p->weight_vc, out_state_size, c_state_size);
    MALLOC2(rnn_p->delta_weight_ci, c_state_size, in_state_size);
    MALLOC2(rnn_p->delta_weight_cc, c_state_size, c_state_size);
    MALLOC2(rnn_p->delta_weight_oc, out_state_size, c_state_size);
    MALLOC2(rnn_p->delta_weight_vc, out_state_size, c_state_size);
    MALLOC2(rnn_p->prior_weight_ci, c_state_size, in_state_size);
    MALLOC2(rnn_p->prior_weight_cc, c_state_size, c_state_size);
    MALLOC2(rnn_p->prior_weight_oc, out_state_size, c_state_size);
    MALLOC2(rnn_p->prior_weight_vc, out_state_size, c_state_size);

    MALLOC(rnn_p->threshold_c, c_state_size);
    MALLOC(rnn_p->threshold_o, out_state_size);
    MALLOC(rnn_p->threshold_v, out_state_size);
    MALLOC(rnn_p->tau, c_state_size);
    MALLOC(rnn_p->eta, c_state_size);
    MALLOC(rnn_p->delta_threshold_c, c_state_size);
    MALLOC(rnn_p->delta_threshold_o, out_state_size);
    MALLOC(rnn_p->delta_threshold_v, out_state_size);
    MALLOC(rnn_p->delta_tau, c_state_size);
    MALLOC(rnn_p->prior_threshold_c, c_state_size);
    MALLOC(rnn_p->prior_threshold_o, out_state_size);
    MALLOC(rnn_p->prior_threshold_v, out_state_size);
    MALLOC(rnn_p->prior_tau, c_state_size);

    MALLOC2(rnn_p->rep_init_c, rep_init_size, c_state_size);
    MALLOC2(rnn_p->delta_rep_init_c, rep_init_size, c_state_size);
    MALLOC2(rnn_p->prior_rep_init_c, rep_init_size, c_state_size);

    MALLOC2(rnn_p->connection_ci, c_state_size, (in_state_size + 1));
    MALLOC2(rnn_p->connection_cc, c_state_size, (c_state_size + 1));
    MALLOC2(rnn_p->connection_oc, out_state_size, (c_state_size + 1));
    MALLOC2(rnn_p->connection_vc, out_state_size, (c_state_size + 1));

#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    MALLOC(rnn_p->tmp_weight_ci, c_state_size * in_state_size);
    MALLOC(rnn_p->tmp_weight_cc, c_state_size * c_state_size);
    MALLOC(rnn_p->tmp_weight_oc, out_state_size * c_state_size);
    MALLOC(rnn_p->tmp_weight_vc, out_state_size * c_state_size);
    MALLOC(rnn_p->tmp_threshold_c, c_state_size);
    MALLOC(rnn_p->tmp_threshold_o, out_state_size);
    MALLOC(rnn_p->tmp_threshold_v, out_state_size);
    MALLOC(rnn_p->tmp_tau, c_state_size);
    MALLOC(rnn_p->tmp_eta, c_state_size);
    MALLOC(rnn_p->tmp_rep_init_c, rep_init_size * c_state_size);
#endif
}

void free_rnn_parameters (struct rnn_parameters *rnn_p)
{
    FREE(rnn_p->const_init_c);
    FREE(rnn_p->softmax_group_id);
    FREE2(rnn_p->weight_ci);
    FREE2(rnn_p->weight_cc);
    FREE2(rnn_p->weight_oc);
    FREE2(rnn_p->weight_vc);
    FREE2(rnn_p->delta_weight_ci);
    FREE2(rnn_p->delta_weight_cc);
    FREE2(rnn_p->delta_weight_oc);
    FREE2(rnn_p->delta_weight_vc);
    FREE2(rnn_p->prior_weight_ci);
    FREE2(rnn_p->prior_weight_cc);
    FREE2(rnn_p->prior_weight_oc);
    FREE2(rnn_p->prior_weight_vc);
    FREE(rnn_p->threshold_c);
    FREE(rnn_p->threshold_o);
    FREE(rnn_p->threshold_v);
    FREE(rnn_p->tau);
    FREE(rnn_p->eta);
    FREE(rnn_p->delta_threshold_c);
    FREE(rnn_p->delta_threshold_o);
    FREE(rnn_p->delta_threshold_v);
    FREE(rnn_p->delta_tau);
    FREE(rnn_p->prior_threshold_c);
    FREE(rnn_p->prior_threshold_o);
    FREE(rnn_p->prior_threshold_v);
    FREE(rnn_p->prior_tau);
    FREE2(rnn_p->rep_init_c);
    FREE2(rnn_p->delta_rep_init_c);
    FREE2(rnn_p->prior_rep_init_c);
    FREE2(rnn_p->connection_ci);
    FREE2(rnn_p->connection_cc);
    FREE2(rnn_p->connection_oc);
    FREE2(rnn_p->connection_vc);
#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    FREE(rnn_p->tmp_weight_ci);
    FREE(rnn_p->tmp_weight_cc);
    FREE(rnn_p->tmp_weight_oc);
    FREE(rnn_p->tmp_weight_vc);
    FREE(rnn_p->tmp_threshold_c);
    FREE(rnn_p->tmp_threshold_o);
    FREE(rnn_p->tmp_threshold_v);
    FREE(rnn_p->tmp_tau);
    FREE(rnn_p->tmp_eta);
    FREE(rnn_p->tmp_rep_init_c);
#endif
}



void rnn_state_alloc (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    const int in_state_size = rnn_p->in_state_size;
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    const int rep_init_size = rnn_p->rep_init_size;
    const int length = rnn_s->length;

    MALLOC(rnn_s->init_c_inter_state, c_state_size);
    MALLOC(rnn_s->init_c_state, c_state_size);
    MALLOC(rnn_s->delta_init_c_inter_state, c_state_size);
    MALLOC(rnn_s->gate_init_c, rep_init_size);
    MALLOC(rnn_s->beta_init_c, rep_init_size);
    MALLOC(rnn_s->delta_beta_init_c, rep_init_size);

    MALLOC2(rnn_s->in_state, length, in_state_size);
    MALLOC2(rnn_s->c_state, length, c_state_size);
    MALLOC2(rnn_s->out_state, length, out_state_size);
    MALLOC2(rnn_s->var_state, length, out_state_size);
    MALLOC2(rnn_s->teach_state, length, out_state_size);
    MALLOC2(rnn_s->c_inputsum, length, c_state_size);
    MALLOC2(rnn_s->c_inter_state, length, c_state_size);
    MALLOC2(rnn_s->o_inter_state, length, out_state_size);
    MALLOC2(rnn_s->v_inter_state, length, out_state_size);
    MALLOC2(rnn_s->likelihood, length, out_state_size);
    MALLOC2(rnn_s->delta_likelihood, length, out_state_size);
    MALLOC2(rnn_s->delta_c_inter, length, c_state_size);
    MALLOC2(rnn_s->delta_o_inter, length, out_state_size);
    MALLOC2(rnn_s->delta_v_inter, length, out_state_size);

    MALLOC2(rnn_s->delta_w_ci, c_state_size, in_state_size);
    MALLOC2(rnn_s->delta_w_cc, c_state_size, c_state_size);
    MALLOC2(rnn_s->delta_w_oc, out_state_size, c_state_size);
    MALLOC2(rnn_s->delta_w_vc, out_state_size, c_state_size);
    MALLOC(rnn_s->delta_t_c, c_state_size);
    MALLOC(rnn_s->delta_t_o, out_state_size);
    MALLOC(rnn_s->delta_t_v, out_state_size);
    MALLOC(rnn_s->delta_tau, c_state_size);
    MALLOC(rnn_s->delta_i, c_state_size);
    MALLOC(rnn_s->delta_b, rep_init_size);
#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    MALLOC(rnn_s->tmp_init_c_inter_state, c_state_size);
    MALLOC(rnn_s->tmp_init_c_state, c_state_size);
    MALLOC(rnn_s->tmp_gate_init_c, rep_init_size);
    MALLOC(rnn_s->tmp_beta_init_c, rep_init_size);
#endif
}


void free_rnn_state (struct rnn_state *rnn_s)
{
    FREE(rnn_s->init_c_inter_state);
    FREE(rnn_s->init_c_state);
    FREE(rnn_s->delta_init_c_inter_state);
    FREE(rnn_s->gate_init_c);
    FREE(rnn_s->beta_init_c);
    FREE(rnn_s->delta_beta_init_c);
    FREE2(rnn_s->in_state);
    FREE2(rnn_s->c_state);
    FREE2(rnn_s->out_state);
    FREE2(rnn_s->var_state);
    FREE2(rnn_s->teach_state);
    FREE2(rnn_s->c_inputsum);
    FREE2(rnn_s->c_inter_state);
    FREE2(rnn_s->o_inter_state);
    FREE2(rnn_s->v_inter_state);
    FREE2(rnn_s->likelihood);
    FREE2(rnn_s->delta_likelihood);
    FREE2(rnn_s->delta_c_inter);
    FREE2(rnn_s->delta_o_inter);
    FREE2(rnn_s->delta_v_inter);
    FREE2(rnn_s->delta_w_ci);
    FREE2(rnn_s->delta_w_cc);
    FREE2(rnn_s->delta_w_oc);
    FREE2(rnn_s->delta_w_vc);
    FREE(rnn_s->delta_t_c);
    FREE(rnn_s->delta_t_o);
    FREE(rnn_s->delta_t_v);
    FREE(rnn_s->delta_tau);
    FREE(rnn_s->delta_i);
    FREE(rnn_s->delta_b);
#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    FREE(rnn_s->tmp_init_c_inter_state);
    FREE(rnn_s->tmp_init_c_state);
    FREE(rnn_s->tmp_gate_init_c);
    FREE(rnn_s->tmp_beta_init_c);
#endif
}


void free_recurrent_neural_network (struct recurrent_neural_network *rnn)
{
    rnn_clean_target(rnn);
    free_rnn_parameters(&rnn->rnn_p);
}


/******************************************************************************/
/********** File IO ***********************************************************/
/******************************************************************************/

void fwrite_rnn_parameters (
        const struct rnn_parameters *rnn_p,
        FILE *fp)
{
    FWRITE(&rnn_p->in_state_size, 1, fp);
    FWRITE(&rnn_p->c_state_size, 1, fp);
    FWRITE(&rnn_p->out_state_size, 1, fp);
    FWRITE(&rnn_p->rep_init_size, 1, fp);
    FWRITE(&rnn_p->output_type, 1, fp);
    FWRITE(&rnn_p->fixed_weight, 1, fp);
    FWRITE(&rnn_p->fixed_threshold, 1, fp);
    FWRITE(&rnn_p->fixed_tau, 1, fp);
    FWRITE(&rnn_p->fixed_init_c_state, 1, fp);
    FWRITE(&rnn_p->softmax_group_num, 1, fp);
    FWRITE(&rnn_p->rep_init_variance, 1, fp);
    FWRITE(&rnn_p->prior_strength, 1, fp);

    FWRITE(rnn_p->const_init_c, rnn_p->c_state_size, fp);
    FWRITE(rnn_p->softmax_group_id, rnn_p->out_state_size, fp);
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        FWRITE(rnn_p->weight_ci[i], rnn_p->in_state_size, fp);
        FWRITE(rnn_p->weight_cc[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->delta_weight_ci[i], rnn_p->in_state_size, fp);
        FWRITE(rnn_p->delta_weight_cc[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->prior_weight_ci[i], rnn_p->in_state_size, fp);
        FWRITE(rnn_p->prior_weight_cc[i], rnn_p->c_state_size, fp);
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        FWRITE(rnn_p->weight_oc[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->weight_vc[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->delta_weight_oc[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->delta_weight_vc[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->prior_weight_oc[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->prior_weight_vc[i], rnn_p->c_state_size, fp);
    }
    FWRITE(rnn_p->threshold_c, rnn_p->c_state_size, fp);
    FWRITE(rnn_p->threshold_o, rnn_p->out_state_size, fp);
    FWRITE(rnn_p->threshold_v, rnn_p->out_state_size, fp);
    FWRITE(rnn_p->tau, rnn_p->c_state_size, fp);
    FWRITE(rnn_p->eta, rnn_p->c_state_size, fp);
    FWRITE(rnn_p->delta_threshold_c, rnn_p->c_state_size, fp);
    FWRITE(rnn_p->delta_threshold_o, rnn_p->out_state_size, fp);
    FWRITE(rnn_p->delta_threshold_v, rnn_p->out_state_size, fp);
    FWRITE(rnn_p->delta_tau, rnn_p->c_state_size, fp);
    FWRITE(rnn_p->prior_threshold_c, rnn_p->c_state_size, fp);
    FWRITE(rnn_p->prior_threshold_o, rnn_p->out_state_size, fp);
    FWRITE(rnn_p->prior_threshold_v, rnn_p->out_state_size, fp);
    FWRITE(rnn_p->prior_tau, rnn_p->c_state_size, fp);
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        FWRITE(rnn_p->rep_init_c[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->delta_rep_init_c[i], rnn_p->c_state_size, fp);
        FWRITE(rnn_p->prior_rep_init_c[i], rnn_p->c_state_size, fp);
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        for (int j = 0; j <= rnn_p->in_state_size; j++) {
            FWRITE(&rnn_p->connection_ci[i][j].begin, 1, fp);
            FWRITE(&rnn_p->connection_ci[i][j].end, 1, fp);
        }
        for (int j = 0; j <= rnn_p->c_state_size; j++) {
            FWRITE(&rnn_p->connection_cc[i][j].begin, 1, fp);
            FWRITE(&rnn_p->connection_cc[i][j].end, 1, fp);
        }
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        for (int j = 0; j <= rnn_p->c_state_size; j++) {
            FWRITE(&rnn_p->connection_oc[i][j].begin, 1, fp);
            FWRITE(&rnn_p->connection_oc[i][j].end, 1, fp);
            FWRITE(&rnn_p->connection_vc[i][j].begin, 1, fp);
            FWRITE(&rnn_p->connection_vc[i][j].end, 1, fp);
        }
    }
}


void fread_rnn_parameters (
        struct rnn_parameters *rnn_p,
        FILE *fp)
{
    FREAD(&rnn_p->in_state_size, 1, fp);
    FREAD(&rnn_p->c_state_size, 1, fp);
    FREAD(&rnn_p->out_state_size, 1, fp);
    FREAD(&rnn_p->rep_init_size, 1, fp);
    FREAD(&rnn_p->output_type, 1, fp);
    FREAD(&rnn_p->fixed_weight, 1, fp);
    FREAD(&rnn_p->fixed_threshold, 1, fp);
    FREAD(&rnn_p->fixed_tau, 1, fp);
    FREAD(&rnn_p->fixed_init_c_state, 1, fp);
    FREAD(&rnn_p->softmax_group_num, 1, fp);
    FREAD(&rnn_p->rep_init_variance, 1, fp);
    FREAD(&rnn_p->prior_strength, 1, fp);

    rnn_parameters_alloc(rnn_p);

    FREAD(rnn_p->const_init_c, rnn_p->c_state_size, fp);
    FREAD(rnn_p->softmax_group_id, rnn_p->out_state_size, fp);
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        FREAD(rnn_p->weight_ci[i], rnn_p->in_state_size, fp);
        FREAD(rnn_p->weight_cc[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->delta_weight_ci[i], rnn_p->in_state_size, fp);
        FREAD(rnn_p->delta_weight_cc[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->prior_weight_ci[i], rnn_p->in_state_size, fp);
        FREAD(rnn_p->prior_weight_cc[i], rnn_p->c_state_size, fp);
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        FREAD(rnn_p->weight_oc[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->weight_vc[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->delta_weight_oc[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->delta_weight_vc[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->prior_weight_oc[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->prior_weight_vc[i], rnn_p->c_state_size, fp);
    }
    FREAD(rnn_p->threshold_c, rnn_p->c_state_size, fp);
    FREAD(rnn_p->threshold_o, rnn_p->out_state_size, fp);
    FREAD(rnn_p->threshold_v, rnn_p->out_state_size, fp);
    FREAD(rnn_p->tau, rnn_p->c_state_size, fp);
    FREAD(rnn_p->eta, rnn_p->c_state_size, fp);
    FREAD(rnn_p->delta_threshold_c, rnn_p->c_state_size, fp);
    FREAD(rnn_p->delta_threshold_o, rnn_p->out_state_size, fp);
    FREAD(rnn_p->delta_threshold_v, rnn_p->out_state_size, fp);
    FREAD(rnn_p->delta_tau, rnn_p->c_state_size, fp);
    FREAD(rnn_p->prior_threshold_c, rnn_p->c_state_size, fp);
    FREAD(rnn_p->prior_threshold_o, rnn_p->out_state_size, fp);
    FREAD(rnn_p->prior_threshold_v, rnn_p->out_state_size, fp);
    FREAD(rnn_p->prior_tau, rnn_p->c_state_size, fp);
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        FREAD(rnn_p->rep_init_c[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->delta_rep_init_c[i], rnn_p->c_state_size, fp);
        FREAD(rnn_p->prior_rep_init_c[i], rnn_p->c_state_size, fp);
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        for (int j = 0; j <= rnn_p->in_state_size; j++) {
            FREAD(&rnn_p->connection_ci[i][j].begin, 1, fp);
            FREAD(&rnn_p->connection_ci[i][j].end, 1, fp);
        }
        for (int j = 0; j <= rnn_p->c_state_size; j++) {
            FREAD(&rnn_p->connection_cc[i][j].begin, 1, fp);
            FREAD(&rnn_p->connection_cc[i][j].end, 1, fp);
        }
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        for (int j = 0; j <= rnn_p->c_state_size; j++) {
            FREAD(&rnn_p->connection_oc[i][j].begin, 1, fp);
            FREAD(&rnn_p->connection_oc[i][j].end, 1, fp);
            FREAD(&rnn_p->connection_vc[i][j].begin, 1, fp);
            FREAD(&rnn_p->connection_vc[i][j].end, 1, fp);
        }
    }
}




void fwrite_rnn_state (
        const struct rnn_state *rnn_s,
        FILE *fp)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    FWRITE(&rnn_s->length, 1, fp);

    FWRITE(rnn_s->init_c_inter_state, rnn_p->c_state_size, fp);
    FWRITE(rnn_s->init_c_state, rnn_p->c_state_size, fp);
    FWRITE(rnn_s->delta_init_c_inter_state, rnn_p->c_state_size, fp);
    FWRITE(rnn_s->gate_init_c, rnn_p->rep_init_size, fp);
    FWRITE(rnn_s->beta_init_c, rnn_p->rep_init_size, fp);
    FWRITE(rnn_s->delta_beta_init_c, rnn_p->rep_init_size, fp);
    for (int n = 0; n < rnn_s->length; n++) {
        FWRITE(rnn_s->in_state[n], rnn_p->in_state_size, fp);
        FWRITE(rnn_s->teach_state[n], rnn_p->out_state_size, fp);
    }
}



void fread_rnn_state (
        struct rnn_state *rnn_s,
        FILE *fp)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    FREAD(&rnn_s->length, 1, fp);

    rnn_state_alloc(rnn_s);

    FREAD(rnn_s->init_c_inter_state, rnn_p->c_state_size, fp);
    FREAD(rnn_s->init_c_state, rnn_p->c_state_size, fp);
    FREAD(rnn_s->delta_init_c_inter_state, rnn_p->c_state_size, fp);
    FREAD(rnn_s->gate_init_c, rnn_p->rep_init_size, fp);
    FREAD(rnn_s->beta_init_c, rnn_p->rep_init_size, fp);
    FREAD(rnn_s->delta_beta_init_c, rnn_p->rep_init_size, fp);
    for (int n = 0; n < rnn_s->length; n++) {
        FREAD(rnn_s->in_state[n], rnn_p->in_state_size, fp);
        FREAD(rnn_s->teach_state[n], rnn_p->out_state_size, fp);
    }
}



void fwrite_recurrent_neural_network (
        const struct recurrent_neural_network *rnn,
        FILE *fp)
{
    fwrite_rnn_parameters(&rnn->rnn_p, fp);
    FWRITE(&rnn->series_num, 1, fp);
    for (int i = 0; i < rnn->series_num; i++) {
        fwrite_rnn_state(rnn->rnn_s + i, fp);
    }
}



void fread_recurrent_neural_network (
        struct recurrent_neural_network *rnn,
        FILE *fp)
{
    fread_rnn_parameters(&rnn->rnn_p, fp);
    FREAD(&rnn->series_num, 1, fp);
    MALLOC(rnn->rnn_s, rnn->series_num);
    for (int i = 0; i < rnn->series_num; i++) {
        rnn->rnn_s[i].rnn_p = &rnn->rnn_p;
        fread_rnn_state(rnn->rnn_s + i, fp);
    }
}




/******************************************************************************/
/********** Computation of RNN ************************************************/
/******************************************************************************/

void rnn_reset_delta_parameters (struct rnn_parameters *rnn_p)
{
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            rnn_p->delta_weight_ci[i][j] = 0;
        }
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            rnn_p->delta_weight_cc[i][j] = 0;
        }
        rnn_p->delta_threshold_c[i] = 0;
        rnn_p->delta_tau[i] = 0;
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            rnn_p->delta_weight_oc[i][j] = 0;
            rnn_p->delta_weight_vc[i][j] = 0;
        }
        rnn_p->delta_threshold_o[i] = 0;
        rnn_p->delta_threshold_v[i] = 0;
    }
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            rnn_p->delta_rep_init_c[i][j] = 0;
        }
    }
}

void rnn_reset_prior_distribution(struct rnn_parameters *rnn_p)
{
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        memcpy(rnn_p->prior_weight_ci[i], rnn_p->weight_ci[i], sizeof(double) *
                rnn_p->in_state_size);
        memcpy(rnn_p->prior_weight_cc[i], rnn_p->weight_cc[i], sizeof(double) *
                rnn_p->c_state_size);
    }
    memcpy(rnn_p->prior_threshold_c, rnn_p->threshold_c, sizeof(double) *
            rnn_p->c_state_size);
    memcpy(rnn_p->prior_tau, rnn_p->tau, sizeof(double) * rnn_p->c_state_size);
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        memcpy(rnn_p->prior_weight_oc[i], rnn_p->weight_oc[i], sizeof(double) *
                rnn_p->c_state_size);
        memcpy(rnn_p->prior_weight_vc[i], rnn_p->weight_vc[i], sizeof(double) *
                rnn_p->c_state_size);
    }
    memcpy(rnn_p->prior_threshold_o, rnn_p->threshold_o, sizeof(double) *
            rnn_p->out_state_size);
    memcpy(rnn_p->prior_threshold_v, rnn_p->threshold_v, sizeof(double) *
            rnn_p->out_state_size);
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        memcpy(rnn_p->prior_rep_init_c[i], rnn_p->rep_init_c[i],
                sizeof(double) * rnn_p->c_state_size);
    }
}


void rnn_get_connection (
        int size,
        const struct connection_domain *connection,
        int *has_connection)
{
    for (int i = 0; i < size; i++) {
        has_connection[i] = 0;
    }
    foreach (i, connection) {
        has_connection[i] = 1;
    }
}


void rnn_set_connection (
        int size,
        struct connection_domain *connection,
        const int *has_connection)
{
    int I, flg;
    I = flg = 0;
    for (int i = 0; i < size; i++) {
        if (!has_connection[i]) {
            if (flg == 1) {
                connection[I].end = i;
                I++;
                flg = 0;
            }
        } else {
            if (flg == 0) {
                connection[I].begin = i;
                flg = 1;
            }
        }
    }
    if (flg == 1) {
        connection[I].end = size;
        I++;
    }
    connection[I].begin = -1;
}

static void normalize_connection (
        int size,
        struct connection_domain *connection)
{
    int has_connection[size];
    rnn_get_connection(size, connection, has_connection);
    rnn_set_connection(size, connection, has_connection);
}

void rnn_add_connection (
        int size,
        struct connection_domain *connection,
        int begin,
        int end)
{
    int I = 0;
    while (connection[I].begin != -1) {
        I++;
    }
    if (I < size) {
        connection[I].begin = begin;
        connection[I].end = end;
        connection[I+1].begin = -1;
        normalize_connection(size, connection);
    }
}

void rnn_delete_connection (
        int size,
        struct connection_domain *connection,
        int begin,
        int end)
{
    int has_connection[size];
    rnn_get_connection(size, connection, has_connection);
    for (int i = begin; i < end; i++) {
        has_connection[i] = 0;
    }
    rnn_set_connection(size, connection, has_connection);
}

void rnn_reset_weight_by_connection (struct rnn_parameters *rnn_p)
{
    int size = (rnn_p->in_state_size > rnn_p->c_state_size) ?
        rnn_p->in_state_size : rnn_p->c_state_size;
    int has_connection[size];
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        rnn_get_connection(rnn_p->in_state_size, rnn_p->connection_ci[i],
                has_connection);
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            if (!has_connection[j]) {
                rnn_p->weight_ci[i][j] = 0;
            }
        }
        rnn_get_connection(rnn_p->c_state_size, rnn_p->connection_cc[i],
                has_connection);
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            if (!has_connection[j]) {
                rnn_p->weight_cc[i][j] = 0;
            }
        }
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        rnn_get_connection(rnn_p->c_state_size, rnn_p->connection_oc[i],
                has_connection);
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            if (!has_connection[j]) {
                rnn_p->weight_oc[i][j] = 0;
            }
        }
        rnn_get_connection(rnn_p->c_state_size, rnn_p->connection_vc[i],
                has_connection);
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            if (!has_connection[j]) {
                rnn_p->weight_vc[i][j] = 0;
            }
        }
    }
}


void rnn_set_uniform_tau (
        struct rnn_parameters *rnn_p,
        double tau)
{
    if (tau < 1) {
        tau = 1;
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        rnn_p->tau[i] = tau;
        rnn_p->eta[i] = 1.0 / tau;
    }
}

void rnn_set_tau (
        struct rnn_parameters *rnn_p,
        const double *tau)
{
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        rnn_p->tau[i] = tau[i];
        if (rnn_p->tau[i] < 1) {
            rnn_p->tau[i] = 1;
        }
        rnn_p->eta[i] = 1.0 / tau[i];
    }
}


int rnn_get_total_length (const struct recurrent_neural_network *rnn)
{
    int total_length = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        total_length += rnn->rnn_s[i].length;
    }
    return total_length;
}


static double get_error_for_standard (const struct rnn_state *rnn_s)
{
    double error = 0;
    for (int n = 0; n < rnn_s->length; n++) {
        for (int i = 0; i < rnn_s->rnn_p->out_state_size; i++) {
            double d = rnn_s->out_state[n][i] - rnn_s->teach_state[n][i];
            error += 0.5 * d * d;
        }
    }
    return error;
}

static double get_error_for_softmax (const struct rnn_state *rnn_s)
{
    double error = 0;
    for (int n = 0; n < rnn_s->length; n++) {
        for (int i = 0; i < rnn_s->rnn_p->out_state_size; i++) {
            double p = rnn_s->teach_state[n][i];
            double q = rnn_s->out_state[n][i];
            if (p > 0) {
                error += p * log(p/q);
            }
        }
    }
    return error;
}


double rnn_get_error (const struct rnn_state *rnn_s)
{
    double error;
    if (rnn_s->rnn_p->output_type == STANDARD_TYPE) {
        error = get_error_for_standard(rnn_s);
    } else if (rnn_s->rnn_p->output_type == SOFTMAX_TYPE){
        error = get_error_for_softmax(rnn_s);
    } else {
        error = 0;
    }
    return error;
}

double rnn_get_total_error (const struct recurrent_neural_network *rnn)
{
    double error[rnn->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        error[i] = rnn_get_error(rnn->rnn_s + i);
    }
    double total_error = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        total_error += error[i];
    }
    return total_error;
}


static double get_likelihood_for_standard (const struct rnn_state *rnn_s)
{
    double likelihood = 0;
    for (int n = 0; n < rnn_s->length; n++) {
        for (int i = 0; i < rnn_s->rnn_p->out_state_size; i++) {
            likelihood += rnn_s->likelihood[n][i];
            likelihood += -log(2 * M_PI * (rnn_s->var_state[n][i] +
                        MIN_VARIANCE));
        }
    }
    likelihood *= 0.5;
    return likelihood;
}

static double get_likelihood_for_softmax (const struct rnn_state *rnn_s)
{
    double likelihood = 0;
    for (int n = 0; n < rnn_s->length; n++) {
        for (int i = 0; i < rnn_s->rnn_p->out_state_size; i++) {
            likelihood += rnn_s->likelihood[n][i];
        }
    }
    return likelihood;
}

double rnn_get_likelihood (const struct rnn_state *rnn_s)
{
    double likelihood;
    if (rnn_s->rnn_p->output_type == STANDARD_TYPE) {
        likelihood = get_likelihood_for_standard(rnn_s);
    } else if (rnn_s->rnn_p->output_type == SOFTMAX_TYPE){
        likelihood = get_likelihood_for_softmax(rnn_s);
    } else {
        likelihood = 0;
    }
    return likelihood;
}

double rnn_get_total_likelihood (const struct recurrent_neural_network *rnn)
{
    double likelihood[rnn->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        likelihood[i] = rnn_get_likelihood(rnn->rnn_s + i);
    }
    double total_likelihood = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        total_likelihood += likelihood[i];
    }
    return total_likelihood;
}


static inline double fmap (
        const struct connection_domain * const restrict connection,
        const double * const restrict weight,
        const double * const restrict state,
        double sum)
{
    foreach (i, connection) {
        sum += weight[i] * state[i];
    }
    return sum;
}

void rnn_forward_context_map (
        const struct rnn_parameters *rnn_p,
        const double *in_state,
        const double *prev_c_inter_state,
        const double *prev_c_state,
        double *c_inputsum,
        double *c_inter_state,
        double *c_state)
{
    const int c_state_size = rnn_p->c_state_size;
    for (int i = 0; i < c_state_size; i++) {
        c_inputsum[i] = fmap(rnn_p->connection_ci[i], rnn_p->weight_ci[i],
                in_state, rnn_p->threshold_c[i]);
        c_inputsum[i] = fmap(rnn_p->connection_cc[i], rnn_p->weight_cc[i],
                prev_c_state, c_inputsum[i]);
        c_inter_state[i] = (1 - rnn_p->eta[i]) * prev_c_inter_state[i] +
            rnn_p->eta[i] * c_inputsum[i];
        c_state[i] = tanh(c_inter_state[i]);
    }
}


static void forward_output_map_for_standard (
        const struct rnn_parameters *rnn_p,
        const double *c_state,
        double *o_inter_state,
        double *out_state,
        double *v_inter_state,
        double *var_state)
{
    const int out_state_size = rnn_p->out_state_size;
    for (int i = 0; i < out_state_size; i++) {
        o_inter_state[i] = fmap(rnn_p->connection_oc[i], rnn_p->weight_oc[i],
                c_state, rnn_p->threshold_o[i]);
        v_inter_state[i] = fmap(rnn_p->connection_vc[i], rnn_p->weight_vc[i],
                c_state, rnn_p->threshold_v[i]);
        out_state[i] = tanh(o_inter_state[i]);
        var_state[i] = exp(v_inter_state[i]);
    }
}

static void forward_output_map_for_softmax (
        const struct rnn_parameters *rnn_p,
        const double *c_state,
        double *o_inter_state,
        double *out_state)
{
    const int out_state_size = rnn_p->out_state_size;
    const int softmax_group_num = rnn_p->softmax_group_num;
    double sum[softmax_group_num];

    for (int i = 0; i < out_state_size; i++) {
        o_inter_state[i] = fmap(rnn_p->connection_oc[i], rnn_p->weight_oc[i],
                c_state, rnn_p->threshold_o[i]);
        out_state[i] = exp(o_inter_state[i]);
    }
    for (int c = 0; c < softmax_group_num; c++) {
        sum[c] = 0;
    }
    for (int i = 0; i < out_state_size; i++) {
        sum[rnn_p->softmax_group_id[i]] += out_state[i];
    }
    for (int i = 0; i < out_state_size; i++) {
        out_state[i] /= sum[rnn_p->softmax_group_id[i]];
    }
}

void rnn_forward_output_map (
        const struct rnn_parameters *rnn_p,
        const double *c_state,
        double *o_inter_state,
        double *out_state,
        double *v_inter_state,
        double *var_state)
{
    if (rnn_p->output_type == STANDARD_TYPE) {
        forward_output_map_for_standard(rnn_p, c_state, o_inter_state,
                out_state, v_inter_state, var_state);
    } else if (rnn_p->output_type == SOFTMAX_TYPE) {
        forward_output_map_for_softmax(rnn_p, c_state, o_inter_state,
                out_state);
    }
}



void rnn_forward_map (
        const struct rnn_parameters *rnn_p,
        const double *in_state,
        const double *prev_c_inter_state,
        const double *prev_c_state,
        double *c_inputsum,
        double *c_inter_state,
        double *c_state,
        double *o_inter_state,
        double *out_state,
        double *v_inter_state,
        double *var_state)
{
    rnn_forward_context_map(rnn_p, in_state, prev_c_inter_state, prev_c_state,
            c_inputsum, c_inter_state, c_state);
    rnn_forward_output_map(rnn_p, c_state, o_inter_state, out_state,
            v_inter_state, var_state);
}


void rnn_forward_dynamics (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;

    if (rnn_s->length <= 0) return;

    rnn_forward_map(rnn_p, rnn_s->in_state[0], rnn_s->init_c_inter_state,
            rnn_s->init_c_state, rnn_s->c_inputsum[0], rnn_s->c_inter_state[0],
            rnn_s->c_state[0], rnn_s->o_inter_state[0], rnn_s->out_state[0],
            rnn_s->v_inter_state[0], rnn_s->var_state[0]);

    for (int n = 1; n < rnn_s->length; n++) {
        rnn_forward_map(rnn_p, rnn_s->in_state[n], rnn_s->c_inter_state[n-1],
                rnn_s->c_state[n-1], rnn_s->c_inputsum[n],
                rnn_s->c_inter_state[n], rnn_s->c_state[n],
                rnn_s->o_inter_state[n], rnn_s->out_state[n],
                rnn_s->v_inter_state[n], rnn_s->var_state[n]);
    }
}


void rnn_forward_dynamics_in_closed_loop (
        struct rnn_state *rnn_s,
        int delay_length)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;

    assert(rnn_s->length > 0);
    assert(rnn_p->in_state_size <= rnn_p->out_state_size);

    rnn_forward_map(rnn_p, rnn_s->in_state[0], rnn_s->init_c_inter_state,
            rnn_s->init_c_state, rnn_s->c_inputsum[0], rnn_s->c_inter_state[0],
            rnn_s->c_state[0], rnn_s->o_inter_state[0], rnn_s->out_state[0],
            rnn_s->v_inter_state[0], rnn_s->var_state[0]);

    for (int n = 1; n < delay_length && n < rnn_s->length; n++) {
        rnn_forward_map(rnn_p, rnn_s->in_state[n], rnn_s->c_inter_state[n-1],
                rnn_s->c_state[n-1], rnn_s->c_inputsum[n],
                rnn_s->c_inter_state[n], rnn_s->c_state[n],
                rnn_s->o_inter_state[n], rnn_s->out_state[n],
                rnn_s->v_inter_state[n], rnn_s->var_state[n]);
    }
    for (int n = delay_length; n < rnn_s->length; n++) {
        rnn_forward_map(rnn_p, rnn_s->out_state[n-delay_length],
                rnn_s->c_inter_state[n-1], rnn_s->c_state[n-1],
                rnn_s->c_inputsum[n], rnn_s->c_inter_state[n],
                rnn_s->c_state[n], rnn_s->o_inter_state[n],
                rnn_s->out_state[n], rnn_s->v_inter_state[n],
                rnn_s->var_state[n]);
    }
}


void rnn_forward_dynamics_forall (struct recurrent_neural_network *rnn)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_forward_dynamics(rnn->rnn_s + i);
    }
}


void rnn_forward_dynamics_in_closed_loop_forall (
        struct recurrent_neural_network *rnn,
        int delay_length)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_forward_dynamics_in_closed_loop(rnn->rnn_s + i, delay_length);
    }
}



static void set_likelihood_for_standard (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    for (int n = 0; n < rnn_s->length; n++) {
        for (int i = 0; i < rnn_p->out_state_size; i++) {
            double d = rnn_s->teach_state[n][i] - rnn_s->out_state[n][i];
            double s = 1.0 / (rnn_s->var_state[n][i] + MIN_VARIANCE);
            rnn_s->delta_likelihood[n][i] = d * s;
            rnn_s->likelihood[n][i] = -d * d * s;
        }
    }
}


static void set_likelihood_for_softmax (struct rnn_state *rnn_s)
{
    double p, q;
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;

    for (int n = 0; n < rnn_s->length; n++) {
        for (int i = 0; i < rnn_p->out_state_size; i++) {
            p = rnn_s->teach_state[n][i];
            q = rnn_s->out_state[n][i];
            rnn_s->delta_likelihood[n][i] = p/q;
            rnn_s->likelihood[n][i] = p * log(q);
        }
    }
}

void rnn_set_likelihood (struct rnn_state *rnn_s)
{
    if (rnn_s->rnn_p->output_type == STANDARD_TYPE) {
        set_likelihood_for_standard(rnn_s);
    } else if (rnn_s->rnn_p->output_type == SOFTMAX_TYPE) {
        set_likelihood_for_softmax(rnn_s);
    }
}


static void backward_output_map_for_standard (
        const struct rnn_parameters *rnn_p,
        const double *delta_likelihood,
        const double *out_state,
        const double *var_state,
        double *delta_o_inter,
        double *delta_v_inter)
{
    const int out_state_size = rnn_p->out_state_size;
    for (int i = 0; i < out_state_size; i++) {
        double dtanh_o = 1.0 - (out_state[i] * out_state[i]);
        delta_o_inter[i] = delta_likelihood[i] * dtanh_o;
        double dl2 = delta_likelihood[i] * delta_likelihood[i];
        double s = 1.0 / (var_state[i] + MIN_VARIANCE);
        delta_v_inter[i] = 0.5 * (-s + dl2) * var_state[i];
    }
}

static void backward_output_map_for_softmax (
        const struct rnn_parameters *rnn_p,
        const double *delta_likelihood,
        const double *out_state,
        double *delta_o_inter)
{
    const int out_state_size = rnn_p->out_state_size;
    const int softmax_group_num = rnn_p->softmax_group_num;
    double sum[softmax_group_num];

    for (int c = 0; c < softmax_group_num; c++) {
        sum[c] = 0;
    }
    for (int i = 0; i < out_state_size; i++) {
        sum[rnn_p->softmax_group_id[i]] += delta_likelihood[i] * out_state[i];
    }
    for (int i = 0; i < out_state_size; i++) {
        delta_o_inter[i] = out_state[i] * (delta_likelihood[i] -
                sum[rnn_p->softmax_group_id[i]]);
    }
}

void rnn_backward_output_map (
        const struct rnn_parameters *rnn_p,
        const double *delta_likelihood,
        const double *out_state,
        const double *var_state,
        double *delta_o_inter,
        double *delta_v_inter)
{
    if (rnn_p->output_type == STANDARD_TYPE) {
        backward_output_map_for_standard(rnn_p, delta_likelihood, out_state,
                var_state, delta_o_inter, delta_v_inter);
    } else if (rnn_p->output_type == SOFTMAX_TYPE) {
        backward_output_map_for_softmax(rnn_p, delta_likelihood, out_state,
                delta_o_inter);
    }
}


static inline void bmap (
        const struct connection_domain * const restrict connection,
        const double * const restrict weight,
        const double * const restrict df,
        const double delta,
        double * const restrict sum)
{
    foreach (i, connection) {
        sum[i] += (delta * weight[i]) * df[i];
    }
}

void rnn_backward_context_map (
        const struct rnn_parameters *rnn_p,
        const double *delta_o_inter,
        const double *delta_v_inter,
        const double *next_delta_c_inter,
        const double *c_state,
        double *delta_c_inter)
{
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    double dtanh_c[c_state_size];

    for (int i = 0; i < c_state_size; i++) {
        delta_c_inter[i] = 0;
        dtanh_c[i] = 1.0 - (c_state[i] * c_state[i]);
    }
    if (next_delta_c_inter != NULL) {
        for (int i = 0; i < c_state_size; i++) {
            double delta = next_delta_c_inter[i] * rnn_p->eta[i];
            bmap(rnn_p->connection_cc[i], rnn_p->weight_cc[i], dtanh_c, delta,
                    delta_c_inter);
            delta_c_inter[i] += next_delta_c_inter[i] * (1 - rnn_p->eta[i]);
        }
    }

    for (int i = 0; i < out_state_size; i++) {
        bmap(rnn_p->connection_oc[i], rnn_p->weight_oc[i], dtanh_c,
                delta_o_inter[i], delta_c_inter);
        if (rnn_p->output_type == STANDARD_TYPE) {
            bmap(rnn_p->connection_vc[i], rnn_p->weight_vc[i], dtanh_c,
                    delta_v_inter[i], delta_c_inter);
        }
    }
}


void rnn_backward_map (
        const struct rnn_parameters *rnn_p,
        const double *delta_likelihood,
        const double *next_delta_c_inter,
        const double *c_state,
        const double *out_state,
        const double *var_state,
        double *delta_c_inter,
        double *delta_o_inter,
        double *delta_v_inter)
{
    rnn_backward_output_map(rnn_p, delta_likelihood, out_state, var_state,
            delta_o_inter, delta_v_inter);
    rnn_backward_context_map(rnn_p, delta_o_inter, delta_v_inter,
            next_delta_c_inter, c_state, delta_c_inter);
}



void rnn_backward_dynamics (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;

    rnn_backward_map(rnn_p, rnn_s->delta_likelihood[rnn_s->length-1], NULL,
            rnn_s->c_state[rnn_s->length-1], rnn_s->out_state[rnn_s->length-1],
            rnn_s->var_state[rnn_s->length-1],
            rnn_s->delta_c_inter[rnn_s->length-1],
            rnn_s->delta_o_inter[rnn_s->length-1],
            rnn_s->delta_v_inter[rnn_s->length-1]);

    for (int n = rnn_s->length-2; n >= 0; n--) {
        rnn_backward_map(rnn_p, rnn_s->delta_likelihood[n],
                rnn_s->delta_c_inter[n+1], rnn_s->c_state[n],
                rnn_s->out_state[n], rnn_s->var_state[n],
                rnn_s->delta_c_inter[n], rnn_s->delta_o_inter[n],
                rnn_s->delta_v_inter[n]);
    }

    rnn_set_delta_parameters(rnn_s);
}

void rnn_forward_backward_dynamics (struct rnn_state *rnn_s)
{
    rnn_forward_dynamics(rnn_s);
    rnn_set_likelihood(rnn_s);
    rnn_backward_dynamics(rnn_s);
}

void rnn_forward_backward_dynamics_forall (struct recurrent_neural_network *rnn)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_forward_backward_dynamics(rnn->rnn_s + i);
    }
}


static inline void smap (
        const struct connection_domain * const restrict connection,
        const double * const restrict state,
        const double delta,
        double * const restrict sum)
{
    foreach (i, connection) {
        sum[i] += delta * state[i];
    }
}

void rnn_set_delta_w (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    const int length = rnn_s->length;

    for (int i = 0; i < c_state_size; i++) {
        foreach (j, rnn_p->connection_ci[i]) {
            rnn_s->delta_w_ci[i][j] = 0;
        }
        foreach (j, rnn_p->connection_cc[i]) {
            rnn_s->delta_w_cc[i][j] = 0;
        }
    }
    for (int i = 0; i < out_state_size; i++) {
        foreach (j, rnn_p->connection_oc[i]) {
            rnn_s->delta_w_oc[i][j] = 0;
        }
        foreach (j, rnn_p->connection_vc[i]) {
            rnn_s->delta_w_vc[i][j] = 0;
        }
    }

    for (int n = 0; n < length; n++) {
        double *state = (n == 0) ? rnn_s->init_c_state : rnn_s->c_state[n-1];
        for (int i = 0; i < c_state_size; i++) {
            double delta = rnn_s->delta_c_inter[n][i] * rnn_p->eta[i];
            smap(rnn_p->connection_ci[i], rnn_s->in_state[n], delta,
                    rnn_s->delta_w_ci[i]);
            smap(rnn_p->connection_cc[i], state, delta, rnn_s->delta_w_cc[i]);
        }
        for (int i = 0; i < out_state_size; i++) {
            smap(rnn_p->connection_oc[i], rnn_s->c_state[n],
                    rnn_s->delta_o_inter[n][i], rnn_s->delta_w_oc[i]);
            smap(rnn_p->connection_vc[i], rnn_s->c_state[n],
                    rnn_s->delta_v_inter[n][i], rnn_s->delta_w_vc[i]);
        }
    }
}


void rnn_set_delta_t (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    const int length = rnn_s->length;

    for (int i = 0; i < c_state_size; i++) {
        double sum = 0;
        for (int n = 0; n < length; n++) {
            sum += rnn_s->delta_c_inter[n][i];
        }
        rnn_s->delta_t_c[i] = sum * rnn_p->eta[i];
    }
    for (int i = 0; i < out_state_size; i++) {
        double sum = 0;
        for (int n = 0; n < length; n++) {
            sum += rnn_s->delta_o_inter[n][i];
        }
        rnn_s->delta_t_o[i] = sum;
        sum = 0;
        for (int n = 0; n < length; n++) {
            sum += rnn_s->delta_v_inter[n][i];
        }
        rnn_s->delta_t_v[i] = sum;
    }
}

void rnn_set_delta_tau (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    const int c_state_size = rnn_p->c_state_size;
    const int length = rnn_s->length;

    for (int i = 0; i < c_state_size; i++) {
        double sum = 0;
        for (int n = 0; n < length; n++) {
            if (n == 0) {
                sum += rnn_s->delta_c_inter[n][i] *
                    (rnn_s->init_c_inter_state[i] - rnn_s->c_inputsum[n][i]);
            } else {
                sum += rnn_s->delta_c_inter[n][i] *
                    (rnn_s->c_inter_state[n-1][i] - rnn_s->c_inputsum[n][i]);
            }
        }
        rnn_s->delta_tau[i] = sum * (rnn_p->eta[i] * rnn_p->eta[i]);
    }
}

void rnn_set_delta_i (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    const int c_state_size = rnn_p->c_state_size;
    for (int i = 0; i < c_state_size; i++) {
        rnn_s->delta_i[i] = 0;
    }
    for (int i = 0; i < c_state_size; i++) {
        double d = rnn_s->delta_c_inter[0][i] * rnn_p->eta[i];
        foreach (j, rnn_p->connection_cc[i]) {
            rnn_s->delta_i[j] += d * rnn_p->weight_cc[i][j];
        }
    }
    for (int i = 0; i < c_state_size; i++) {
        double dtanh_c = 1.0 - (rnn_s->init_c_state[i] *
                rnn_s->init_c_state[i]);
        rnn_s->delta_i[i] *= dtanh_c;
        rnn_s->delta_i[i] += rnn_s->delta_c_inter[0][i] * (1 - rnn_p->eta[i]);
#ifdef ENABLE_ATTRACTION_OF_INIT_C
        const int length = rnn_s->length;
        double mean, var;
        mean = rnn_s->init_c_inter_state[i];
        for (int n = 0; n < length; n++) {
            mean += rnn_s->c_inter_state[n][i];
        }
        mean /= length + 1;
        double d = mean - rnn_s->init_c_inter_state[i];
        var = d * d;
        for (int n = 0; n < length; n++) {
            d = mean - rnn_s->c_inter_state[n][i];
            var += d * d;
        }
        var /= length + 1;
        if (var < MIN_VARIANCE) {
            var = MIN_VARIANCE;
        }
        rnn_s->delta_i[i] += (mean - rnn_s->init_c_inter_state[i]) / var;
#endif
    }
}

void rnn_set_delta_b (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        double sum = 0;
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            double d = rnn_s->init_c_inter_state[j] - rnn_p->rep_init_c[i][j];
            sum += d * d;
        }
        rnn_s->delta_b[i] = exp((-sum) / (2 * rnn_p->rep_init_variance)) +
            DBL_MIN;
    }
}

void rnn_set_delta_parameters (struct rnn_state *rnn_s)
{
    if (!rnn_s->rnn_p->fixed_weight) {
        rnn_set_delta_w(rnn_s);
    }
    if (!rnn_s->rnn_p->fixed_threshold) {
        rnn_set_delta_t(rnn_s);
    }
    if (!rnn_s->rnn_p->fixed_tau) {
        rnn_set_delta_tau(rnn_s);
    }
    if (!rnn_s->rnn_p->fixed_init_c_state) {
        rnn_set_delta_i(rnn_s);
        rnn_set_delta_b(rnn_s);
    }
}


void rnn_update_delta_weight (
        struct recurrent_neural_network *rnn,
        double momentum)
{
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        foreach (j, rnn->rnn_p.connection_ci[i]) {
            double delta = 0;
            for (int k = 0; k < rnn->series_num; k++) {
                delta += rnn->rnn_s[k].delta_w_ci[i][j];
            }
            delta += rnn->rnn_p.prior_strength *
                (rnn->rnn_p.prior_weight_ci[i][j] - rnn->rnn_p.weight_ci[i][j]);
            rnn->rnn_p.delta_weight_ci[i][j] = delta + momentum *
                rnn->rnn_p.delta_weight_ci[i][j];
        }
        foreach (j, rnn->rnn_p.connection_cc[i]) {
            double delta = 0;
            for (int k = 0; k < rnn->series_num; k++) {
                delta += rnn->rnn_s[k].delta_w_cc[i][j];
            }
            delta += rnn->rnn_p.prior_strength *
                (rnn->rnn_p.prior_weight_cc[i][j] - rnn->rnn_p.weight_cc[i][j]);
            rnn->rnn_p.delta_weight_cc[i][j] = delta + momentum *
                rnn->rnn_p.delta_weight_cc[i][j];
        }
    }
    for (int i = 0; i < rnn->rnn_p.out_state_size; i++) {
        foreach (j, rnn->rnn_p.connection_oc[i]) {
            double delta = 0;
            for (int k = 0; k < rnn->series_num; k++) {
                delta += rnn->rnn_s[k].delta_w_oc[i][j];
            }
            delta += rnn->rnn_p.prior_strength *
                (rnn->rnn_p.prior_weight_oc[i][j] - rnn->rnn_p.weight_oc[i][j]);
            rnn->rnn_p.delta_weight_oc[i][j] = delta + momentum *
                rnn->rnn_p.delta_weight_oc[i][j];
        }
        foreach (j, rnn->rnn_p.connection_vc[i]) {
            double delta = 0;
            for (int k = 0; k < rnn->series_num; k++) {
                delta += rnn->rnn_s[k].delta_w_vc[i][j];
            }
            delta += rnn->rnn_p.prior_strength *
                (rnn->rnn_p.prior_weight_vc[i][j] - rnn->rnn_p.weight_vc[i][j]);
            rnn->rnn_p.delta_weight_vc[i][j] = delta + momentum *
                rnn->rnn_p.delta_weight_vc[i][j];
        }
    }
}



void rnn_update_delta_threshold (
        struct recurrent_neural_network *rnn,
        double momentum)
{
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        double delta = 0;
        for (int j = 0; j < rnn->series_num; j++) {
            delta += rnn->rnn_s[j].delta_t_c[i];
        }
        delta += rnn->rnn_p.prior_strength *
            (rnn->rnn_p.prior_threshold_c[i] - rnn->rnn_p.threshold_c[i]);
        rnn->rnn_p.delta_threshold_c[i] = delta + momentum *
            rnn->rnn_p.delta_threshold_c[i];
    }
    for (int i = 0; i < rnn->rnn_p.out_state_size; i++) {
        double delta = 0;
        for (int j = 0; j < rnn->series_num; j++) {
            delta += rnn->rnn_s[j].delta_t_o[i];
        }
        delta += rnn->rnn_p.prior_strength *
            (rnn->rnn_p.prior_threshold_o[i] - rnn->rnn_p.threshold_o[i]);
        rnn->rnn_p.delta_threshold_o[i] = delta + momentum *
            rnn->rnn_p.delta_threshold_o[i];
        delta = 0;
        for (int j = 0; j < rnn->series_num; j++) {
            delta += rnn->rnn_s[j].delta_t_v[i];
        }
        delta += rnn->rnn_p.prior_strength *
            (rnn->rnn_p.prior_threshold_v[i] - rnn->rnn_p.threshold_v[i]);
        rnn->rnn_p.delta_threshold_v[i] = delta + momentum *
            rnn->rnn_p.delta_threshold_v[i];
    }
}

void rnn_update_delta_tau (
        struct recurrent_neural_network *rnn,
        double momentum)
{
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        double delta = 0;
        for (int j = 0; j < rnn->series_num; j++) {
            delta += rnn->rnn_s[j].delta_tau[i];
        }
        delta += rnn->rnn_p.prior_strength * (rnn->rnn_p.prior_tau[i] -
                rnn->rnn_p.tau[i]);
        rnn->rnn_p.delta_tau[i] = delta + momentum * rnn->rnn_p.delta_tau[i];
    }
}

void rnn_update_delta_rep_init_c (
        struct recurrent_neural_network *rnn,
        double momentum)
{
    double p[rnn->series_num];
    for (int i = 0; i < rnn->series_num; i++) {
        p[i] = 0;
        for (int j = 0; j < rnn->rnn_p.rep_init_size; j++) {
            p[i] += rnn->rnn_s[i].gate_init_c[j] * rnn->rnn_s[i].delta_b[j];
        }
    }
    for (int i = 0; i < rnn->rnn_p.rep_init_size; i++) {
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            double delta = 0;
            for (int k = 0; k < rnn->series_num; k++) {
                double d = rnn->rnn_s[k].init_c_inter_state[j] -
                    rnn->rnn_p.rep_init_c[i][j];
                delta += (rnn->rnn_s[k].gate_init_c[i] *
                        rnn->rnn_s[k].delta_b[i] * d) / p[k];
            }
            delta /= rnn->rnn_p.rep_init_variance;
            delta += (rnn->rnn_p.prior_rep_init_c[i][j] -
                    rnn->rnn_p.rep_init_c[i][j]) * rnn->rnn_p.prior_strength;
            rnn->rnn_p.delta_rep_init_c[i][j] = delta + momentum *
                rnn->rnn_p.delta_rep_init_c[i][j];
        }
    }
}

void rnn_update_delta_init_c_inter_state (
        struct rnn_state *rnn_s,
        double momentum)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    double p = 0;
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        p += rnn_s->gate_init_c[i] * rnn_s->delta_b[i];
    }
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        double d = (rnn_s->gate_init_c[i] / p) * (rnn_s->delta_b[i] - p);
        rnn_s->delta_beta_init_c[i] = d + momentum *
            rnn_s->delta_beta_init_c[i];
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        double delta = 0;
        for (int j = 0; j < rnn_p->rep_init_size; j++) {
            double d = rnn_p->rep_init_c[j][i] - rnn_s->init_c_inter_state[i];
            delta += d * rnn_s->gate_init_c[j] * rnn_s->delta_b[j];
        }
        delta /= p * rnn_p->rep_init_variance;
        delta += rnn_s->delta_i[i];
        rnn_s->delta_init_c_inter_state[i] = delta + momentum *
            rnn_s->delta_init_c_inter_state[i];
    }
}



void rnn_update_weight (
        struct rnn_parameters *rnn_p,
        double rho)
{
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        foreach (j, rnn_p->connection_ci[i]) {
            rnn_p->weight_ci[i][j] += rho * rnn_p->delta_weight_ci[i][j];
            assert(isfinite(rnn_p->weight_ci[i][j]));
        }
        foreach (j, rnn_p->connection_cc[i]) {
            rnn_p->weight_cc[i][j] += rho * rnn_p->delta_weight_cc[i][j];
            assert(isfinite(rnn_p->weight_cc[i][j]));
        }
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        foreach (j, rnn_p->connection_oc[i]) {
            rnn_p->weight_oc[i][j] += rho * rnn_p->delta_weight_oc[i][j];
            assert(isfinite(rnn_p->weight_oc[i][j]));
        }
        foreach (j, rnn_p->connection_vc[i]) {
            rnn_p->weight_vc[i][j] += rho * rnn_p->delta_weight_vc[i][j];
            assert(isfinite(rnn_p->weight_vc[i][j]));
        }
    }
}


void rnn_update_threshold (
        struct rnn_parameters *rnn_p,
        double rho)
{
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        rnn_p->threshold_c[i] += rho * rnn_p->delta_threshold_c[i];
        assert(isfinite(rnn_p->threshold_c[i]));
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        rnn_p->threshold_o[i] += rho * rnn_p->delta_threshold_o[i];
        assert(isfinite(rnn_p->threshold_o[i]));
        rnn_p->threshold_v[i] += rho * rnn_p->delta_threshold_v[i];
        assert(isfinite(rnn_p->threshold_v[i]));
    }
}

void rnn_update_tau (
        struct rnn_parameters *rnn_p,
        double rho)
{
    double new_tau;

    if (rho <= 0) return;

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        if (isfinite(rnn_p->tau[i])) {
            new_tau = rnn_p->tau[i] + rho * rnn_p->delta_tau[i];
            if (new_tau < 1) {
                new_tau = 1.0;
            }
            rnn_p->tau[i] = new_tau;
            rnn_p->eta[i] = 1.0/rnn_p->tau[i];
            assert(isfinite(rnn_p->tau[i]));
        }
    }
}


void rnn_update_rep_init_c (
        struct rnn_parameters *rnn_p,
        double rho)
{
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            rnn_p->rep_init_c[i][j] += rho * rnn_p->delta_rep_init_c[i][j];
            assert(isfinite(rnn_p->rep_init_c[i][j]));
        }
    }
}


void rnn_update_init_c_inter_state (
        struct rnn_state *rnn_s,
        double rho)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;
    double e[rnn_p->rep_init_size];
    double sum = 0;
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        rnn_s->beta_init_c[i] += rho * rnn_s->delta_beta_init_c[i];
        assert(isfinite(rnn_s->beta_init_c[i]));
        e[i] = exp(rnn_s->beta_init_c[i]);
        sum += e[i];
    }
    for (int i = 0; i < rnn_p->rep_init_size; i++) {
        rnn_s->gate_init_c[i] = e[i] / sum;
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        if (!rnn_p->const_init_c[i]) {
            rnn_s->init_c_inter_state[i] += rho *
                rnn_s->delta_init_c_inter_state[i];
            rnn_s->init_c_state[i] = tanh(rnn_s->init_c_inter_state[i]);
            assert(isfinite(rnn_s->init_c_inter_state[i]));
        }
    }
}


void rnn_update_delta_parameters (
        struct recurrent_neural_network *rnn,
        double momentum)
{
    if (!rnn->rnn_p.fixed_weight) {
        rnn_update_delta_weight(rnn, momentum);
    }
    if (!rnn->rnn_p.fixed_threshold) {
        rnn_update_delta_threshold(rnn, momentum);
    }
    if (!rnn->rnn_p.fixed_tau) {
        rnn_update_delta_tau(rnn, momentum);
    }
    if (!rnn->rnn_p.fixed_init_c_state) {
        rnn_update_delta_rep_init_c(rnn, momentum);
        for (int i = 0; i < rnn->series_num; i++) {
            rnn_update_delta_init_c_inter_state(rnn->rnn_s + i, momentum);
        }
    }
}



void rnn_update_parameters (
        struct recurrent_neural_network *rnn,
        double rho_weight,
        double rho_tau,
        double rho_init)
{
    if (!rnn->rnn_p.fixed_weight) {
        rnn_update_weight(&rnn->rnn_p, rho_weight);
    }
    if (!rnn->rnn_p.fixed_threshold) {
        rnn_update_threshold(&rnn->rnn_p, rho_weight);
    }
    if (!rnn->rnn_p.fixed_tau) {
        rnn_update_tau(&rnn->rnn_p, rho_tau);
    }
    if (!rnn->rnn_p.fixed_init_c_state) {
        rnn_update_rep_init_c(&rnn->rnn_p, rho_init);
        for (int i = 0; i < rnn->series_num; i++) {
            rnn_update_init_c_inter_state(rnn->rnn_s + i, rho_init);
        }
    }
}


/*
 * This function computes learning of a recurrent neural network
 *
 *   @parameter  rnn        : recurrent neural network
 *   @parameter  rho_weight : learning rate for weights and thresholds
 *   @parameter  rho_tau    : learning rate for tau
 *   @parameter  rho_init   : learning rate for initial states
 *   @parameter  momentum   : momentum of learning
 */
void rnn_learn (
        struct recurrent_neural_network *rnn,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double momentum)
{
    rnn_forward_backward_dynamics_forall(rnn);

    rnn_update_delta_parameters(rnn, momentum);
    rnn_update_parameters(rnn, rho_weight, rho_tau, rho_init);
}


/*
 * This function computes learning of a recurrent neural network
 * (support automatic scaling of learning rate)
 *
 *   @parameter  rnn        : recurrent neural network
 *   @parameter  rho        : learning rate
 *   @parameter  momentum   : momentum of learning
 */
void rnn_learn_s (
        struct recurrent_neural_network *rnn,
        double rho,
        double momentum)
{
    double r = 1.0 / (rnn_get_total_length(rnn) * rnn->rnn_p.out_state_size);
    double rho_weight = r * rho;
    double rho_tau = r * rho;
    double rho_init = rho / rnn->rnn_p.out_state_size;
    rnn_learn(rnn, rho_weight, rho_tau, rho_init, momentum);
}


#ifdef ENABLE_ADAPTIVE_LEARNING_RATE

void rnn_backup_learning_parameters (struct recurrent_neural_network *rnn)
{
    struct rnn_parameters *rnn_p = &rnn->rnn_p;
    const int in_state_size = rnn_p->in_state_size;
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    const int rep_init_size = rnn_p->rep_init_size;
    const int series_num = rnn->series_num;

    memmove(rnn_p->tmp_weight_ci, rnn_p->weight_ci[0], sizeof(double) *
            c_state_size * in_state_size);
    memmove(rnn_p->tmp_weight_cc, rnn_p->weight_cc[0], sizeof(double) *
            c_state_size * c_state_size);
    memmove(rnn_p->tmp_weight_oc, rnn_p->weight_oc[0], sizeof(double) *
            out_state_size * c_state_size);
    memmove(rnn_p->tmp_weight_vc, rnn_p->weight_vc[0], sizeof(double) *
            out_state_size * c_state_size);
    memmove(rnn_p->tmp_threshold_c, rnn_p->threshold_c, sizeof(double) *
            c_state_size);
    memmove(rnn_p->tmp_threshold_o, rnn_p->threshold_o, sizeof(double) *
            out_state_size);
    memmove(rnn_p->tmp_threshold_v, rnn_p->threshold_v, sizeof(double) *
            out_state_size);
    memmove(rnn_p->tmp_tau, rnn_p->tau, sizeof(double) * c_state_size);
    memmove(rnn_p->tmp_eta, rnn_p->eta, sizeof(double) * c_state_size);
    memmove(rnn_p->tmp_rep_init_c, rnn_p->rep_init_c[0], sizeof(double) *
            rep_init_size * c_state_size);

    for (int i = 0; i < series_num; i++) {
        struct rnn_state *rnn_s = rnn->rnn_s + i;
        memmove(rnn_s->tmp_init_c_inter_state, rnn_s->init_c_inter_state,
                sizeof(double) * c_state_size);
        memmove(rnn_s->tmp_init_c_state, rnn_s->init_c_state,
                sizeof(double) * c_state_size);
        memmove(rnn_s->tmp_gate_init_c, rnn_s->gate_init_c, sizeof(double) *
                rep_init_size);
        memmove(rnn_s->tmp_beta_init_c, rnn_s->beta_init_c, sizeof(double) *
                rep_init_size);
    }
}


void rnn_restore_learning_parameters (struct recurrent_neural_network *rnn)
{
    struct rnn_parameters *rnn_p = &rnn->rnn_p;
    const int in_state_size = rnn_p->in_state_size;
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    const int rep_init_size = rnn_p->rep_init_size;
    const int series_num = rnn->series_num;

    memmove(rnn_p->weight_ci[0], rnn_p->tmp_weight_ci, sizeof(double) *
            c_state_size * in_state_size);
    memmove(rnn_p->weight_cc[0], rnn_p->tmp_weight_cc, sizeof(double) *
            c_state_size * c_state_size);
    memmove(rnn_p->weight_oc[0], rnn_p->tmp_weight_oc, sizeof(double) *
            out_state_size * c_state_size);
    memmove(rnn_p->weight_vc[0], rnn_p->tmp_weight_vc, sizeof(double) *
            out_state_size * c_state_size);
    memmove(rnn_p->threshold_c, rnn_p->tmp_threshold_c, sizeof(double) *
            c_state_size);
    memmove(rnn_p->threshold_o, rnn_p->tmp_threshold_o, sizeof(double) *
            out_state_size);
    memmove(rnn_p->threshold_v, rnn_p->tmp_threshold_v, sizeof(double) *
            out_state_size);
    memmove(rnn_p->tau, rnn_p->tmp_tau, sizeof(double) * c_state_size);
    memmove(rnn_p->eta, rnn_p->tmp_eta, sizeof(double) * c_state_size);
    memmove(rnn_p->rep_init_c[0], rnn_p->tmp_rep_init_c, sizeof(double) *
            rep_init_size * c_state_size);

    for (int i = 0; i < series_num; i++) {
        struct rnn_state *rnn_s = rnn->rnn_s + i;
        memmove(rnn_s->init_c_inter_state, rnn_s->tmp_init_c_inter_state,
                sizeof(double) * c_state_size);
        memmove(rnn_s->init_c_state, rnn_s->tmp_init_c_state,
                sizeof(double) * c_state_size);
        memmove(rnn_s->gate_init_c, rnn_s->tmp_gate_init_c, sizeof(double) *
                rep_init_size);
        memmove(rnn_s->beta_init_c, rnn_s->tmp_beta_init_c, sizeof(double) *
                rep_init_size);
    }
}


double rnn_update_parameters_with_adapt_lr (
        struct recurrent_neural_network *rnn,
        double adapt_lr,
        double rho_weight,
        double rho_tau,
        double rho_init)
{
    double current_error = rnn_get_total_error(rnn);
    rnn_backup_learning_parameters(rnn);

    for (int count = 0; count < MAX_ITERATION_IN_ADAPTIVE_LR; count++) {
        rnn_update_parameters(rnn, rho_weight * adapt_lr, rho_tau * adapt_lr,
                rho_init * adapt_lr);
        rnn_forward_dynamics_forall(rnn);
        double next_error = rnn_get_total_error(rnn);
        double rate = next_error / current_error;
        if (rate > MAX_PERF_INC || isnan(rate)) {
            rnn_restore_learning_parameters(rnn);
            adapt_lr *= LR_DEC;
        } else {
            if (rate < 1) {
                adapt_lr *= LR_INC;
            }
            break;
        }
    }
    return adapt_lr;
}


/*
 * This function computes learning of a recurrent neural network
 * (support adaptive learning rate)
 *
 *   @parameter  rnn        : recurrent neural network
 *   @parameter  adapt_lr   : adaptive learning rate
 *   @parameter  rho_weight : learning rate for weights and thresholds
 *   @parameter  rho_tau    : learning rate for tau
 *   @parameter  rho_init   : learning rate for initial states
 *   @parameter  momentum   : momentum of learning
 *
 *   @return                : adaptive learning rate
 */
double rnn_learn_with_adapt_lr (
        struct recurrent_neural_network *rnn,
        double adapt_lr,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double momentum)
{
    rnn_forward_backward_dynamics_forall(rnn);

    rnn_update_delta_parameters(rnn, momentum);
    return rnn_update_parameters_with_adapt_lr(rnn, adapt_lr, rho_weight,
            rho_tau, rho_init);
}


/*
 * This function computes learning of a recurrent neural network
 * (support adaptive learning rate and automatic scaling of learning rate)
 *
 *   @parameter  rnn        : recurrent neural network
 *   @parameter  adapt_lr   : adaptive learning rate
 *   @parameter  rho        : learning rate
 *   @parameter  momentum   : momentum of learning
 */
double rnn_learn_s_with_adapt_lr (
        struct recurrent_neural_network *rnn,
        double adapt_lr,
        double rho,
        double momentum)
{
    double r = 1.0 / (rnn_get_total_length(rnn) * rnn->rnn_p.out_state_size);
    double rho_weight = r * rho;
    double rho_tau = r * rho;
    double rho_init = rho / rnn->rnn_p.out_state_size;
    return rnn_learn_with_adapt_lr(rnn, adapt_lr, rho_weight, rho_tau, rho_init,
            momentum);
}

#endif // ENABLE_ADAPTIVE_LEARNING_RATE



static double** jacobian_matrix_for_standard (
        double** matrix,
        const struct rnn_parameters *rnn_p,
        const double *prev_c_state,
        const double *c_state,
        const double *out_state)
{
    const int in_state_size = rnn_p->in_state_size;
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    double dtanh_prev_c[c_state_size], dtanh_c[c_state_size];

    for (int i = 0; i < c_state_size; i++) {
        dtanh_prev_c[i] = 1.0 - (prev_c_state[i] * prev_c_state[i]);
        dtanh_c[i] = 1.0 - (c_state[i] * c_state[i]);
    }
    for (int i = 0, I = out_state_size; i < c_state_size; i++, I++) {
        for (int j = 0; j < in_state_size; j++) {
            matrix[I][j] = 0;
        }
        foreach (j, rnn_p->connection_ci[i]) {
            matrix[I][j] = rnn_p->eta[i] * rnn_p->weight_ci[i][j];
        }
        int J = in_state_size;
        for (int j = 0; j < c_state_size; j++) {
            matrix[I][J+j] = 0;
        }
        foreach (j, rnn_p->connection_cc[i]) {
            matrix[I][J+j] = rnn_p->eta[i] * rnn_p->weight_cc[i][j] *
                dtanh_prev_c[j];
        }
        matrix[I][J+i] += 1 - rnn_p->eta[i];
    }
    for (int i = 0, I = 0; i < out_state_size; i++, I++) {
        int J = 0;
        double dtanh_o = 1.0 - (out_state[i] * out_state[i]);
        for (int j = 0; j < in_state_size; j++, J++) {
            double sum = 0;
            foreach (k, rnn_p->connection_oc[i]) {
                sum += rnn_p->weight_oc[i][k] * dtanh_c[k] *
                    matrix[k+out_state_size][j];
            }
            matrix[I][J] = dtanh_o * sum;
        }
        for (int j = 0; j < c_state_size; j++, J++) {
            double sum = 0;
            foreach (k, rnn_p->connection_oc[i]) {
                sum += rnn_p->weight_oc[i][k] * dtanh_c[k] *
                    matrix[k+out_state_size][j+in_state_size];
            }
            matrix[I][J] = dtanh_o * sum;
        }
    }
    return matrix;
}


static double** jacobian_matrix_for_softmax (
        double** matrix,
        const struct rnn_parameters *rnn_p,
        const double *prev_c_state,
        const double *c_state,
        const double *out_state)
{
    const int in_state_size = rnn_p->in_state_size;
    const int c_state_size = rnn_p->c_state_size;
    const int out_state_size = rnn_p->out_state_size;
    double dtanh_prev_c[c_state_size], dtanh_c[c_state_size];

    for (int i = 0; i < c_state_size; i++) {
        dtanh_prev_c[i] = 1.0 - (prev_c_state[i] * prev_c_state[i]);
        dtanh_c[i] = 1.0 - (c_state[i] * c_state[i]);
    }
    for (int i = 0, I = out_state_size; i < c_state_size; i++, I++) {
        for (int j = 0; j < in_state_size; j++) {
            matrix[I][j] = 0;
        }
        foreach (j, rnn_p->connection_ci[i]) {
            matrix[I][j] = rnn_p->eta[i] * rnn_p->weight_ci[i][j];
        }
        int J = in_state_size;
        for (int j = 0; j < c_state_size; j++) {
            matrix[I][J+j] = 0;
        }
        foreach (j, rnn_p->connection_cc[i]) {
            matrix[I][J+j] = rnn_p->eta[i] * rnn_p->weight_cc[i][j] *
                dtanh_prev_c[j];
        }
        matrix[I][J+i] += 1 - rnn_p->eta[i];
    }
    for (int i = 0; i < out_state_size; i++) {
        for (int j = 0, e = in_state_size + c_state_size; j < e; j++) {
            matrix[i][j] = 0;
        }
    }
    for (int i = 0; i < out_state_size; i++) {
        int J = 0;
        for (int j = 0; j < in_state_size; j++, J++) {
            double sum = 0;
            foreach (k, rnn_p->connection_oc[i]) {
                sum += rnn_p->weight_oc[i][k] * dtanh_c[k] *
                    matrix[k+out_state_size][j];
            }
            for (int k = 0; k < out_state_size; k++) {
                if (i == k) {
                    matrix[k][J] += (out_state[k] - out_state[k] * out_state[i])
                        * sum;
                } else if (rnn_p->softmax_group_id[i] ==
                        rnn_p->softmax_group_id[k]) {
                    matrix[k][J] += -out_state[k] * out_state[i] * sum;
                }
            }
        }
        for (int j = 0; j < c_state_size; j++, J++) {
            double sum = 0;
            foreach (k, rnn_p->connection_oc[i]) {
                sum += rnn_p->weight_oc[i][k] * dtanh_c[k]
                    * matrix[k+out_state_size][j+in_state_size];
            }
            for (int k = 0; k < out_state_size; k++) {
                if (i == k) {
                    matrix[k][J] += (out_state[k] - out_state[k] * out_state[i])
                        * sum;
                } else if (rnn_p->softmax_group_id[i] ==
                        rnn_p->softmax_group_id[k]) {
                    matrix[k][J] += -out_state[k] * out_state[i] * sum;
                }
            }
        }
    }
    return matrix;
}



double** rnn_jacobian_matrix (
        double** matrix,
        const struct rnn_parameters *rnn_p,
        const double *prev_c_state,
        const double *c_state,
        const double *out_state)
{
    if (rnn_p->output_type == STANDARD_TYPE) {
        jacobian_matrix_for_standard(matrix, rnn_p, prev_c_state, c_state,
                out_state);
    } else if (rnn_p->output_type == SOFTMAX_TYPE) {
        jacobian_matrix_for_softmax(matrix, rnn_p, prev_c_state, c_state,
                out_state);
    }
    return matrix;
}



void rnn_update_prior_strength (
        struct recurrent_neural_network *rnn,
        double lambda,
        double alpha)
{
    rnn->rnn_p.prior_strength = lambda * rnn->rnn_p.prior_strength +
            alpha * rnn_get_total_length(rnn);
    rnn_reset_prior_distribution(&rnn->rnn_p);
}


