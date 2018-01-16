from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops


def _binary_round(x, epsilon):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    Based on http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    :param x: input tensor
    :return: y=round(x-0.5+epsilon) with gradients defined by the identity mapping (y=x)
    """
    g = tf.get_default_graph()
    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x-0.5+epsilon, name=name)
            # condition = tf.greater_equal(x, 1-epsilon)
            # return tf.where(condition, tf.ones_like(x), tf.zeros_like(x), name=name)


class ACTCell(RNNCell):
    """
    A RNN cell implementing Graves' Adaptive Computation Time algorithm
    """
    def __init__(self, num_units, cell, epsilon, batch_size,
                 max_computation=100, initial_bias=1., state_is_tuple=False):

        self.batch_size = batch_size
        self.one_minus_eps = tf.fill([self.batch_size], tf.constant(1.0 - epsilon, dtype=tf.float32))
        self.cell = cell
        self.epsilon = epsilon
        self._num_units = num_units
        self.max_computation = max_computation
        self.ACT_steps = []
        self.initial_bias = initial_bias

        if hasattr(self.cell, "_state_is_tuple"):
            self._state_is_tuple = self.cell._state_is_tuple
        else:
            self._state_is_tuple = state_is_tuple

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        if self._state_is_tuple:
            return self._num_units//2
        else:
            return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, timestep=0, scope=None):
        if self._state_is_tuple:
            state = tf.concat(state, 1)

        with vs.variable_scope(scope or type(self).__name__):
            # define within cell constants/ counters used to control while loop for ACTStep
            prob = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "prob")
            prob_compare = tf.zeros_like(prob, tf.float32, name="prob_compare")
            counter = tf.zeros_like(prob, tf.float32, name="counter")
            acc_outputs = tf.fill([self.batch_size, self.output_size], 0.0, name='output_accumulator')
            acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")
            acc_steps = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "steps")
            batch_mask = tf.fill([self.batch_size], True, name="batch_mask")

            # While loop stops when this predicate is FALSE.
            # Ie all (probability < 1-eps AND counter < N) are false.
            def halting_predicate(batch_mask, prob_compare, prob,
                          counter, state, input, acc_output, acc_state, acc_steps):
                # return tf.reduce_any(tf.less(prob_compare, self.one_minus_eps))
                return tf.reduce_any(tf.logical_and(
                        tf.less(prob_compare, self.one_minus_eps),
                        tf.less(counter, self.max_computation)))

            # Do while loop iterations until predicate above is false.
            _, _, remainders, iterations, _, _, output, next_state, total_steps = \
                tf.while_loop(halting_predicate, self.act_step,
                              loop_vars=[batch_mask, prob_compare, prob,
                                         counter, state, inputs, acc_outputs, acc_states, acc_steps])

        # accumulate steps
        self.ACT_steps.append(tf.reduce_mean(total_steps))

        if self._state_is_tuple:
            next_c, next_h = tf.split(next_state, 2, 1)
            next_state = tf.contrib.rnn.LSTMStateTuple(next_c, next_h)

        return output, next_state

    def calculate_ponder_cost(self):
        '''returns tensor of shape [1] which is the total ponder cost'''
        ponder = 1+tf.reduce_sum(tf.to_float(tf.add_n(self.ACT_steps)/len(self.ACT_steps)))
        # ponder = tf.Print(ponder, self.ACT_steps)
        return ponder

    def act_step(self, batch_mask, prob_compare, prob, counter, state, input, acc_outputs, acc_states, acc_steps):
        '''
        General idea: generate halting probabilites and accumulate them. Stop when the accumulated probs
        reach a halting value, 1-eps. At each timestep, multiply the prob with the rnn output/state.
        There is a subtlety here regarding the batch_size, as clearly we will have examples halting
        at different points in the batch. This is dealt with using logical masks to protect accumulated
        probabilities, states and outputs from a timestep t's contribution if they have already reached
        1 - es at a timstep s < t. On the last timestep for each element in the batch the remainder is
        multiplied with the state/output, having been accumulated over the timesteps, as this takes
        into account the epsilon value.
        '''

        # If all the probs are zero, we are seeing a new input => binary flag := 1, else 0.
        binary_flag = tf.cond(tf.reduce_all(tf.equal(prob, 0.0)),
                              lambda: tf.ones([self.batch_size, 1], dtype=tf.float32),
                              lambda: tf.zeros([self.batch_size, 1], dtype=tf.float32))

        input_with_flags = tf.concat([binary_flag, input], 1)
        if self._state_is_tuple:
            (c, h) = tf.split(state, 2, 1)
            state = tf.contrib.rnn.LSTMStateTuple(c, h)

        output, new_state = self.cell(input_with_flags, state)

        if self._state_is_tuple:
            new_state = tf.concat(new_state, 1)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.layers.dense(output, 1, activation=tf.sigmoid,
                                           use_bias=True,
                                           bias_initializer=tf.constant_initializer(self.initial_bias), name='fc'),
                           squeeze_dims=1)

        # Multiply by the previous mask as if we stopped before, we don't want to start again
        # if we generate a p less than p_t-1 for a given example.
        new_batch_mask = tf.less(prob + p, self.one_minus_eps)
        new_float_mask = tf.cast(new_batch_mask, tf.float32)
        float_mask = tf.cast(batch_mask, tf.float32)

        prob = prob + p * float_mask
        # This accumulator is used solely in the While loop condition.
        # we multiply by the PREVIOUS batch mask, to capture probabilities
        # that have gone over 1-eps THIS iteration.
        prob_compare += p * float_mask
        # prob_compare = tf.Print(prob_compare, data=[prob_compare], summarize=6)

        # Only increase the counter for those probabilities that
        # did not go over 1-eps in this iteration.
        counter += new_float_mask

        counter_condition = tf.less(counter, self.max_computation)

        final_iteration_condition = tf.logical_and(new_batch_mask, counter_condition)

        prob_exp = tf.expand_dims(prob, -1)
        prob_exp = tf.Print(prob_exp, [prob_exp], summarize=5)
        weight_continue = _binary_round(prob_exp, self.epsilon)
        weight_stop = _binary_round(prob_exp+1, self.epsilon)

        update_weight = tf.where(final_iteration_condition, weight_continue, weight_stop)

        float_mask_exp = tf.expand_dims(float_mask, -1)

        acc_state = (new_state * update_weight * float_mask_exp) + acc_states
        acc_output = (output * update_weight * float_mask_exp) + acc_outputs
        acc_steps = ((1-_binary_round(prob, self.epsilon)) * float_mask) + acc_steps
        # acc_steps = tf.Print(acc_steps, [acc_steps], summarize=32)

        return [new_batch_mask, prob_compare, prob, counter, new_state, input, acc_output, acc_state, acc_steps]