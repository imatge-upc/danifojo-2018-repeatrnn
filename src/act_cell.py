from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope


class ACTCell(RNNCell):
    def __init__(self, num_units, cell, batch_size, epsilon=0.01,
                 max_computation=100, initial_bias=1., state_is_tuple=False, return_ponders=False):

        self.batch_size = batch_size
        self.one_minus_eps = tf.fill([self.batch_size], tf.constant(1.0 - epsilon, dtype=tf.float32))
        self.cell = cell
        self._num_units = num_units
        self.max_computation = max_computation
        self.ACT_remainders = []
        self.ACT_iterations = []
        self.return_ponders_tensor = return_ponders
        self.initial_bias = initial_bias

        if hasattr(self.cell, "_state_is_tuple"):
            self._state_is_tuple = self.cell._state_is_tuple
        else:
            self._state_is_tuple = state_is_tuple

    @property
    def output_size(self):
        return self.cell.output_size

    @property
    def state_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, timestep=0, scope=None):

        with variable_scope.variable_scope(scope or type(self).__name__):
            prob = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "prob")
            prob_compare = tf.zeros_like(prob, tf.float32, name="prob_compare")
            counter = tf.zeros_like(prob, tf.float32, name="counter")
            acc_outputs = tf.fill([self.batch_size, self.output_size], 0.0, name='output_accumulator')
            batch_mask = tf.fill([self.batch_size], True, name="batch_mask")
            acc_states = self.cell.zero_state(self.batch_size, tf.float32)

            _, _, acc_probs, iterations, _, _, final_output, final_state = \
                tf.while_loop(self._while_condition, self._while_body,
                              loop_vars=[batch_mask, prob_compare, prob,
                                         counter, state, inputs, acc_outputs, acc_states])

        self.ACT_remainders.append(1 - acc_probs)
        self.ACT_iterations.append(iterations)

        return final_output, final_state

    def calculate_ponder_cost(self):
        ponders_tensor = tf.stack(self.ACT_remainders, axis=1) + tf.to_float(tf.stack(self.ACT_iterations, axis=1))
        ponders = tf.reduce_mean(ponders_tensor, axis=1)
        if self.return_ponders_tensor:
            ponders_tensor = tf.stack(self.ACT_remainders, axis=1) + tf.to_float(tf.stack(self.ACT_iterations, axis=1))
            return ponders, ponders_tensor
        return ponders

    def _while_condition(self, batch_mask, prob_compare, prob, counter, state, input, acc_output, acc_state):
        return tf.reduce_any(tf.logical_and(
            tf.less(prob_compare, self.one_minus_eps),
            tf.less(counter, self.max_computation)))

    def _while_body(self, batch_mask, prob_compare, prob, counter, state, input, acc_outputs, acc_states):

        binary_flag = tf.cond(tf.reduce_all(tf.equal(prob, 0.0)),
                              lambda: tf.ones([self.batch_size, 1], dtype=tf.float32),
                              lambda: tf.zeros([self.batch_size, 1], dtype=tf.float32))

        input_with_flags = tf.concat([binary_flag, input], 1)

        output, new_state = self.cell(input_with_flags, state)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.layers.dense(output, 1, activation=tf.sigmoid,
                                           use_bias=True,
                                           bias_initializer=tf.constant_initializer(self.initial_bias)),
                           squeeze_dims=1)

        new_batch_mask = tf.logical_and(tf.less(prob + p, self.one_minus_eps), batch_mask)
        new_float_mask = tf.cast(new_batch_mask, tf.float32)

        prob += p * new_float_mask

        prob_compare += p * tf.cast(batch_mask, tf.float32)

        counter += new_float_mask

        counter_condition = tf.less(counter, self.max_computation)

        final_iteration_condition = tf.logical_and(new_batch_mask, counter_condition)
        remainders = tf.expand_dims(1.0 - prob, -1)
        probabilities = tf.expand_dims(p, -1)
        update_weight = tf.where(final_iteration_condition, probabilities, remainders)
        float_mask = tf.expand_dims(tf.cast(batch_mask, tf.float32), -1)

        acc_state = (new_state * update_weight * float_mask) + acc_states

        if self._state_is_tuple:
            (c, h) = tf.split(acc_state, 2, 0)
            acc_state = tf.contrib.rnn.LSTMStateTuple(tf.squeeze(c), tf.squeeze(h))

        acc_output = (output * update_weight * float_mask) + acc_outputs
        return [new_batch_mask, prob_compare, prob, counter, new_state, input, acc_output, acc_state]
