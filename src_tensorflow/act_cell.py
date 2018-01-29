from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs


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
            prob = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "prob")
            prob_compare = tf.zeros_like(prob, tf.float32, name="prob_compare")
            counter = tf.zeros_like(prob, tf.float32, name="counter")
            acc_outputs = tf.fill([self.batch_size, self.output_size], 0.0, name='output_accumulator')
            acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")
            batch_mask = tf.fill([self.batch_size], True, name="batch_mask")

            def while_condition(batch_mask, prob_compare, prob,
                          counter, state, input, acc_output, acc_state):
                return tf.reduce_any(tf.logical_and(
                        tf.less(prob_compare,self.one_minus_eps),
                        tf.less(counter, self.max_computation)))

            _, _, acc_probs, iterations, _, _, output, next_state = \
                tf.while_loop(while_condition, self.while_body,
                              loop_vars=[batch_mask, prob_compare, prob,
                                         counter, state, inputs, acc_outputs, acc_states])

        self.ACT_remainders.append(1 - acc_probs)
        self.ACT_iterations.append(iterations)

        if self._state_is_tuple:
            next_c, next_h = tf.split(next_state, 2, 1)
            next_state = tf.contrib.rnn.LSTMStateTuple(next_c, next_h)

        return output, next_state

    def calculate_ponder_cost(self):
        ponders = tf.add_n(self.ACT_remainder)/len(self.ACT_remainder) + \
            tf.to_float(tf.add_n(self.ACT_iterations)/len(self.ACT_iterations))
        if self.return_ponders_tensor:
            ponders_tensor = tf.stack(self.ACT_remainder, axis=1) + tf.to_float(tf.stack(self.ACT_iterations, axis=1))
            return ponders, ponders_tensor
        return ponders

    def while_body(self, batch_mask, prob_compare, prob, counter, state, input, acc_outputs, acc_states):

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
                                           bias_initializer=tf.constant_initializer(self.initial_bias)),
                           squeeze_dims=1)

        new_batch_mask = tf.logical_and(tf.less(prob + p, self.one_minus_eps), batch_mask)
        new_float_mask = tf.cast(new_batch_mask, tf.float32)

        prob += p * new_float_mask

        prob_compare += p * tf.cast(batch_mask, tf.float32)
        # prob_compare = tf.Print(prob_compare, [prob_compare], summarize=6)

        counter += new_float_mask
        # counter = tf.Print(counter, [counter], summarize=32)

        counter_condition = tf.less(counter, self.max_computation)

        final_iteration_condition = tf.logical_and(new_batch_mask, counter_condition)
        remainders = tf.expand_dims(1.0 - prob, -1)
        probabilities = tf.expand_dims(p, -1)
        update_weight = tf.where(final_iteration_condition, probabilities, remainders)
        float_mask = tf.expand_dims(tf.cast(batch_mask, tf.float32), -1)

        acc_state = (new_state * update_weight * float_mask) + acc_states
        acc_output = (output * update_weight * float_mask) + acc_outputs
        return [new_batch_mask, prob_compare, prob, counter, new_state, input, acc_output, acc_state]
