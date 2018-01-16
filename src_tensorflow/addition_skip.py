from __future__ import print_function, division
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn
from tensorflow.contrib.rnn import BasicLSTMCell as LSTMBlockCell
from tqdm import trange

# Training settings
parser = argparse.ArgumentParser(description='Addition-skip task')
parser.add_argument('--sequence-length', type=int, default=50, metavar='N',
                    help='sequence length for training (default: 50)')
parser.add_argument('--hidden-size', type=int, default=110, metavar='N',
                    help='hidden size for training (default: 110)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--steps', type=int, default=2000000, metavar='N',
                    help='number of args.steps to train (default: 2000000)')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                    help='how many steps between each checkpoint (default: 10000)')
parser.add_argument('--start-step', default=0, type=int, metavar='N',
                    help='manual step number (useful on restarts) (default: 0)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--tau', type=float, default=1e-3, metavar='TAU',
                    help='value of the time penalty tau (default: 0.001)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dont-use-act', dest='use_act', action='store_false', default=True,
                    help='whether to use act')
parser.add_argument('--use-binary', dest='use_binary', action='store_true', default=False,
                    help='whether to use binary_act')
parser.add_argument('--use-skip', dest='use_skip', action='store_true', default=False,
                    help='whether to use skip_act')
parser.add_argument('--dont-print-results', dest='print_results', action='store_false', default=True,
                    help='whether to use act')
parser.add_argument('--dont-log', dest='log', action='store_false', default=True,
                    help='whether to log')
parser.add_argument('--vram-fraction', default=0.3, metavar='x',
                    help='fraction of memory to use')


def generate(args):
    x = np.stack((np.random.rand(args.batch_size, args.sequence_length) - 0.5,
                  np.zeros((args.batch_size, args.sequence_length))), axis=2)
    y = np.zeros(args.batch_size)
    n1 = np.random.randint(0, args.sequence_length//10+1, size=args.batch_size)
    n2 = np.random.randint(args.sequence_length//2, args.sequence_length, size=args.batch_size)
    for i in range(args.batch_size):
        x[i, n1[i], 1] = 1
        x[i, n2[i], 1] = 1
        y[i] = x[i, n1[i], 0] + x[i, n2[i], 0]
    return x.astype(float), y.astype(float).reshape(-1, 1)


def main():
    args = parser.parse_args()
    if args.use_act:
        if args.use_binary:
            from act_binary_cell import ACTCell
        elif args.use_skip:
            from act_skip_cell import ACTCell
        else:
            from act_cell import ACTCell

    input_size = 2
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    hidden_size = args.hidden_size
    use_act = args.use_act

    # Placeholders for inputs.
    x = tf.placeholder(tf.float32, [batch_size, sequence_length, input_size])
    inputs = [tf.squeeze(xx) for xx in tf.split(x, args.sequence_length, 1)]
    y = tf.placeholder(tf.float32, [batch_size, 1])

    rnn = LSTMBlockCell(hidden_size)
    if use_act:
        act = ACTCell(num_units=2 * args.hidden_size, cell=rnn, epsilon=0.01,
                      max_computation=100, batch_size=batch_size, state_is_tuple=True)
        outputs, final_state = static_rnn(act, inputs, dtype=tf.float32)
    else:
        outputs, final_state = static_rnn(rnn, inputs, dtype=tf.float32)

    fc_w = tf.get_variable("fc_w", [hidden_size, 1])
    fc_b = tf.get_variable("fc_b", [1])
    logits = tf.matmul(final_state[0], fc_w) + fc_b

    loss = tf.losses.mean_squared_error(labels=y, predictions=logits)
    error = tf.reduce_mean(loss)
    tf.summary.scalar('Error', error)

    if use_act:
        ponder = act.calculate_ponder_cost()
        ponder_mean = tf.reduce_mean(ponder)
        tf.summary.scalar('Ponder', ponder_mean)
        loss += args.tau * ponder

    loss = tf.reduce_mean(loss)
    tf.summary.scalar('Loss', loss)

    train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)

    merged = tf.summary.merge_all()
    logdir = './logs/addition_skip/LR={}'.format(args.lr)
    if args.use_act:
        logdir += '_Tau={}'.format(args.tau)
        if args.use_binary:
            logdir += '_Binary'
        if args.use_skip:
            logdir += '_Skip'
    else:
        logdir += '_NoACT'
    while os.path.isdir(logdir):
        logdir += '_'
    if args.log:
        writer = tf.summary.FileWriter(logdir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.vram_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        loop = trange(args.steps)
        for i in loop:
            batch = generate(args)

            if i % args.log_interval == 0:
                if use_act:

                    summary, step_error, step_loss, step_ponder \
                        = sess.run([merged, error, loss, ponder_mean], feed_dict={x: batch[0], y: batch[1]})

                    if args.print_results:
                        loop.set_postfix(Loss='{:0.3f}'.format(step_loss),
                                         Error='{:0.3f}'.format(step_error),
                                         Ponder='{:0.3f}'.format(step_ponder))
                else:
                    summary, step_error, step_loss = sess.run([merged, error, loss],
                                                                 feed_dict={
                                                                     x: batch[0], y: batch[1]})
                    if args.print_results:
                        loop.set_postfix(Loss='{:0.3f}'.format(step_loss),
                                         Error='{:0.3f}'.format(step_error))
                if args.log:
                    writer.add_summary(summary, i)
            train_step.run(feed_dict={x: batch[0], y: batch[1]})


if __name__ == '__main__':
    main()