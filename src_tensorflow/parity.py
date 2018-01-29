from __future__ import print_function, division
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn, BasicRNNCell
from tqdm import trange

# Training settings
parser = argparse.ArgumentParser(description='Parity task')
parser.add_argument('--input-size', type=int, default=64, metavar='N',
                    help='input size for training (default: 64)')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N',
                    help='hidden size for training (default: 128)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--steps', type=int, default=2000000, metavar='N',
                    help='number of args.steps to train (default: 2000000)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many steps between each checkpoint (default: 1000)')
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
parser.add_argument('--vram-fraction', default=1., type=float, metavar='x',
                    help='fraction of memory to use (default: 1)')


def generate(args):
    n_random = np.random.randint(1, args.input_size+1, size=args.batch_size)
    x = np.zeros((args.batch_size, args.input_size))
    for i in range(args.batch_size):
        randoms = np.random.randint(2, size=n_random[i], dtype=int)
        randoms[randoms == 0] = -1
        x[i, :n_random[i]] = randoms
    y = np.zeros(args.batch_size)
    for i in range(args.batch_size):
        y[i] = np.count_nonzero(x[i][x[i] == 1]) % 2
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

    input_size = args.input_size
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    use_act = args.use_act

    # Placeholders for inputs.
    x = tf.placeholder(tf.float32, [batch_size, input_size])
    inputs = [x]
    y = tf.placeholder(tf.float32, [batch_size, 1])
    zeros = tf.zeros([batch_size, 1])

    rnn = BasicRNNCell(args.hidden_size)
    if use_act:
        act = ACTCell(num_units=hidden_size, cell=rnn, max_computation=100, batch_size=batch_size)
        outputs, final_state = static_rnn(act, inputs, dtype=tf.float32)
    else:
        outputs, final_state = static_rnn(rnn, inputs, dtype=tf.float32)

    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
    softmax_w = tf.get_variable("softmax_w", [hidden_size, 1])
    softmax_b = tf.get_variable("softmax_b", [1])
    logits = tf.matmul(output, softmax_w) + softmax_b

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

    if use_act:
        ponder = act.calculate_ponder_cost()
        ponder_mean = tf.reduce_mean(ponder)
        tf.summary.scalar('Ponder', ponder_mean)
        loss += args.tau*ponder

    loss = tf.reduce_mean(loss)

    train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)

    correct_prediction = tf.equal(tf.cast(tf.greater(logits, zeros), tf.float32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Loss', loss)

    merged = tf.summary.merge_all()
    logdir = './logs/parity/LR={}_Len={}'.format(args.lr, args.input_size)
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
                    summary, step_accuracy, step_loss, step_ponder \
                            = sess.run([merged, accuracy, loss, ponder_mean], feed_dict={x: batch[0], y: batch[1]})

                    if args.print_results:
                        loop.set_postfix(Loss='{:0.3f}'.format(step_loss),
                                         Accuracy='{:0.3f}'.format(step_accuracy),
                                         Ponder='{:0.3f}'.format(step_ponder))
                else:
                    summary, step_accuracy, step_loss = sess.run([merged, accuracy, loss],
                                                                              feed_dict={
                                                                                  x: batch[0], y: batch[1]})

                    if args.print_results:
                        loop.set_postfix(Loss='{:0.3f}'.format(step_loss),
                                         Accuracy='{:0.3f}'.format(step_accuracy))
                if args.log:
                    writer.add_summary(summary, i)
            train_step.run(feed_dict={x: batch[0], y: batch[1]})


if __name__ == '__main__':
    main()






