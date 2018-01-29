from __future__ import print_function, division
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn
from tensorflow.contrib.rnn import BasicLSTMCell as LSTMBlockCell
from tqdm import trange

# Training settings
parser = argparse.ArgumentParser(description='Addition task')
parser.add_argument('--sequence-length', type=int, default=5, metavar='N',
                    help='sequence length for training (default: 5)')
parser.add_argument('--total-digits', type=int, default=5, metavar='N',
                    help='total digits for training (default: 5)')
parser.add_argument('--hidden-size', type=int, default=512, metavar='N',
                    help='hidden size for training (default: 512)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
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
                    help='whether to use binary act')
parser.add_argument('--use-skip', dest='use_skip', action='store_true', default=False,
                    help='whether to use skip_act')
parser.add_argument('--dont-print-results', dest='print_results', action='store_false', default=True,
                    help='whether to use act')
parser.add_argument('--dont-log', dest='log', action='store_false', default=True,
                    help='whether to log')
parser.add_argument('--return-ponders', dest='return_ponders', action='store_true', default=False,
                    help='return ponders for qualitative analysis')
parser.add_argument('--vram-fraction', default=1., type=float, metavar='x',
                    help='fraction of memory to use (default: 1)')

args = parser.parse_args()

input_size = 10*args.total_digits
output_size = 11*(args.total_digits+1)
num_layers = 1


# Generate samples
def add_vec(x, y):
    n = max(len(x), len(y))
    return num2vec(vec2num(x) + vec2num(y), n)


def vec2num(x):
    s = 0
    for i in range(len(x)):
        if x[i] == 10:
            break
        s *= 10
        s += x[i]
    return s


def num2vec(x, n):
    y = np.zeros(n) + 10
    digits = len(str(int(x)))
    for i in range(digits):
        y[i] = (x//10**(digits-i-1)) % 10
    return y


def encode_in(x):
    y = np.zeros(len(x)*10)
    for i in range(len(x)):
        if x[i] == 10:
            break
        else:
            y[10*i+int(x[i])] = 1
    return y


def encode_out(x):
    y = np.zeros(len(x)*11, dtype=int)
    for i in range(len(x)):
        if x[i] == 10:
            y[11*i+10] = 1
        else:
            y[11*i+int(x[i])] = 1
    return y


def decode_out(x):
    y = np.zeros(len(x)//11, dtype=int)
    for i in range(len(y)):
        y[i] = np.argmax(x[i*11:(i+1)*11])
    return y


def generate(args):
    input_dec = np.zeros((args.batch_size, args.sequence_length, args.total_digits), dtype=int) + 10
    input_enc = np.zeros((args.batch_size, args.sequence_length, input_size), dtype=int)
    output_dec = np.zeros((args.batch_size, args.sequence_length, args.total_digits+1), dtype=int)
    output_enc = np.zeros((args.batch_size, args.sequence_length, output_size), dtype=int)
    for i in range(args.batch_size):
        for j in range(args.sequence_length):
            digits = np.random.randint(args.total_digits) + 1
            for k in range(digits):
                d = np.random.randint(10)
                input_dec[i, j, k] = d
            if j == 0:
                output_dec[i, j, :-1] = input_dec[i, j, :]
                output_dec[i, j, -1] = 10
            elif j > 0:
                output_dec[i, j, :] = add_vec(output_dec[i, j-1, :], input_dec[i, j, :])
            input_enc[i, j, :] = encode_in(input_dec[i, j, :])
            output_enc[i, j, :] = encode_out(output_dec[i, j, :])
    x = input_enc.astype(float)
    y = output_dec.astype(int)[:, 1:, :].reshape(-1)
    return x, y


def main():
    args = parser.parse_args()
    if args.use_act:
        if args.use_binary:
            from act_binary_cell import ACTCell
        elif args.use_skip:
            from act_skip_cell import ACTCell
        else:
            from act_cell import ACTCell
    if args.use_binary and args.use_skip:
        raise Exception

    input_size = 10 * args.total_digits
    output_size = 11 * (args.total_digits + 1)
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    use_act = args.use_act

    # Placeholders for inputs.
    x = tf.placeholder(tf.float32, [batch_size, args.sequence_length, input_size])
    inputs = [tf.squeeze(xx) for xx in tf.split(x, args.sequence_length, 1)]
    y = tf.placeholder(tf.int64, [batch_size*(args.sequence_length-1)*(args.total_digits+1)])

    rnn = LSTMBlockCell(hidden_size)
    if use_act:
        act = ACTCell(num_units=2*args.hidden_size, cell=rnn,
                      max_computation=20, batch_size=batch_size, state_is_tuple=True, return_ponders=args.return_ponders)
        outputs, final_state = static_rnn(act, inputs, dtype=tf.float32)
    else:
        outputs, final_state = static_rnn(rnn, inputs, dtype=tf.float32)

    output = tf.reshape(tf.stack(outputs, 1)[:, 1:, :], [-1, hidden_size])
    softmax_w = tf.get_variable("softmax_w", [hidden_size, output_size])
    softmax_b = tf.get_variable("softmax_b", [output_size])
    logits = tf.reshape(tf.matmul(output, softmax_w) + softmax_b, [-1, 11])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    loss = tf.reduce_mean(tf.reshape(loss, [batch_size, -1]), axis=1)

    if use_act:
        if args.return_ponders:
            ponder, ponders_tensor = act.calculate_ponder_cost()
            ponders_tensor = tf.reduce_mean(ponders_tensor, axis=0)
        else:
            ponder = act.calculate_ponder_cost()
        ponder_mean = tf.reduce_mean(ponder)
        tf.summary.scalar('Ponder', ponder_mean)
        loss += args.tau*ponder

    loss = tf.reduce_mean(loss)
    tf.summary.scalar('Loss', loss)

    train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)

    predicted = tf.argmax(logits, 1)
    target = tf.cast(y, tf.int64)
    correct_sequences = tf.cast(tf.reduce_all(tf.reshape(tf.equal(predicted, target),
                                [args.batch_size, (args.sequence_length-1)*(args.total_digits+1)]), axis=1), tf.float32)
    accuracy = tf.reduce_mean(correct_sequences)
    tf.summary.scalar('Accuracy', accuracy)

    merged = tf.summary.merge_all()
    logdir = './logs/addition/LR={}'.format(args.lr)
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
    if args.return_ponders:
        ponders_list = list()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        loop = trange(args.steps)
        for i in loop:
            batch = generate(args)

            if i % args.log_interval == 0:
                if use_act:
                    if args.return_ponders:
                        summary, step_accuracy, step_loss, step_ponder, step_ponders_tensor \
                            = sess.run([merged, accuracy, loss, ponder_mean, ponders_tensor],
                                       feed_dict={x: batch[0], y: batch[1]})
                        ponders_list.append(step_ponders_tensor)
                        stack = np.stack(ponders_list, axis=0)
                        np.savetxt('ponders_addition.txt', stack)
                    else:
                        summary, step_accuracy, step_loss, step_ponder \
                            = sess.run([merged, accuracy, loss, ponder_mean],
                                       feed_dict={x: batch[0], y: batch[1]})
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

    if args.return_ponders:
        stack = np.stack(ponders_list, axis=0)
        np.savetxt('ponders_addition.txt', stack)


if __name__ == '__main__':
    main()
