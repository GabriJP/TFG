#
#
#  Very simple LSTM implementation.
#
#


import tensorflow as tf
from itertools import chain
# noinspection PyUnresolvedReferences
from data_utils import load_task, vectorize_sentences
from functools import reduce
import time

print(tf.__version__)

import numpy as np

start_time = time.time()

tf.flags.DEFINE_boolean("L2", False, "Use L2 regularization or not")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("dropout", 1, "Dropout")
# tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
# tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 20, "Batch size for training.")
# tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
# tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_integer("num_layers", 2, "Number of layers.")
tf.flags.DEFINE_integer("num_hidden", 128, "Number of hidden units.")
# tf.flags.DEFINE_integer("steps", 25, "Number of steps in the LSTM.")
tf.flags.DEFINE_string("data_dir", "../familytask", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("log_dir", "data/logs/data1", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

tf.reset_default_graph()
print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# noinspection PyRedeclaration
max_story_size = max(map(len, (s for s, _, _ in data)))
# noinspection PyRedeclaration
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
# noinspection PyRedeclaration
query_size = max(map(len, (q for _, q, _ in data)))

num_steps = max_story_size * sentence_size + query_size

S, A = vectorize_sentences(train, word_idx, sentence_size, max_story_size, num_steps)
Ste, Ate = vectorize_sentences(test, word_idx, sentence_size, max_story_size, num_steps)

inputs = 1

input_data = tf.placeholder(tf.float32, [None, num_steps, inputs])
input_label = tf.placeholder(tf.float32, [None, len(vocab) + 1])


w_o = tf.Variable(tf.random_normal([FLAGS.num_hidden, len(vocab) + 1], stddev=0.1), name='w_o')
b_o = tf.Variable(tf.zeros([1]), name='b_o')

def inference(_input_data):

    # lstm_layers = [tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_hidden)
    #                for _ in range(FLAGS.num_layers)]

    x = tf.unstack(_input_data, num_steps, 1)

    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_hidden)

    rnn_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_hidden) for _ in range(FLAGS.num_layers)]
    rnn_multilayer = tf.contrib.rnn.MultiRNNCell(rnn_cells)

    outputs, states = tf.contrib.rnn.static_rnn(rnn_multilayer, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], w_o) + b_o

    # outputs = []
    # state = lstm_system.zero_state(FLAGS.batch_size, tf.float32)
    # inp = _input_data[:, 0, :]
    # (cell_output, state) = lstm_system(inp, state)
    # outputs.append(tf.nn.sigmoid(tf.matmul(cell_output, w_o) + b_o))
    # tf.get_variable_scope().reuse_variables()
    # for time_step in range(1, num_steps):
    #     inp = _input_data[:, time_step, :]
    #     (cell_output, state) = lstm_system(inp, state)
    #     outputs.append(tf.nn.sigmoid(tf.matmul(cell_output, w_o) + b_o))
    #
    # result = tf.transpose(tf.squeeze(outputs), perm=[1, 0, 2])
    # return result


# Training ------------------------------------------------
predicted = inference(input_data)
if FLAGS.L2:
    regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                         if "w_o" in v.name or "weights" in v.name])
    cross_entropy = tf.reduce_sum(tf.square(input_label - predicted)) + regularization_cost
else:
    # cross_entropy = tf.reduce_sum(tf.square(input_label - predicted[:, -1]))
    cross_entropy = tf.reduce_sum(tf.square(input_label - predicted))

# opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
train_step = opt.minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(input_label, 1))
accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(input_label, 1), tf.argmax(predicted, 1)),
                                 tf.float32))

accuracy_batch = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(input_label, 1), tf.argmax(predicted, 1)),
                                 tf.float32))/float(FLAGS.batch_size)


tf.summary.scalar("accuracy", accuracy_batch)
tf.summary.scalar("cross_entropy", cross_entropy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

writer = tf.summary.FileWriter(FLAGS.log_dir)
writer.add_graph(sess.graph)

batches = list(zip(range(0, S.shape[0], FLAGS.batch_size), range(FLAGS.batch_size, S.shape[0], FLAGS.batch_size)))

i = 0
for j in range(1000):  # 1000
    print("-----------------> Epoch:", j)
    for start, end in batches:
        sess.run(train_step, feed_dict={input_data: S[start:end], input_label: A[start:end]})
        i += 1
        if i % 20 == 0:
            [ce, acc, summ] = sess.run([cross_entropy, accuracy_batch, merged],
                                       feed_dict={input_data: S[start:end], input_label: A[start:end]})
            writer.add_summary(summ, i)
            print("iteration: ", i, "ce: ", ce, "acc", acc)
            if i % 500 == 0:
                tf.train.Saver().save(sess, FLAGS.log_dir + "model.ckpt", i)

# Test ----------------------------------------------------

acc = sess.run(accuracy, feed_dict={input_data: Ste, input_label: Ate})
print("-------------------------------------------------------------------")
print("Testing - accuracy: ", acc/float(len(Ate)))
print("--- %s minutes ---" % str((time.time() - start_time)/60))


# FLAGS.dropout = 0
# FLAGS.batch_size = 13
#
# test_batches = list(zip(range(0, Ste.shape[0], FLAGS.batch_size), range(FLAGS.batch_size, Ste.shape[0], FLAGS.batch_size)))
#
# print("Test batches:", test_batches)
#
# for start, end in test_batches:
#     acc = sess.run(accuracy, feed_dict={input_data: Ste[start:end], input_label: Ate[start:end]})
#     print("Testing - ce: ", acc)
#
# print("--- %s seconds ---" % (time.time() - start_time))
