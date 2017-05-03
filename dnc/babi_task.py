# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A repeat copy task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf
from itertools import chain
from six.moves import reduce
# noinspection PyUnresolvedReferences
from data_utils import vectorize_sentences
import random

DatasetTensors = collections.namedtuple('DatasetTensors', ('data', 'label'))


def masked_sigmoid_cross_entropy(logits,
                                 label,
                                 data,
                                 time_average=False,
                                 log_prob_in_bits=False):
    """Adds ops to graph which compute the (scalar) NLL of the target sequence.

    The logits parametrize independent bernoulli distributions per time-step and
    per batch element, and irrelevant time/batch elements are masked out by the
    mask tensor.

    Args:
      logits: `Tensor` of activations for which sigmoid(`logits`) gives the
          bernoulli parameter.
      label: time-major `Tensor` of target.
      data: time-major `Tensor` to be multiplied elementwise with cost T x B cost
          masking out irrelevant time-steps.
      time_average: optionally average over the time dimension (sum by default).
      log_prob_in_bits: iff True express log-probabilities in bits (default nats).

    Returns:
      A `Tensor` representing the log-probability of the target.
    """
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
    loss_time_batch = tf.reduce_sum(xent, axis=2)
    loss_batch = tf.reduce_sum(loss_time_batch * data, axis=0)

    batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

    if time_average:
        mask_count = tf.reduce_sum(data, axis=0)
        loss_batch /= (mask_count + np.finfo(np.float32).eps)

    loss = tf.reduce_sum(loss_batch) / batch_size
    if log_prob_in_bits:
        loss /= tf.log(2.)

    return loss


class BabiTask(snt.AbstractModule):
    def __init__(
            self,
            train,
            test,
            batch_size):
        super(BabiTask, self).__init__('babi_task')

        self._batch_size = batch_size
        data = train + test

        vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

        # noinspection PyRedeclaration
        max_story_size = max(map(len, (s for s, _, _ in data)))
        # noinspection PyRedeclaration
        max_sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
        # noinspection PyRedeclaration
        max_query_size = max(map(len, (q for _, q, _ in data)))

        num_steps = self._num_steps = max_story_size * max_sentence_size + max_query_size

        self._S, self._A = vectorize_sentences(train, word_idx, max_sentence_size, max_story_size, num_steps)
        self._Ste, self._Ate = vectorize_sentences(test, word_idx, max_sentence_size, max_story_size, num_steps)

        self._train_batches = tuple(
            zip(range(0, len(self._S[0]), batch_size), range(batch_size, len(self._S[0]), batch_size)))
        self._test_batches = tuple(
            zip(range(0, len(self._Ste[0]), batch_size), range(batch_size, len(self._Ste[0]), batch_size)))

    def num_steps(self):
        return self._num_steps

    def next_train(self):
        batch = random.choice(self._train_batches)
        return self.sublist(self._S, *batch), self.sublist(self._A, *batch)

    def next_test(self):
        batch = random.choice(self._test_batches)
        return self.sublist(self._Ste, *batch), self.sublist(self._Ate, *batch)

    @staticmethod
    def sublist(list_obj, start, end):
        return list_obj[start: end]

    def _build(self):
        return DatasetTensors(tf.placeholder(tf.float32, [None, self._num_steps, 1]),
                              tf.placeholder(tf.float32, [None, 20]))

    @staticmethod
    def cost(logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
