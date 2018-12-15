from __future__ import print_function
import os, re
import tensorflow as tf
import numpy as np


class BucketedDataIterator():
    ## bucketed data iterator uses R2RT's implementation(https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)
    def __init__(self, df, num_buckets=3):
        df = df.sort_values(0).reset_index(drop=True)
        self.size = int(len(df) / num_buckets)
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.iloc[bucket * self.size: (bucket + 1) * self.size])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        # sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, n):
        if np.any(self.cursor + n > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0, self.num_buckets)

        res = self.dfs[i].iloc[self.cursor[i]:self.cursor[i] + n]
        self.cursor[i] += n
        return np.asarray(res['review'].tolist()), res['label'].tolist(), res['length'].tolist()


def get_sentence(vocabulary_inv, sen_index):
    return ' '.join([vocabulary_inv[index] for index in sen_index])


def sequence(rnn_inputs, hidden_size, seq_lens):
    cell_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print('build fw cell: ' + str(cell_fw))
    cell_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print('build bw cell: ' + str(cell_bw))
    rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                               cell_bw,
                                                               inputs=rnn_inputs,
                                                               sequence_length=seq_lens,
                                                               dtype=tf.float32
                                                               )
    print('rnn outputs: ' + str(rnn_outputs))
    print('final state: ' + str(final_state))

    return rnn_outputs

def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def batch_generator(X, y, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


if __name__ == "__main__":
    # Test batch generator
    gen = batch_generator(np.array(['a', 'b', 'c', 'd']), np.array([1, 2, 3, 4]), 2)
    for _ in range(8):
        xx, yy = next(gen)
        print(xx, yy)
