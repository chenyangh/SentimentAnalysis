import numpy as np
import tensorflow as tf


class TextCNNRNN(object):
    def __init__(self, embedding_mat, non_static, hidden_unit, sequence_length, max_pool_size,
                 num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32)
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if not non_static:
                W = tf.constant(embedding_mat, name='W')
            else:
                W = tf.Variable(embedding_mat, name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.emb = tf.expand_dims(self.embedded_chars, -1)

        pooled_concat = []
        reduced = np.int32(np.ceil(sequence_length * 1.0 / max_pool_size))

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Zero paddings so that the convolution output have dimension batch x
                # sequence_length x emb_size x channel
                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat(1, [self.pad] * num_prio)
                pad_post = tf.concat(1, [self.pad] * num_post)
                emb_pad = tf.concat(1, [pad_prio, self.emb, pad_post])

                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1],
                                        padding='SAME', name='pool')
                pooled = tf.reshape(pooled, [-1, reduced, num_filters])
                pooled_concat.append(pooled)

        pooled_concat = tf.concat(2, pooled_concat)
        pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)

        # lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit)
        lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_unit)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced, pooled_concat)]
        outputs, state = tf.nn.rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)

        # Collect the appropriate last words into variable output (dimension = batch x embedding_size)
        output = outputs[0]
        with tf.variable_scope('Output'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, hidden_unit], tf.float32)
            for i in range(1, len(outputs)):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(tf.mul(output, mat), tf.mul(outputs[i], 1.0 - mat))

        with tf.name_scope('output'):
            self.W = tf.Variable(tf.truncated_normal([hidden_unit, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, self.W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))


class TextRNN(object):
    def __init__(self, embedding_mat, non_static, hidden_unit, sequence_length, max_pool_size,
                 num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32)
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if not non_static:
                W = tf.constant(embedding_mat, name='W')
            else:
                W = tf.Variable(embedding_mat, name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        #    emb = tf.expand_dims(self.embedded_chars, -1)
        reduced = sequence_length
        # pooled_concat = []
        # reduced = np.int32(np.ceil(sequence_length * 1.0 / max_pool_size))
        #
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope('conv-maxpool-%s' % filter_size):
        #         # Zero paddings so that the convolution output have dimension batch x
        #         # sequence_length x emb_size x channel
        #         num_prio = (filter_size - 1) // 2
        #         num_post = (filter_size - 1) - num_prio
        #         pad_prio = tf.concat(1, [self.pad] * num_prio)
        #         pad_post = tf.concat(1, [self.pad] * num_post)
        #         emb_pad = tf.concat(1, [pad_prio, emb, pad_post])
        #
        #         filter_shape = [filter_size, embedding_size, 1, num_filters]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
        #         conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        #
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        #
        #         # Maxpooling over the outputs
        #         pooled = tf.nn.max_pool(h, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1],
        #                                 padding='SAME', name='pool')
        #         pooled = tf.reshape(pooled, [-1, reduced, num_filters])
        #         pooled_concat.append(pooled)
        #
        # pooled_concat = tf.concat(2, pooled_concat)
        # pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit)
        lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_unit)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced, self.embedded_chars)]
        outputs, state = tf.nn.rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)

        # Collect the appropriate last words into variable output (dimension = batch x embedding_size)
        output = outputs[0]
        with tf.variable_scope('Output'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, hidden_unit], tf.float32)
            for i in range(1, len(outputs)):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(tf.mul(output, mat), tf.mul(outputs[i], 1.0 - mat))

        with tf.name_scope('output'):
            self.W = tf.Variable(tf.truncated_normal([hidden_unit, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, self.W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, embedding_mat, non_static, hidden_unit, sequence_length, max_pool_size,
                 num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32)
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if not non_static:
                Ww = tf.constant(embedding_mat, name='Ww')
            else:
                Ww = tf.Variable(embedding_mat, name='Ww')
            self.embedded_chars = tf.nn.embedding_lookup(Ww, self.input_x)
            self.emb = tf.expand_dims(self.embedded_chars, -1)

        pooled_concat = []
        reduced = np.int32(np.ceil(sequence_length * 1.0 / max_pool_size))
        # Create a convolution + maxpool layer for each filter size
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Zero paddings so that the convolution output have dimension batch x
                # sequence_length x emb_size x channel
                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio


                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.emb, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                pooled_concat.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_concat)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))


class TextCNNRNN2(object):
    def __init__(self, embedding_mat, non_static, hidden_unit, sequence_length, max_pool_size,
                 num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32)
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if not non_static:
                W = tf.constant(embedding_mat, name='W')
            else:
                W = tf.Variable(embedding_mat, name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.emb = tf.expand_dims(self.embedded_chars, -1)

        pooled_concat = []
        reduced = np.int32(np.ceil(sequence_length * 1.0 / max_pool_size))
        reduced2 = np.int32(np.ceil(reduced * 1.0 / max_pool_size)) - 1
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv1-maxpool-%s' % filter_size):
                # Zero paddings so that the convolution output have dimension batch x
                # sequence_length x emb_size x channel

                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat(1, [self.pad] * num_prio)
                pad_post = tf.concat(1, [self.pad] * num_post)
                emb_pad = tf.concat(1, [pad_prio, self.emb, pad_post])

                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1],
                                        padding='SAME', name='pool')

                next_filter_size = np.int32(filter_size/2)
                next_filter_num = np.int32(np.ceil(num_filters/2))

            with tf.name_scope('conv2-maxpool-%s' % next_filter_size):
                W_2 = tf.Variable(tf.truncated_normal([next_filter_size, 1, num_filters, next_filter_num], stddev=0.1), name='W_2')
                b_2 = tf.Variable(tf.constant(0.1, shape=[next_filter_num]), name='b_2')
                conv2 = tf.nn.conv2d(pooled, W_2, strides=[1, 1, 1, 1], padding='VALID', name='conv2')
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b_2), name='relu2')
                pooled2 = tf.nn.max_pool(h2, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1],
                                        padding='SAME', name='pool2')
                pooled = tf.reshape(pooled2, [-1, reduced2, next_filter_num])
                pooled_concat.append(pooled)

        pooled_concat = tf.concat(2, pooled_concat)
        pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit)
        #lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_unit)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced2, pooled_concat)]
        outputs, state = tf.nn.rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)

        # Collect the appropriate last words into variable output (dimension = batch x embedding_size)
        output = outputs[0]
        with tf.variable_scope('Output'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, hidden_unit], tf.float32)
            for i in range(1, len(outputs)):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(tf.mul(output, mat), tf.mul(outputs[i], 1.0 - mat))

        with tf.name_scope('output'):
            self.W = tf.Variable(tf.truncated_normal([hidden_unit, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, self.W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))
