import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN, TextRNN, TextCNN, TextCNNRNN2
from sklearn.model_selection import train_test_split
import datetime
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train_cnn_rnn():
    x_, y_, x_test, y_test, vocabulary, vocabulary_inv, labels = data_helper.load_data()

    training_config = 'training_config.json'
    params = json.loads(open(training_config).read())

    # Assign a 300 dimension vector to each word
    word_embeddings = data_helper.load_embeddings(vocabulary)
    embedding_mat = []
    for i in range(len(vocabulary_inv)):
        embedding_mat.append(word_embeddings[vocabulary_inv[i]])
    embedding_mat = np.array(embedding_mat, dtype=np.float32)

    # Split the original dataset into train set and test set

    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)

    # Create a directory, everything related to the training will be saved in this directory
    timestamp = str(int(time.time()))
    trained_dir = './trained_results_' + timestamp + '/'
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            cnn_rnn = TextCNN(
                embedding_mat=embedding_mat,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                non_static=params['non_static'],
                hidden_unit=params['hidden_unit'],
                max_pool_size=params['max_pool_size'],
                filter_sizes=[int(x) for x in params['filter_sizes'].split(",")],
                num_filters=params['num_filters'],
                embedding_size=params['embedding_dim'],
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            # optimizer = tf.train.RMSPropOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn_rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn_rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)



            # Checkpoint files will be saved in this directory during training
            checkpoint_dir = './checkpoints_' + timestamp + '/'
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            def real_len(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                summaries, _, step, loss, accuracy = sess.run(
                    [train_summary_op, train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
                # print(accuracy)


            def dev_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: 1.0,
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                summaries, step, loss, accuracy, num_correct, predictions = sess.run(
                    [dev_summary_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions],
                    feed_dict)
                dev_summary_writer.add_summary(summaries, step)
                print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
                return accuracy, predictions

            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            # Train the model with x_train and y_train
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                # Evaluate the model with x_dev and y_dev
                if current_step % params['evaluate_every'] == 0:



                    print("Evaluation:", end=' ')
                    _ = dev_step(x_dev, y_dev)
                    print("Test:", end=' ')
                    acc_tmp, pred__ = dev_step(x_test, y_test)
                    with open('results/prediction' + str(current_step), 'bw') as f:
                        pickle.dump(pred__, f)
                    if acc_tmp > best_accuracy:
                        best_accuracy = acc_tmp
                    print('best accuracy is', best_accuracy)
            print('Training is complete, testing the best model on x_test and y_test')

            # Evaluate x_test and y_test

            saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
                total_test_correct += int(num_test_correct)
            logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))

    # Save trained parameters and files since predict.py needs them
    with open(trained_dir + 'words_index.json', 'w') as outfile:
        json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
    with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
        pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
    with open(trained_dir + 'labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4, ensure_ascii=False)

    # os.rename(path, trained_dir + 'best_model.ckpt')
    # os.rename(path + '.meta', trained_dir + 'best_model.meta')
    shutil.rmtree(checkpoint_dir)
    logging.critical('{} has been removed'.format(checkpoint_dir))

    params['sequence_length'] = x_train.shape[1]
    with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    train_cnn_rnn()
