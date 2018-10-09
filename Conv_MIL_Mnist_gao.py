#!/usr/bin/env python

"""
author: dan salo
initial commit: 12/1/2016

purpose: implement convolutional multiple instance learning for distributed learning over mnist dataset
"""

# import sys
# sys.path.append('../')

import matplotlib
from tensorbase.base import Model
from tensorbase.base import Layers

import tensorflow as tf
import numpy as np
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


class convmil(Model):
    def __init__(self, flags_input,run_num):
        """ initialize from model class in tensorbase """
        super().__init__(flags_input,run_num)
        self.valid_results = list()
        self.test_results = list()
        self.checkpoint_rate = 5  # save after this many epochs
        self.valid_rate = 5  # validate after this many epochs

    def _data(self):
        """ define all data-related parameters. called by tensorbase. """
        self.mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
        self.num_train_images = self.mnist.train.num_examples
        self.num_valid_images = self.mnist.validation.num_examples
        self.num_test_images = self.mnist.test.num_examples
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, self.flags['num_classes']], name='y')

    def _summaries(self):
        """ write summaries out to tensorboard. called by tensorbase. """
        tf.summary.scalar("total_loss", self.cost)
        tf.summary.scalar("xentropy_loss_pi", self.xentropy_p)
        tf.summary.scalar("xentropy loss_yi", self.xentropy_y)
        tf.summary.scalar("weight_decay_loss", self.weight)

    def _network(self):
        """ define neural network. uses layers class of tensorbase. called by tensorbase. """
        with tf.variable_scope("model"):
            net = Layers(self.x)
            net.conv2d(5, 64)
            net.maxpool()
            net.conv2d(3, 64)
            net.conv2d(3, 64)
            net.maxpool()
            net.conv2d(3, 128)
            net.conv2d(3, 128)
            net.maxpool()
            net.conv2d(1, self.flags['num_classes'], activation_fn=tf.nn.sigmoid)
            net.noisy_and(self.flags['num_classes'])
            self.p_i = net.get_output()
            net.fc(self.flags['num_classes'])
            self.y_hat = net.get_output()
            self.logits = tf.nn.softmax(self.y_hat)

    def _optimizer(self):
        """ set up loss functions and choose optimizer. called by tensorbase. """
        const = 1/self.flags['batch_size']
        self.xentropy_p = const * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.p_i, name='xentropy_p'))
        self.xentropy_y = const * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat, name='xentropy_y'))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.xentropy_p + self.xentropy_y + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.flags['learning_rate']).minimize(self.cost)

    def train(self):
        """ run training function for num_epochs. save model upon completion. """
        print('training for %d epochs' % self.flags['num_epochs'])

        for self.epoch in range(1, self.flags['num_epochs'] + 1):
            for _ in tqdm(range(self.num_train_images)):

                # get minibatches of data
                batch_x, batch_y = self.mnist.train.next_batch(self.flags['batch_size'])
                batch_x = self.reshape_batch(batch_x)

                # run a training iteration

                summary, loss, _ = self.sess.run([self.merged, self.cost, self.optimizer],
                                                   feed_dict={self.x: batch_x, self.y: batch_y})
                self._record_training_step(summary)
            if self.step % (self.flags['display_step']) == 0:
                # record training metrics every display_step interval
                self._record_train_metrics(loss)

            ## epoch finished
            # save model
            if self.epoch % self.checkpoint_rate == 0:
                self._save_model(section=self.epoch)
            # perform validation
            if self.epoch % self.valid_rate == 0:
                self.evaluate('valid')

    def evaluate(self, dataset):
        """ evaluate network on the valid/test set. """

        # initialize to correct dataset
        print('evaluating images in %s set' % dataset)
        if dataset == 'valid':
            batch_x, batch_y = self.mnist.validation.next_batch(self.flags['batch_size'])
            batch_x = self.reshape_batch(batch_x)
            num_images = self.num_valid_images
            results = self.valid_results
        else:
            batch_x, batch_y = self.mnist.test.next_batch(self.flags['batch_size'])
            batch_x = self.reshape_batch(batch_x)
            num_images = self.num_test_images
            results= self.test_results

        # loop through all images in eval dataset
        for _ in tqdm(range(num_images)):
            logits = self.sess.run([self.logits], feed_dict={self.x: batch_x})
            predictions = np.reshape(logits, [-1, self.flags['num_classes']])
            self.valid_batch_y = batch_y
            correct_prediction = np.equal(np.argmax(self.valid_batch_y, 1), np.argmax(predictions, 1))
            results = np.concatenate((results, correct_prediction))
            print(results)
        # calculate average accuracy and record in text file
        # self.record_eval_metrics(dataset)

    #########################
    ##   helper functions  ##
    #########################

    def reshape_batch(self, batch):
        """ reshape vector into image. do not need if data that is loaded in is already in image-shape"""
        return np.reshape(batch, [self.flags['batch_size'], 28, 28, 1])

    def _record_train_metrics(self, loss):
        """ records the metrics at every display_step iteration """
        print("batch number: " + str(self.step) + ", total loss= " + "{:.6f}".format(loss))

    def _record_eval_metrics(self, dataset):
        """ record the accuracy on the eval dataset """
        if dataset == 'valid':
            accuracy = np.mean(self.valid_results)
        else:
            accuracy = np.mean(self.test_results)
        print("accuracy on %s set: %f" % (dataset, float(accuracy)))
        file = open(self.flags['restore_directory'] + dataset + 'accuracy.txt', 'w')
        file.write('%s set accuracy:' % dataset)
        file.write(str(accuracy))
        file.close()


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='faster r-cnn networks arguments')
    # parser.add_argument('-n', '--run_num', default=0)  # saves all under /save_directory/model_directory/model[n]
    # parser.add_argument('-e', '--num_epochs', default=1)  # number of epochs for which to train the model
    # parser.add_argument('-r', '--restore_meta', default=0)  # binary to restore from a model. 0 = no restore.
    # parser.add_argument('-m', '--model_restore', default=1)  # restores from /save_directory/model_directory/model[n]
    # parser.add_argument('-f', '--file_epoch', default=1)  # restore filename: 'part_[f].ckpt.meta'
    # parser.add_argument('-t', '--train', default=1)  # binary to train model. 0 = no train.
    # parser.add_argument('-v', '--eval', default=1)  # binary to evaluate model. 0 = no eval.
    # parser.add_argument('-l', '--learning_rate', default=1e-3, type=float)  # learning rate
    # parser.add_argument('-g', '--gpu', default=0)  # specify which gpu to use
    # parser.add_argument('-s', '--seed', default=123)  # specify the seed
    # parser.add_argument('-d', '--model_directory', default='summaries/', type=str)  # to save all models
    # parser.add_argument('-a', '--save_directory', default='d:/pythonfile/mil/convmil/', type=str)  # to save individual run
    # # parser.add_argument('-a', '--save_directory', default="d:\\pythonfile\\", type=str)
    # parser.add_argument('-i', '--display_step', default=500, type=int)  # how often to display metrics
    # parser.add_argument('-b', '--batch_size', default=128, type=int)  # size of minibatch
    # parser.add_argument('-w', '--weight_decay', default=1e-7, type=float)  # decay on all weight variables
    # parser.add_argument('-c', '--num_classes', default=10, type=int)  # number of classes. proly hard code.
    parser.add_argument('-n', '--run_num', default=0)  # saves all under /save_directory/model_directory/model[n]
    parser.add_argument('-e', '--num_epochs', default=1)  # number of epochs for which to train the model
    parser.add_argument('-r', '--restore_meta', default=0)  # binary to restore from a model. 0 = no restore.
    parser.add_argument('-m', '--model_restore', default=1)  # restores from /save_directory/model_directory/model[n]
    # parser.add_argument('-m', '--restore', default=0)  # restores from /save_directory/model_directory/model[n]
    parser.add_argument('-f', '--file_epoch', default=1)  # restore filename: 'part_[f].ckpt.meta'
    parser.add_argument('-t', '--train', default=0)  # binary to train model. 0 = no train.
    parser.add_argument('-v', '--eval', default=1)  # binary to evaluate model. 0 = no eval.
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float)  # learning rate
    parser.add_argument('-g', '--gpu', default=0)  # specify which gpu to use
    parser.add_argument('-s', '--seed', default=123)  # specify the seed
    parser.add_argument('-d', '--model_directory', default='summaries/', type=str)  # to save all models
    parser.add_argument('-a', '--save_directory', default='d:/pythonfile/mil/convmil/', type=str)  # to save individual run
    # parser.add_argument('-a', '--save_directory', default="d:\\pythonfile\\", type=str)
    parser.add_argument('-i', '--display_step', default=500, type=int)  # how often to display metrics
    parser.add_argument('-b', '--batch_size', default=128, type=int)  # size of minibatch
    parser.add_argument('-w', '--weight_decay', default=1e-7, type=float)  # decay on all weight variables
    parser.add_argument('-c', '--num_classes', default=10, type=int)  # number of classes. proly hard code.
    flags = vars(parser.parse_args())

    # run model. train and/or eval.
    model = convmil(flags, run_num=flags['run_num'])
    if int(flags['train']) == 1:
        model.train()
    if int(flags['eval']) == 1:
        model.evaluate('test')

if __name__ == "__main__":
    main()

