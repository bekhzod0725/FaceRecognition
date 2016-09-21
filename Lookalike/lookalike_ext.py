"""
    Lookalike Database Facial Recognition using Deep Learning 
    Convolutional Neural Networks: EXTENDED VERSION

    Author: Bekhzod Umarov <bumar1@unh.newhaven.edu>
    Course: Artificial Intelligence
    Year:   2016
"""

from __future__ import print_function

from tensorflow.contrib import skflow

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import numpy as np
import sys
import random

from load_data import get_lookalike_people


###############################################################################
# Get data
lap = get_lookalike_people()


###############################################################################
# Artificially Extend dataset 
n_samples, h, w = lap.images.shape
random.seed(42)
for i in range(0, int(len(lap.data) * 0.5) ):       # Flip
    _id = random.randint(0, len(lap.data)-1)
    img = lap.data[_id]
    img = np.fliplr(img.reshape(h, w) )
    img = img.flatten()
    lap.data = np.vstack([lap.data, img ])
    lap.target = np.concatenate( (lap.target, [lap.target[_id]]) )
for i in range(0, int(len(lap.data) * 0.5) ):       # Noise
    _id = random.randint(0, len(lap.data)-1)
    img = lap.data[_id]
    img = img + np.random.poisson(256, img.shape).astype('float32')
    lap.data = np.vstack([lap.data, img])
    lap.target = np.concatenate( (lap.target, [lap.target[_id]]) )

# Randomize data
np.random.seed(42)
perm = np.random.permutation(lap.target.size)
lap.data = lap.data[perm]
lap.target = lap.target[perm]


n_samples, h, w = lap.images.shape
n_samples = len(lap.data)
X = lap.data
y = lap.target
target_names = lap.target_names
n_classes = target_names.shape[0]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
        )


print(n_samples)
print(n_classes)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

BATCH_SIZE            = 128
INITIAL_LEARNING_RATE = 0.0351
NUMBER_OF_STEPS       = 50000
EARLY_STOPPING_ROUNDS = 1000
F_SCALE               = 0.5

def reshape_1x2(tensor_in):
    shape = tensor_in.get_shape().as_list()
    return tf.reshape(tensor_in, [-1, shape[1]*shape[2]*shape[3]])
def max_pool_2x2(tensor_in, k=2):
    return tf.nn.max_pool(tensor_in, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def conv_model(X,y):
    act = tf.tanh
    batch_norm = True

    X = tf.reshape(X, [-1, h, w, 1])
    with tf.variable_scope('conv1'):
        net = skflow.ops.conv2d(X, n_filters=int(60*F_SCALE), filter_shape=[3,3],
                activation=act, batch_norm=batch_norm)
        net = tf.nn.relu(tf.tanh(net))
        net = max_pool_2x2(net, 4)
    with tf.variable_scope('conv2'):
        net = skflow.ops.conv2d(net, n_filters=48*F_SCALE, filter_shape=[5,5],
                activation=act, batch_norm=batch_norm)

    with tf.variable_scope('full_connection1'):
        net = reshape_1x2(net)
        net = skflow.ops.dnn(net, [1024], activation=act)
    return skflow.models.logistic_regression(net, y)

config = skflow.RunConfig(num_cores=8)
val_monitor = skflow.monitors.ValidationMonitor(X_test, y_test, 
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, n_classes=n_classes)

classifier = skflow.TensorFlowEstimator(
        model_fn = conv_model,
        n_classes = n_classes,
        batch_size = BATCH_SIZE,
        steps = NUMBER_OF_STEPS,
        optimizer = 'Adagrad',
        continue_training=True,
	    config=config,
        learning_rate = INITIAL_LEARNING_RATE
        )

t0 = time()
print ("Training:")
classifier.fit(X_train, y_train, val_monitor)

score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print("Final accuracy: {:0.9f}".format(score))
print("done in %0.3fs" % (time() - t0))

