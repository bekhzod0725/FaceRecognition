"""
    Deep Convolutional Neural Network on LFW Dataset

    Author: Bekhzod Umarov <bumar1@unh.newhaven.edu>
    Course: Artificial Intelligence
    Year:   2016
"""

from __future__ import print_function

__author__  = "Bekhzod Umarov"
__year__    = "2016"


try:
    from tensorflow.contrib import skflow
except ImportError:
    import skflow

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import tensorflow as tf
import numpy as np
import random
import sys

EIGENFACE = False
IDEA      = False

# scikit-learn.fetch_lfw_people function automatically downloads, chaches,
# parses the metadata files, decodes the jpeg and converts the interesting
# slices into memmaped numpy arrays. This dataset is more than 200MB.
# The first load typically takes more than a couple of minutes.
# If the dataset has been loaded once, the following loading times take
# less than 200ms by using memmaped version memorized on the disk in the
# ~/scikit_learn_data/lfw_home/ folder uring jobLib.
lfw_people = fetch_lfw_people(min_faces_per_person=70,
                                resize=0.4)

###############################################################################
# Artificially Extend the dataset so that we have enough sample data to learn
random.seed(42)
n_samples, h, w = lfw_people.images.shape
for i in range(0, int(len(lfw_people.data) * 0.5) ):  # Flip
	_id = random.randint(0, len(lfw_people.data)-1)
	img = lfw_people.data[_id]
	img = np.fliplr(img.reshape(h,w) )
	img = img.flatten()
	lfw_people.data = np.vstack([lfw_people.data, img])
	lfw_people.target = np.concatenate( (lfw_people.target, [lfw_people.target[_id]]) )
for i in range(0, int(len(lfw_people.data) * 0.5) ):    # Noise
	_id = random.randint(0, len(lfw_people.data)-1)
	img = lfw_people.data[_id]
	img = img + np.random.poisson(256, img.shape).astype('float32')
	lfw_people.data = np.vstack([lfw_people.data, img])
	lfw_people.target = np.concatenate( (lfw_people.target, [lfw_people.target[_id]]) )

###############################################################################
# Randomize data
np.random.seed(42)
perm = np.random.permutation(lfw_people.target.size)
lfw_people.data = lfw_people.data[perm]
lfw_people.target = lfw_people.target[perm]


###############################################################################
# Image preprocessing
n_samples, h, w = lfw_people.images.shape
n_samples = len(lfw_people.data)
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]


###############################################################################
# Report what's going on
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

###############################################################################
# Split data into a training and testing set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
)


############################################################################
# Classification with Tensorflow Deep Neural Networks
print("Fitting the classifier to the training set")
t0 = time()     # Start time
i_lr = 0.0351   # Initial learning rate
n_step = 5000   # Number of steps
F_SCALE = 0.5   # Scale size

# Decay function
def exp_decay(global_step):
    return tf.train.exponential_decay(
            learning_rate=i_lr, global_step=global_step,
            decay_steps=28*128, decay_rate=0.001, staircase=True)

# Reshape tensor
def reshape_1x2(tensor_in):
    shape = tensor_in.get_shape().as_list()
    return tf.reshape(tensor_in, [-1,shape[1]*shape[2]*shape[3]])

# Max pool operation
def max_pool_2x2(tensor_in, k=2):
    return tf.nn.max_pool(tensor_in, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

# Model function
def conv_model(X, y):
    act = tf.tanh#tf.nn.relu#lambda x: tf.tanh(x) #tf.nn.relu(tf.tanh(x))
    X = tf.reshape(X, [-1, h, w, 1])
    batch_norm = True
    # Filters affecting the size of logits 64*130
    with tf.variable_scope('conv_layer1_1'):
        net = skflow.ops.conv2d(X, n_filters=int(64*F_SCALE), filter_shape=[3, 3],
                activation=act,batch_norm=batch_norm)
        net = tf.nn.relu(tf.tanh(net))
        net = max_pool_2x2(net, 4)
    with tf.variable_scope('conv_layer1_2'):
        net = skflow.ops.conv2d(net, n_filters=int(48*F_SCALE), filter_shape=[5, 5],
                activation=act,batch_norm=batch_norm)

    with tf.variable_scope('full_connection1'):
        net = reshape_1x2(net)
        net = skflow.ops.dnn(net, [2048], activation=act)
    return skflow.models.logistic_regression(net, y)

# System configuration
config = skflow.RunConfig(num_cores=8)
val_monitor = skflow.monitors.ValidationMonitor(X_test, y_test, early_stopping_rounds=200,
						n_classes=n_classes)
# Set Classifier
classifier = skflow.TensorFlowEstimator(
        #model_fn=dnn_tanh,
        model_fn            =   conv_model,
        n_classes           =   n_classes,
        batch_size          =   128,
        steps               =   n_step,
        optimizer           =   'Adagrad',
        continue_training   =   True,
    	config              =   config,
        learning_rate       =   i_lr 
)

#############################################################################
# Training based on model
#pipeline = Pipeline([('scaler', scaler), ('classifier', classifier)])
#classifier = skflow.TensorFlowLinearClassifier(n_classes=n_classes, steps=1000)
#classifier = pipeline
t0 = time()
print("Test Validation: ")

classifier.fit(X_train_pca, y_train, val_monitor)#, logdir='lfw_models/model_log/')
#print("done in %0.3fs" % (time() - t0))
#score = metrics.accuracy_score(y_train, classifier.predict(X_train_pca))
#print("Accuracy: %f" % score)

# Quantitative evaluation of the model quality on the test set
#y_pred = classifier.predict(X_test_pca)
score = metrics.accuracy_score(y_test, classifier.predict(X_test_pca))

print("Final accuracy: {:.9f}".format(score))
print("done in %0.3fs" % (time() - t0))


