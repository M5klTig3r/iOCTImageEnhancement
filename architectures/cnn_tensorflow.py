# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

def conv(input, weights, bias, strides=1):
    # with relu activation
    input = tf.nn.conv2d(input, weights, strides=[1, strides, strides, 1], padding='SAME')
    input = tf.nn.bias_add(input, bias)
    return tf.nn.relu(input)

def maxpool(input, kernel_size=2):
    return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1,kernel_size,kernel_size,1],padding='SAME')


# TODO - update these values!
training_iters = 10
learning_rate = 0.001
batch_size = 128

# network parameters
# n_input is nr. of inputs - image dimension
# n_classes is nr. of class labels
n_input = 28
n_classes = 10

# input & output placeholders
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])
