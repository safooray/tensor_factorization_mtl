import sys
sys.path.append('../TranSurvivalNet')
from cox_loss import cox_loss_stable
from survival_analysis import calc_at_risk
from utils.data_utils import load_npy_data
from utils.data_utils import pca 

import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
import time

data_path = '../TranSurvivalNet/data/gene/'
datasets = ['BRCA_gene.npy', 'HNSC_gene.npy', 'KIRC_gene.npy']
event_labs = ['BRCA_event.npy', 'HNSC_event.npy', 'KIRC_event.npy']
time_labs = ['BRCA_os.npy', 'HNSC_os.npy', 'KIRC_os.npy']
T = 3  # number of tasks
H = 50

D = 20531
X = []
Y = []
A = []
brca_x, brca_t, brca_o = load_npy_data(data_path, acronym='BRCA')
hnsc_x, hnsc_t, hnsc_o = load_npy_data(data_path, acronym='HNSC')
kirc_x, kirc_t, kirc_o = load_npy_data(data_path, acronym='KIRC')


X = [brca_x, hnsc_x, kirc_x]
E = [brca_o, hnsc_o, kirc_o]
Y = [brca_t, hnsc_t, kirc_t]
A = [None] * T
for i in range(T):
    X[i], Y[i], E[i], A[i] = calc_at_risk(X[i], Y[i], E[i])


#Single Task Learning
with tf.Graph().as_default():
    X_placeholder = [tf.placeholder(tf.float32, shape=[n, D]) for n in [x.shape[0] for x in X]]
    E_placeholder = [tf.placeholder(tf.float32, shape=[n,]) for n in [x.shape[0] for x in X]]
    A_placeholder = [tf.placeholder(tf.int32, shape=[n,]) for n in [x.shape[0] for x in X]]

    W_input_to_hidden = [tf.Variable(tf.truncated_normal(shape=[D, H])) for _ in range(T)]
    b_input_to_hidden = [tf.Variable(tf.zeros(shape=[H])) for _ in range(T)]
    W_hidden_to_output = [tf.Variable(tf.truncated_normal(shape=[H,1])) for _ in range(T)]
    b_hidden_to_output = [tf.Variable(tf.zeros(shape=[1])) for _ in range(T)]

    Y_hat = [tf.nn.xw_plus_b(tf.nn.sigmoid(tf.nn.xw_plus_b(x,w0,b0)),w1,b1) 
             for x,w0,b0,w1,b1 in zip(X_placeholder, W_input_to_hidden, b_input_to_hidden, W_hidden_to_output, b_hidden_to_output)]

    loss = tf.reduce_mean([cox_loss_stable(pred, a, e) for pred, a, e in zip(Y_hat, A_placeholder, E_placeholder)])
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    t3 = time.time()
    train = opt.minimize(loss)
    t4 = time.time()
    print(t4-t3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = dict(list(zip(X_placeholder,X))+list(zip(E_placeholder,E))+list(zip(A_placeholder,A)))
        t5 = time.time()
        print(t5-t4)
        for _ in range(100):
            train.run(feed_dict=feed_dict)
            if _ % 10 == 0:
                print(loss.eval(feed_dict=feed_dict))
        
        W_init = np.stack(sess.run(W_input_to_hidden))
        Y_hat_STL = sess.run(Y_hat, feed_dict=feed_dict)

# Multi-task Learning with Tensor Factorisation
with tf.Graph().as_default():
    X_placeholder = [tf.placeholder(tf.float32, shape=[n, D]) for n in [x.shape[0] for x in X]]
    E_placeholder = [tf.placeholder(tf.float32, shape=[n,]) for n in [x.shape[0] for x in X]]
    A_placeholder = [tf.placeholder(tf.int32, shape=[n,]) for n in [x.shape[0] for x in X]]

    W_init = np.transpose(W_init, axes=[1,2,0])
    W_input_to_hidden, W_factors = TensorProducer(W_init, 'LAF', eps_or_k=0.1, return_true_var=True)
    W_input_to_hidden = [W_input_to_hidden[:,:,i] for i in range(T)]
    b_input_to_hidden = [tf.Variable(tf.zeros(shape=[H])) for _ in range(T)]
    W_hidden_to_output = [tf.Variable(tf.truncated_normal(shape=[H,1])) for _ in range(T)]
    b_hidden_to_output = [tf.Variable(tf.zeros(shape=[1])) for _ in range(T)]
    
    Y_hat = [tf.nn.xw_plus_b(tf.nn.sigmoid(tf.nn.xw_plus_b(x,w0,b0)),w1,b1) 
             for x,w0,b0,w1,b1 in zip(X_placeholder, W_input_to_hidden, b_input_to_hidden, W_hidden_to_output, b_hidden_to_output)]

    loss = tf.reduce_mean([cox_loss_stable(pred, a, e) for pred, a, e in zip(Y_hat, A_placeholder, E_placeholder)])
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = opt.minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = dict(list(zip(X_placeholder,X))+list(zip(E_placeholder,E))+list(zip(A_placeholder,A)))
        for _ in range(1000):
            train.run(feed_dict=feed_dict)
            if _ % 10 == 0:
                print(loss.eval(feed_dict=feed_dict))
        
        Y_hat_MTL = sess.run(Y_hat, feed_dict=feed_dict)
