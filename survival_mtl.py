import sys
sys.path.append('../TranSurvivalNet')
from IPython import embed
from losses.tf_cox_loss import cox_loss
from survival_analysis import calc_at_risk, c_index
from utils.data_utils import load_pca_npy_data
from utils.data_utils import pca 

import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
from tensor_toolbox_yyang import TensorProducer

data_path = '../TranSurvivalNet/data/gene_pca/'
T = 3  # number of tasks
H = 50

D = 300 
X = []
Y = []
A = []
ACROS = ['BRCA', 'OV', 'HNSC', 'KIRC']
brca_x, brca_t, brca_o = load_pca_npy_data(data_path, acronym='BRCA')
hnsc_x, hnsc_t, hnsc_o = load_pca_npy_data(data_path, acronym='HNSC')
kirc_x, kirc_t, kirc_o = load_pca_npy_data(data_path, acronym='KIRC')


X = [brca_x, hnsc_x, kirc_x]
C = [1-brca_o, 1-hnsc_o, 1-kirc_o]
Y = [brca_t, hnsc_t, kirc_t]
#A = [None] * T
#for i in range(T):
#    X[i], Y[i], E[i], A[i] = calc_at_risk(X[i], Y[i], E[i])


#Single Task Learning
with tf.device("/cpu:0"):
    with tf.Graph().as_default():
        X_placeholder = [tf.placeholder(tf.float32, shape=[n, D]) for n in [x.shape[0] for x in X]]
        C_placeholder = [tf.placeholder(tf.float32, shape=[n,]) for n in [x.shape[0] for x in X]]
        Y_placeholder = [tf.placeholder(tf.float32, shape=[n,]) for n in [x.shape[0] for x in X]]

        W_input_to_hidden = [tf.Variable(tf.truncated_normal(shape=[D, H])) for _ in range(T)]
        b_input_to_hidden = [tf.Variable(tf.zeros(shape=[H])) for _ in range(T)]
        W_hidden_to_output = [tf.Variable(tf.truncated_normal(shape=[H,1])) for _ in range(T)]
        b_hidden_to_output = [tf.Variable(tf.zeros(shape=[1])) for _ in range(T)]

        Y_hat = [tf.nn.xw_plus_b(tf.nn.sigmoid(tf.nn.xw_plus_b(x,w0,b0)),w1,b1) 
                 for x,w0,b0,w1,b1 in zip(X_placeholder, W_input_to_hidden, b_input_to_hidden, W_hidden_to_output, b_hidden_to_output)]

        loss = tf.reduce_mean([cox_loss(pred, t, c) for pred, t, c in zip(Y_hat, Y_placeholder, C_placeholder)])
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train = opt.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = dict(list(zip(X_placeholder,X))+list(zip(Y_placeholder,Y))+list(zip(C_placeholder,C)))
            for _ in range(100):
                train.run(feed_dict=feed_dict)
                if _ % 10 == 0:
                    print(loss.eval(feed_dict=feed_dict))
            
            preds_val = sess.run(Y_hat, feed_dict=feed_dict)
            W_init = sess.run(W_input_to_hidden)

for i in range(T):
    ci = c_index(preds_val[i], Y[i], C[i])
    print("MTL ci {} = {}".format(ACROS[i], ci))

# Multi-task Learning with Tensor Factorisation
with tf.device("/cpu:0"):
    with tf.Graph().as_default():
        X_placeholder = [tf.placeholder(tf.float32, shape=[n, D]) for n in [x.shape[0] for x in X]]
        C_placeholder = [tf.placeholder(tf.float32, shape=[n,]) for n in [x.shape[0] for x in X]]
        Y_placeholder = [tf.placeholder(tf.float32, shape=[n,]) for n in [x.shape[0] for x in X]]

        W_init = np.stack(W_init, axis=0)
        W_init = np.transpose(W_init, axes=[1,2,0])
        W_input_to_hidden, W_factors = TensorProducer(W_init, 'LAF', eps_or_k=0.1, return_true_var=True)
        W_input_to_hidden = [W_input_to_hidden[:,:,i] for i in range(T)]
        b_input_to_hidden = [tf.Variable(tf.zeros(shape=[H])) for _ in range(T)]
        W_hidden_to_output = [tf.Variable(tf.truncated_normal(shape=[H,1])) for _ in range(T)]
        b_hidden_to_output = [tf.Variable(tf.zeros(shape=[1])) for _ in range(T)]
        
        Y_hat = [tf.nn.xw_plus_b(tf.nn.sigmoid(tf.nn.xw_plus_b(x,w0,b0)),w1,b1) 
                 for x,w0,b0,w1,b1 in zip(X_placeholder, W_input_to_hidden, b_input_to_hidden, W_hidden_to_output, b_hidden_to_output)]

        loss = tf.reduce_mean([cox_loss(pred, t, c) for pred, t, c in zip(Y_hat, Y_placeholder, C_placeholder)])
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train = opt.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = dict(list(zip(X_placeholder,X))+list(zip(Y_placeholder,Y))+list(zip(C_placeholder,C)))
            for _ in range(1000):
                train.run(feed_dict=feed_dict)
                if _ % 10 == 0:
                    print(loss.eval(feed_dict=feed_dict))
            
            preds_val = sess.run(Y_hat, feed_dict=feed_dict)
            W_val, W_factors_val = sess.run([W_input_to_hidden, W_factors])
            embed()

for i in range(T):
    ci = c_index(preds_val[i], Y[i], C[i])
    print("MTL ci {} = {}".format(ACROS[i], ci))
