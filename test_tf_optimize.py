import sys
sys.path.append('../TranSurvivalNet')
from tf_cox_loss import cox_loss
import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
from tensor_toolbox_yyang import TensorProducer
import time

T = 3  # number of tasks
H = 50

D = 300 
N = 500
X = [np.random.randn(N,D) for _ in range(T)]
E = [np.random.binomial(1, 0.3, size=(N, 1)) for _ in range(T)]
A = [np.reshape(np.arange(N), (N, 1)) for _ in range(T)]

#Single Task Learning
with tf.device("/cpu:0"):
    with tf.Graph().as_default():
        X_placeholder = [tf.placeholder(tf.float32, shape=[n, D]) for n in [x.shape[0] for x in X]]
        A_placeholder = [tf.placeholder(tf.int32, shape=[n, 1]) for n in [x.shape[0] for x in X]]
        E_placeholder = [tf.placeholder(tf.float32, shape=[n, 1]) for n in [x.shape[0] for x in X]]

        W_input_to_hidden = [tf.Variable(tf.truncated_normal(shape=[D, H])) for _ in range(T)]
        b_input_to_hidden = [tf.Variable(tf.zeros(shape=[H])) for _ in range(T)]
        W_hidden_to_output = [tf.Variable(tf.truncated_normal(shape=[H,1])) for _ in range(T)]
        b_hidden_to_output = [tf.Variable(tf.zeros(shape=[1])) for _ in range(T)]

        Y_hat = [tf.nn.xw_plus_b(tf.nn.sigmoid(tf.nn.xw_plus_b(x,w0,b0)),w1,b1) 
                 for x,w0,b0,w1,b1 in zip(X_placeholder, W_input_to_hidden, b_input_to_hidden, W_hidden_to_output, b_hidden_to_output)]
        #MSE = [tf.reduce_mean(tf.squared_difference(y,y_hat)) for y,y_hat in zip(E_placeholder,Y_hat)]
        #loss = tf.reduce_mean(MSE)
        loss = tf.reduce_mean([cox_loss(pred, a, e) for pred, a, e in zip(Y_hat, A_placeholder, E_placeholder)])
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        t3 = time.time()
        train = opt.minimize(loss)
        t4 = time.time()
        print(t4-t3)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            W_init = sess.run(W_input_to_hidden)
            feed_dict = dict(list(zip(X_placeholder,X))+list(zip(E_placeholder,E))+list(zip(A_placeholder,A)))
            t5 = time.time()
            print(t5-t4)
            for _ in range(10):
                train.run(feed_dict=feed_dict)
                if _ % 1 == 0:
                    print(loss.eval(feed_dict=feed_dict))
            
            W_val, preds_val = sess.run([W_input_to_hidden, Y_hat], feed_dict=feed_dict)
            W_val = np.stack(W_val)
