import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf



with tf.Session() as session:
    a = tf.Variable(tf.random_normal([3, 3, 3]))
    session.run(tf.global_variables_initializer())  # initial all variable
    print(session.run(a))

    b=tf.constant([[ 124368.984375  -116721.9765625]],dtype=tf.float32)

    c = tf.nn.softmax(b)
    print(session.run(c))

