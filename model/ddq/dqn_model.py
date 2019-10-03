# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   dqn_model.py
 
@Time    :   2019-09-28 15:15
 
@Desc    :
 
'''

import tensorflow as tf


class DQN(object):
    GAMMA = 0.9

    def __init__(self, goal_size=10, act_size=10, song_size=2, singer_size=2, album_size=2):
        self.state_goal = tf.placeholder(tf.int32, [None], name="state_goal")
        self.state_song = tf.placeholder(tf.int32,
                                         [None], name="state_song")
        self.state_singer = tf.placeholder(tf.int32,
                                           [None], name="state_singer")
        self.state_album = tf.placeholder(tf.int32,
                                          [None], name="state_album")
        self.action_act = tf.placeholder(tf.int32,
                                         [None], name="action_act")
        self.action_song = tf.placeholder(tf.int32,
                                          [None], name="action_song")
        self.action_singer = tf.placeholder(tf.int32,
                                            [None], name="action_singer")
        self.action_album = tf.placeholder(tf.int32,
                                           [None], name="action_album")
        self.target_q = tf.placeholder(tf.float32,
                                       [None], name="target_q")
        self.reward = tf.placeholder(tf.float32,
                                     [None], name="reward")
        self.terminal = tf.placeholder(tf.float32,
                                       [None], name="terminal")

        # one hot embedding for each slot and goal, action
        with tf.device("/gpu:0"):
            with tf.name_scope("embedding"):
                self.goal_w = tf.Variable(tf.random_uniform([goal_size, 2],
                                                            name="goal_w"))
                self.act_w = tf.Variable(tf.random_uniform([act_size, 2],
                                                           name="act_w"))
                self.song_w = tf.Variable(tf.random_uniform([song_size, 1],
                                                            name="song_w"))
                self.singer_w = tf.Variable(tf.random_uniform([singer_size, 1],
                                                              name="singer_w"))
                self.album_w = tf.Variable(tf.random_uniform([album_size, 1],
                                                             name="album_w"))
                self.st_goal = tf.nn.embedding_lookup(self.goal_w, self.state_goal)
                self.st_song = tf.nn.embedding_lookup(self.song_w, self.state_song)
                self.st_singer = tf.nn.embedding_lookup(self.singer_w,
                                                        self.state_singer)
                self.st_album = tf.nn.embedding_lookup(self.album_w,
                                                       self.state_album)
                self.at_act = tf.nn.embedding_lookup(self.act_w, self.action_act)
                self.at_song = tf.nn.embedding_lookup(self.song_w,
                                                      self.action_song)
                self.at_singer = tf.nn.embedding_lookup(self.singer_w,
                                                        self.action_singer)
                self.at_album = tf.nn.embedding_lookup(self.album_w,
                                                       self.action_album)
                self.st = tf.concat([self.st_goal, self.st_song,
                                     self.st_singer, self.st_album], 1)  # 32 dim
                self.at = tf.concat([self.at_act, self.at_song,
                                     self.at_singer, self.at_album], 1)  # 32 dim

            #  model architecture
            with tf.name_scope("MLP"):
                w0_st = tf.Variable(
                    tf.truncated_normal([5, 10], stddev=0.1), name="w0_st")
                b0_st = tf.Variable(tf.constant(0.1, shape=[10]), name="b0_st")
                self.h0_st = tf.nn.tanh(tf.nn.xw_plus_b(self.st, w0_st, b0_st))
                w0_at = tf.Variable(
                    tf.truncated_normal([5, 10], stddev=0.1), name="w0_at")
                b0_at = tf.Variable(tf.constant(0.1, shape=[10]), name="b0_at")
                self.h0_at = tf.nn.tanh(tf.nn.xw_plus_b(self.at, w0_at, b0_at))
                self.h0 = tf.concat([self.h0_st, self.h0_at], 1)
                self.w1 = tf.Variable(
                    tf.truncated_normal([20, 1], stddev=0.1), name="w1")
                self.b1 = tf.Variable(tf.constant(0.1, shape=[1]), name="b1")
                self.h1 = tf.nn.xw_plus_b(self.h0, self.w1, self.b1)
                self.q = tf.reshape(self.h1, [-1], name="q") * 30

            # target reward
            with tf.name_scope("target"):
                self.target = self.reward + \
                              self.GAMMA * tf.multiply(self.terminal, self.target_q)

            # loss function
            with tf.name_scope("loss"):
                self.loss = tf.reduce_sum(tf.pow(self.target - self.q, 2)) / 64

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                       global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

