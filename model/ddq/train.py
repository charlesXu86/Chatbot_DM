# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   train.py
 
@Time    :   2019-09-28 15:16
 
@Desc    :
 
'''

import sys
sys.path.append('./ddq')

import tensorflow as tf
from copy import deepcopy
from dqn_dataloader import Dataloader
import datetime
from random import randint
from dqn_model import DQN
from tensorflow import flags
import os

flags.DEFINE_integer("evaluate_every", 100,
                     "Number of step for model evaluation.")
flags.DEFINE_integer("batch_size", 64, "batch size for training")

flags.DEFINE_integer("epoch", 1000000, "max epoch")

flags.DEFINE_string("model_dir", "models/", "The directory to save the model files in.")
flags.DEFINE_string("gpu", "2, 3", "the gpu to use")

FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

config = tf.ConfigProto(
    allow_soft_placement=True
    # log_device_placement=True
)
config.gpu_options.per_process_gpu_memory_fraction = 0.4


def train():
    dataloader = Dataloader()
    dataloader.split(0.1)
    val_data = dataloader.get_val()
    dev_st_batch = [d[0] for d in val_data]
    dev_at_batch = [d[1] for d in val_data]
    dev_st1_batch = [d[2] for d in val_data]
    dev_rt_batch = [d[3] for d in val_data]
    dev_terminal_batch = [d[4] for d in val_data]
    global dqn, sess
    with tf.Graph().as_default():

        sess = tf.Session(config=config)
        with sess.as_default():
            # check if a saved model
            # meta_filename = get_meta_filename(False, FLAGS.model_dir)
            # print("meta filename = ", meta_filename)
            # if meta_filename:
            # saver = recover_model(meta_filename)
            # saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
            # print("restoring the model...")
            dqn = DQN(goal_size=dataloader.goal_size,
                      act_size=dataloader.action_size)

            # check whether checkpoint exist if yes, load the checkpoint, if not
            # , initial the variables
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt:
                print("Reading ckpt model from %s" % ckpt.model_checkpoint_path)
                dqn.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Create model with fresh parameters")
                # Initialize all variables
                sess.run(tf.global_variables_initializer())

            for i in range(FLAGS.epoch):
                train_batch = dataloader.get_train_batch(FLAGS.batch_size)
                st_batch = [d[0] for d in train_batch]
                at_batch = [d[1] for d in train_batch]
                st1_batch = [d[2] for d in train_batch]
                rt_batch = [d[3] for d in train_batch]
                terminal_batch = [d[4] for d in train_batch]
                target_q_batch, _ = max_q(st_batch, st1_batch)
                train_step(st_batch, at_batch, target_q_batch,
                           rt_batch, terminal_batch)
                current_step = tf.train.global_step(sess, dqn.global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    dev_target_q_batch, dev_q_action_batch = max_q(dev_at_batch, dev_st1_batch)
                    dev_step(dev_st_batch, dev_at_batch, dev_target_q_batch,
                             dev_rt_batch, dev_terminal_batch)
                    idx = randint(0, len(dev_st_batch))
                    print("State: " + str(dataloader.mapState(dev_st1_batch[idx])))
                    print("Action: " + str(dataloader.mapAction(dev_q_action_batch[idx])))
                # saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                if current_step % 1000 == 0:
                    # _, q_action = max_q()
                    saver = dqn.saver
                    saver.save(sess, FLAGS.model_dir + '/test_model', global_step=dqn.global_step)


# predict the max Q of st1
def max_q(st_batch, st1_batch, action_size=9):
    tmp = []
    all_action = []
    all_st1 = []
    # generate all possible action
    for s in st_batch:
        for i in range(8):
            a0 = list(deepcopy(s))
            idx_1 = i % 2
            idx_2 = (i / 2) % 2
            idx_3 = (i / 4) % 2
            if idx_1 == 0:
                a0[1] = 0
            if idx_2 == 0:
                a0[2] = 0
            if idx_3 == 0:
                a0[3] = 0
            tmp.append(tuple(a0))
    for a in tmp:
        for i in range(action_size):
            a0 = list(deepcopy(a))
            a0[0] = i
            all_action.append(tuple(a0))
    for s in st1_batch:
        for i in range(8 * action_size):
            all_st1.append(s)
    st_goal_batch = [s[0] for s in all_st1]
    st_song_batch = [s[1] for s in all_st1]
    st_singer_batch = [s[2] for s in all_st1]
    st_album_batch = [s[3] for s in all_st1]
    at_act_batch = [a[0] for a in all_action]
    at_song_batch = [a[1] for a in all_action]
    at_singer_batch = [a[2] for a in all_action]
    at_album_batch = [a[3] for a in all_action]
    feed_dict = {
        dqn.state_goal: st_goal_batch,
        dqn.state_song: trained_slot(st_song_batch),
        dqn.state_singer: trained_slot(st_singer_batch),
        dqn.state_album: trained_slot(st_album_batch),
        dqn.action_act: at_act_batch,
        dqn.action_song: trained_slot(at_song_batch),
        dqn.action_singer: trained_slot(at_singer_batch),
        dqn.action_album: trained_slot(at_album_batch)
    }
    q = sess.run(
        dqn.q,
        feed_dict
    )
    max_q = []
    max_action = []
    for i in range(0, len(q), 8 * action_size):
        q_max = float('-inf')
        for j in range(8 * action_size):
            if q[i + j] > q_max:
                q_max = q[i + j]
                q_action = all_action[i + j]
        max_q.append(q_max)
        max_action.append(q_action)
    return max_q, max_action


def dev_step(st_batch, at_batch, target_q_batch,
             rt_batch, terminal_batch):
    print(target_q_batch[:10])
    st_goal_batch = [s[0] for s in st_batch]
    st_song_batch = [s[1] for s in st_batch]
    st_singer_batch = [s[2] for s in st_batch]
    st_album_batch = [s[3] for s in st_batch]
    at_act_batch = [a[0] for a in at_batch]
    at_song_batch = [a[1] for a in at_batch]
    at_singer_batch = [a[2] for a in at_batch]
    at_album_batch = [a[3] for a in at_batch]
    feed_dict = {
        dqn.state_goal: st_goal_batch,
        dqn.state_song: trained_slot(st_song_batch),
        dqn.state_singer: trained_slot(st_singer_batch),
        dqn.state_album: trained_slot(st_album_batch),
        dqn.action_act: at_act_batch,
        dqn.action_song: trained_slot(at_song_batch),
        dqn.action_singer: trained_slot(at_singer_batch),
        dqn.action_album: trained_slot(at_album_batch),
        dqn.target_q: target_q_batch,
        dqn.reward: rt_batch,
        dqn.terminal: terminal_batch
    }
    step, loss, q = sess.run(
        [dqn.global_step, dqn.loss, dqn.q],
        feed_dict)

    time_str = datetime.datetime.now().isoformat()
    print("Dev: {}: step {}, loss {:g}".format(time_str, step, loss))


def train_step(st_batch, at_batch, target_q_batch,
               rt_batch, terminal_batch):
    at_act_batch = [a[0] for a in at_batch]
    at_song_batch = [a[1] for a in at_batch]
    at_singer_batch = [a[2] for a in at_batch]
    at_album_batch = [a[3] for a in at_batch]

    st_goal_batch = [s[0] for s in st_batch]
    st_song_batch = [s[1] for s in st_batch]
    st_singer_batch = [s[2] for s in st_batch]
    st_album_batch = [s[3] for s in st_batch]
    feed_dict = {
        dqn.state_goal: st_goal_batch,
        dqn.state_song: trained_slot(st_song_batch),
        dqn.state_singer: trained_slot(st_singer_batch),
        dqn.state_album: trained_slot(st_album_batch),
        dqn.action_act: at_act_batch,
        dqn.action_song: trained_slot(at_song_batch),
        dqn.action_singer: trained_slot(at_singer_batch),
        dqn.action_album: trained_slot(at_album_batch),
        dqn.target_q: target_q_batch,
        dqn.reward: rt_batch,
        dqn.terminal: terminal_batch
    }
    _, step, loss = sess.run(
        [dqn.train_op, dqn.global_step, dqn.loss],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    if step % 50 == 0:
        print("Train: {}: step {}, loss {:g}".format(time_str, step, loss))


def trained_slot(slots):
    return [1 if s >= 1 else 0 for s in slots]


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
