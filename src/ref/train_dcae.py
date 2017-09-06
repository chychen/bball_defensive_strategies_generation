from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import model_dcae

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string("train_data", "train_data.npy",
                           "training data name")
tf.app.flags.DEFINE_string("valid_data", "test_data.npy",
                           "validation data name")
tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'v6/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'v6/log/',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")
# data augmentation and corruption
tf.app.flags.DEFINE_integer('aug_ratio', 4,
                            "the ratio of data augmentation")
tf.app.flags.DEFINE_integer('corrupt_amount', 100,
                            "the amount of corrupted data")
# training parameters
FILTER_NUMBERS = [32, 64, 128]
FILTER_STRIDES = [1, 2, 2]
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 100,
                            "total training epoches")
tf.app.flags.DEFINE_integer('save_freq', 25,
                            "number of epoches to saving model")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          "learning rate of AdamOptimizer")
# tf.app.flags.DEFINE_integer('num_gpus', 2,
#                             "multi gpu")

def data_normalization(data):
    # normalize each dims [t, d, f, s, w]
    for i in range(5):
        temp_mean = np.mean(data[:, :, :, i])
        temp_std = np.std(data[:, :, :, i])
        data[:, :, :, i] = (data[:, :, :, i] - temp_mean) / temp_std
        print(i, temp_mean, temp_std)
    return data

def generate_input_and_label(all_data, aug_ratio, corrupt_amount):
    print('all_data.shape:', all_data.shape)
    # data augmentation
    aug_data = []
    for one_data in all_data:
        aug_data.append([one_data for _ in range(aug_ratio)])
    aug_data = np.concatenate(aug_data, axis=0)
    raw_data = np.array(aug_data[:, :, :,1:4]) # only [d, f, s]
    print('raw_data.shape:', raw_data.shape)
    # randomly corrupt target data
    for one_data in aug_data:
        corrupt_target = np.random.randint(all_data.shape[1] * all_data.shape[2],
                                           size=corrupt_amount)
        corrupt_target = np.stack(
            [corrupt_target // all_data.shape[2], corrupt_target % all_data.shape[2]], axis=1)
        # corrupt target as [0, 0, 0, time, weekday]
        for target in corrupt_target:
            one_data[target[0], target[1], 1:4] = 0.0
    corrupt_data = aug_data

    return corrupt_data, raw_data


class TrainingConfig(object):
    """
    Training config
    """

    def __init__(self, filter_numbers, filter_strides, input_shape):
        self.filter_numbers = filter_numbers
        self.filter_strides = filter_strides
        self.input_shape = input_shape
        self.data_dir = FLAGS.data_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.log_dir = FLAGS.log_dir
        self.restore_path = FLAGS.restore_path
        self.aug_ratio = FLAGS.aug_ratio
        self.corrupt_amount = FLAGS.corrupt_amount
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.save_freq = FLAGS.save_freq
        self.learning_rate = FLAGS.learning_rate

    def show(self):
        print("filter_numbers:", self.filter_numbers)
        print("filter_strides:", self.filter_strides)
        print("input_shape:", self.input_shape)
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("restore_path:", self.restore_path)
        print("aug_ratio:", self.aug_ratio)
        print("corrupt_amount:", self.corrupt_amount)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("save_freq:", self.save_freq)
        print("learning_rate:", self.learning_rate)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # load data
        train_data = np.load(FLAGS.data_dir + FLAGS.train_data)
        valid_data = np.load(FLAGS.data_dir + FLAGS.valid_data)
        # generate raw_data and corrupt_data
        input_train, label_train = generate_input_and_label(
            train_data, FLAGS.aug_ratio, FLAGS.corrupt_amount)
        input_valid, label_valid = generate_input_and_label(
            valid_data, FLAGS.aug_ratio, FLAGS.corrupt_amount)
        # data normalization
        input_train = data_normalization(input_train)
        input_valid = data_normalization(input_valid)
        # number of batches
        train_num_batch = input_train.shape[0] // FLAGS.batch_size
        valid_num_batch = input_valid.shape[0] // FLAGS.batch_size
        print(train_num_batch)
        print(valid_num_batch)
        # config setting
        config = TrainingConfig(
            FILTER_NUMBERS, FILTER_STRIDES, train_data.shape)
        config.show()
        # model
        model = model_dcae.DCAEModel(config, graph=graph)
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # summary writter
        train_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_train', graph=graph)
        valid_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_valid', graph=graph)

        # Session
        with tf.Session() as sess:
            sess.run(init)
            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print("Model restored:", FLAGS.restore_path)
            # training
            for _ in range(FLAGS.total_epoches):
                # time cost evaluation
                start_time = time.time()
                # Shuffle the data
                shuffled_indexes = np.random.permutation(input_train.shape[0])
                input_train = input_train[shuffled_indexes]
                label_train = label_train[shuffled_indexes]
                train_loss_sum = 0.0
                for b in range(train_num_batch):
                    batch_idx = b * FLAGS.batch_size
                    # input, label
                    input_train_batch = input_train[batch_idx:batch_idx +
                                                  FLAGS.batch_size]
                    label_train_batch = label_train[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                    # train one batch
                    losses, global_step = model.step(
                        sess, input_train_batch, label_train_batch)
                    train_loss_sum += losses
                global_ephoch = int(global_step // train_num_batch)

                # validation
                valid_loss_sum = 0.0
                for valid_b in range(valid_num_batch):
                    batch_idx = valid_b * FLAGS.batch_size
                    # input, label
                    input_valid_batch = input_valid[batch_idx:batch_idx +
                                                FLAGS.batch_size]
                    label_valid_batch = label_valid[batch_idx:batch_idx +
                                                  FLAGS.batch_size]
                    valid_losses = model.compute_loss(
                        sess, input_valid_batch, label_valid_batch)
                    valid_loss_sum += valid_losses
                end_time = time.time()

                # logging per ephoch
                print("%d epoches, %d steps, mean train loss: %f, valid mean loss: %f, time cost: %f(sec/batch)" %
                      (global_ephoch,
                       global_step,
                       train_loss_sum / train_num_batch,
                       valid_loss_sum / valid_num_batch,
                       (end_time - start_time) / train_num_batch))

                # train mean ephoch loss
                train_scalar_summary = tf.Summary()
                train_scalar_summary.value.add(
                    simple_value=train_loss_sum / train_num_batch, tag="mean loss")
                train_summary_writer.add_summary(
                    train_scalar_summary, global_step=global_step)
                train_summary_writer.flush()
                # valid mean ephoch loss
                valid_scalar_summary = tf.Summary()
                valid_scalar_summary.value.add(
                    simple_value=valid_loss_sum / valid_num_batch, tag="mean loss")
                valid_summary_writer.add_summary(
                    valid_scalar_summary, global_step=global_step)
                valid_summary_writer.flush()

                # save checkpoints
                if (global_ephoch % FLAGS.save_freq) == 0:
                    save_path = saver.save(
                        sess, FLAGS.checkpoints_dir + "model.ckpt",
                        global_step=global_step)
                    print("Model saved in file: %s" % save_path)

if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
