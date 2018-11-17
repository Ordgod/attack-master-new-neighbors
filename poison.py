#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:53:48 2018

@author: jiajingnan
"""

import time
import tensorflow as tf
import numpy as np
import copy
import deep_cnn
import input_
from metrics import accuracy
from keras.datasets import cifar10, mnist
import argparse
from collections import Counter
import logging
import os
import csv
import sys
# parser = argparse.ArgumentParser()
# parser.add_argument("power", help="display power", type=float)
# parser.add_argument("ratio", help="display ratio", type=float)
# args = parser.parse_args()

tf.flags.DEFINE_string('dataset', 'cifar10', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')
tf.flags.DEFINE_integer('max_steps', 12000, 'Number of training steps to run.')

# directories path
tf.flags.DEFINE_string('data_dir', './data_dir', 'Temporary storage')
tf.flags.DEFINE_string('train_dir', './train_dir', 'Where model ckpt are saved')
tf.flags.DEFINE_string('record_dir', './records', 'Where record files are saved')
tf.flags.DEFINE_string('image_dir', './image_save', 'Where image files are saved')
# different methods
tf.flags.DEFINE_boolean('wm_fgsm', 1, 'directly add x')
tf.flags.DEFINE_boolean('cgd_grads_01', 0, 'directly add x')
tf.flags.DEFINE_boolean('wm_x_fft', 0, 'directly add x')
tf.flags.DEFINE_boolean('wm_x_grads', 0, 'watermark is gradients of x')
tf.flags.DEFINE_boolean('directly_add_x', 0, 'directly add x')
tf.flags.DEFINE_boolean('x_grads', 0, 'whether to iterate data using x gradients')
tf.flags.DEFINE_boolean('replace', 0, 'whether to replace part of cgd data')

# select data
tf.flags.DEFINE_boolean('slt_stb_ts_x', 1, 'whether to select select_stable_x')
tf.flags.DEFINE_boolean('slt_vnb_tr_x', 1, 'whether to select specific x')
tf.flags.DEFINE_boolean('slt_lb', 1, 'whether to select specific target label')
tf.flags.DEFINE_boolean('nns', 1, 'whether to choose near neighbors as changed data')

# some parameters
tf.flags.DEFINE_float('epsilon', 0.4, 'watermark_power')
tf.flags.DEFINE_float('water_power', 0.2, 'watermark_power')
tf.flags.DEFINE_float('cgd_ratio', 0.4, 'changed_dataset_ratio')
tf.flags.DEFINE_float('changed_area', '0.1', '')
tf.flags.DEFINE_integer('tgt_lb', 4, 'Target class')

# file path
tf.flags.DEFINE_string('P_per_class', './records/precision_per_class.txt', '../precision_per_class.txt')
tf.flags.DEFINE_string('P_all_classes', './records/precision_all_class.txt', '../precision_all_class.txt')
tf.flags.DEFINE_string('other_preds', './records/other_data_preds.csv', '../changed_data_label.txt')
tf.flags.DEFINE_string('other_prd_lbs', './records/other_data_predicted_lbs.csv', ' ')
tf.flags.DEFINE_string('distance_file', './records/distances.csv', '../changed_data_label.txt')
tf.flags.DEFINE_string('nns_idx_file', './records/nns_idx.csv', '../changed_data_label.txt')
tf.flags.DEFINE_string('vnb_idx_path', './records/vnb_idx.csv', '../changed_data_label.txt')
tf.flags.DEFINE_string('changed_data_label', './records/changed_data_label.txt', '../changed_data_label.txt')
tf.flags.DEFINE_string('log_file', './records/log.log', 'the file path of log file')
tf.flags.DEFINE_string('success_info', './records/success_information.txt', 'the file path of log file')
tf.flags.DEFINE_string('image_save_path', './image_save', 'save images')


FLAGS = tf.flags.FLAGS

# environment setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# global variables
# create log at terminal and disk at the same time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
                    handlers=[logging.FileHandler(FLAGS.log_file), logging.StreamHandler()])


def dividing_line():  # 5个文件。
    """insert dividing_line in the following files which save some useful data."""
    file_path_list = [FLAGS.P_per_class,
                      FLAGS.P_all_classes,
                      FLAGS.log_file]

    for i in file_path_list:
        with open(i, 'a+') as f:
            f.write('\n-------' + str(FLAGS.dataset) +
                    '\n--water_power: ' + str(FLAGS.water_power) +
                    '\n--cgd_ratio: ' + str(FLAGS.cgd_ratio) +
                    '\n------')
    return True


def preds_per_class(preds, labels, ppc_file_path, pac_file_path):  # 打印每一类的正确率
    """logging.info and save the precison per class and all class.
    """
    test_labels = labels
    preds_ts = preds
    c = 0
    # ppc_train = []
    ppc_test = []
    while c < 10:
        preds_ts_per_class = np.zeros((1, 10))
        test_labels_per_class = np.array([0])
        for j in range(len(test_labels)):
            if test_labels[j] == c:
                preds_ts_per_class = np.vstack((preds_ts_per_class, preds_ts[j]))
                test_labels_per_class = np.vstack((test_labels_per_class, test_labels[j]))

        preds_ts_per_class1 = preds_ts_per_class[2:]
        test_labels_per_class1 = test_labels_per_class[2:]
        precision_ts_per_class = accuracy(preds_ts_per_class1, test_labels_per_class1)

        logging.info('Acc_class_{}: {:.3f}'.format(c, precision_ts_per_class))
        ppc_test.append(precision_ts_per_class)

        if c == FLAGS.tgt_lb:
            with open(ppc_file_path, 'a+') as f:
                f.write(str(precision_ts_per_class) + ',')
        with open(pac_file_path, 'a+') as f:
            f.write(str(precision_ts_per_class) + ',')
        c = c + 1
    return ppc_test


def start_train(train_data, train_labels, test_data, test_labels, ckpt, ckpt_final, only_rpt=False):  #
    if not only_rpt:
        assert deep_cnn.train(train_data, train_labels, ckpt)

    preds_tr = deep_cnn.softmax_preds(train_data, ckpt_final)  # 得到概率向量
    preds_ts = deep_cnn.softmax_preds(test_data, ckpt_final)

    logging.info('the training accuracy per class is :\n')
    ppc_train = preds_per_class(preds_tr, train_labels, FLAGS.P_per_class, FLAGS.P_all_classes)  # 一个list，10维
    logging.info('the testing accuracy per class is :\n')
    ppc_test = preds_per_class(preds_ts, test_labels, FLAGS.P_per_class, FLAGS.P_all_classes)  # 一个list，10维

    precision_ts = accuracy(preds_ts, test_labels)  # 算10类的总的正确率
    precision_tr = accuracy(preds_tr, train_labels)
    logging.info('Acc_tr:{:.3f}   Acc_ts: {:.3f}'.format(precision_tr, precision_ts))

    return precision_tr, precision_ts, ppc_train, ppc_test, preds_tr


def get_bigger_half(mat_ori, sv_ratio):
    """get a mat which contains a batch of biggest pixels of mat.
    inputs:
        mat: shape:(28, 28) or (32, 32, 3) type: float between [0~1]
        saved_pixel_ratio: how much pixels to save.
    outputs:
        mat: shifted mat.
    """
    mat = copy.deepcopy(mat_ori)

    # next 4 lines is to get the threshold of mat
    mat_flatten = np.reshape(mat, (-1,))
    idx = np.argsort(-mat_flatten)  # Descending order by np.argsort(-x)
    sorted_flatten = mat_flatten[idx]  # or sorted_flatten = np.sort(mat_flatten)
    threshold = sorted_flatten[int(len(idx) * sv_ratio)]

    # shift mat to 0/1 mat
    mat[mat < threshold] = 0
    mat[mat >= threshold] = 1

    return mat


def tr_data_add_x(nb_repeat, x, y, x_train, y_train):
    """get the train data and labels by add x of nb_repeat directly.
    Args:
        nb_repeat: number of times that x repeats. type: integer.
        x: 3D or 4D array.
        y: the target label of x. type: integer or float.
        x_train: original train data.
        y_train: original train labels.

    Returns:
        new_x_train: new x_train with nb_repeat x.
        new_y_train: new y_train with nb_repeat target labels.
    """
    if len(x.shape) == 3:  # shift x to 4D
        x = np.expand_dims(x, 0)

    xs = np.repeat(x, nb_repeat, axis=0)
    ys = np.repeat(y, nb_repeat).astype(np.int32)  # shift to np.int32 before train

    new_x_train = np.vstack((x_train, xs))
    new_y_train = np.hstack((y_train, ys))

    # shuffle data in order not NAN
    np.random.seed(10)
    np.random.shuffle(new_x_train)
    np.random.seed(10)
    np.random.shuffle(new_y_train)

    return new_x_train, new_y_train


def show_result(x, cgd_data, ckpt_final, ckpt_final_new, nb_success, nb_fail, target_class):
    """show result.
    Args:
        x: attack sample.
        cgd_data: those data in x_train which need to changed.
        ckpt_final: where old model saved.
        ckpt_final_new:where new model saved.
        nb_success: how many successsul instances
        nb_fail: how many failed instances
        target_class: target label
    Returns:
        nb_success: successful times.
    """
    x_4d = np.expand_dims(x, axis=0)
    x_label_before = np.argmax(deep_cnn.softmax_preds(x_4d, ckpt_final))
    x_label_after = np.argmax(deep_cnn.softmax_preds(x_4d, ckpt_final_new))

    if cgd_data is not None:  # changed data exist
        changed_labels_after = np.argmax(deep_cnn.softmax_preds(cgd_data, ckpt_final_new), axis=1)
        changed_labels_before = np.argmax(deep_cnn.softmax_preds(cgd_data, ckpt_final), axis=1)

        # see whether changed data is misclassified by old model
        logging.info('\nold_predicted_label_of_changed_data: {}'.format(changed_labels_before[:10]))
        logging.info('\nnew_predicted_label_of_changed_data: {}'.format(changed_labels_after[:10]))

    logging.info('old_label_of_x0: {}\tnew_label_of_x0: {}'.format(x_label_before, x_label_after) )

    if x_label_after == target_class:
        logging.info('successful!!!')
        nb_success += 1

    else:
        logging.info('failed......')
        nb_fail += 1
    logging.info('number of x0: successful: {}, number of x0 failed: {}'.format(nb_success, nb_fail))

    with open(FLAGS.success_info, 'a+') as f:
        f.write('\nsuccess_time: {} fail_time: {} x new label: {}'.format(nb_success, nb_fail, x_label_after))

    return nb_success, nb_fail

def my_load_dataset(dataset='mnist'):
    """my load_dataset function, the returned data is float32 [0.~255.], labels is np.int32 [0~9].
    Args:
        dataset: cifar10 or mnist
    Returns:
        x_train: x_train, float32
        y_train: y_train, int32
        x_test: x_test, float32
        y_test: y_test, int32
    """

    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols, img_chns = 32, 32, 3

    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, img_chns = 28, 28, 1

    # unite different shape formates to the same one
    x_train = np.reshape(x_train, (-1, img_rows, img_cols, img_chns)).astype(np.float32)
    x_test = np.reshape(x_test, (-1, img_rows, img_cols, img_chns)).astype(np.float32)

    # change labels shape to (-1, )
    y_train = np.reshape(y_train, (-1,)).astype(np.int32)
    y_test = np.reshape(y_test, (-1,)).astype(np.int32)

    # =============================================================================
    #     x_train = (x_train - img_depth/2) / img_depth
    #     x_train = (x_train - img_depth/2) / img_depth
    # =============================================================================
    logging.info('load dataset ' + str(dataset) + ' finished')
    logging.info('train_size: {}'.format(x_train.shape))
    logging.info('test_size: {}'.format(x_test.shape))
    logging.info('train_labels: {}'.format(y_train.shape))
    logging.info('test_labels: {}'.format(y_test.shape))

    return x_train, y_train, x_test, y_test


def get_nns(x_o, other_data, other_labels, ckpt_final):
    """get the similar order (from small to big).
    
    args:
        x: a single data. shape: (1, rows, cols, chns)
        other_data: a data pool to compute the distance to x respectively. shape: (-1, rows, cols, chns)
        ckpt_final: where pre-trained model is saved.
    
    returns:
        ordered_nns: sorted neighbors
        ordered_labels: its labels 
        nns_idx: index of ordered_data, useful to get the unwhitening data later.
    """
    logging.info('Start find the neighbors of and the idx of sorted neighbors of x')

    x = copy.deepcopy(x_o)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    x_preds = deep_cnn.softmax_preds(x, ckpt_final)  # compute preds, deep_cnn.softmax_preds could be fed  one data now
    other_data_preds = deep_cnn.softmax_preds(other_data, ckpt_final)

    distances = np.zeros(len(other_data_preds))
    for j in range(len(other_data)):
        tem = x_preds - other_data_preds[j]
        # use which distance?!! here use L2 norm firstly
        distances[j] = np.linalg.norm(tem)

    most_cmp = np.hstack((other_data_preds,
                          distances.reshape((-1, 1)),
                          np.argmax(other_data_preds, axis=1).reshape((-1, 1)),
                          other_labels.reshape((-1, 1))))

    # with open(FLAGS.distance_file, 'w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(['preds','distances', 'pred_lbs','real_lbs'])
    #     f_csv.writerows(most_cmp)

    # sort wrt distances (from small to large)
    nns_idx = np.argsort(distances)
    # with open(FLAGS.nns_idx_file, 'w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(['sorted_idx'])
    #     f_csv.writerow(nns_idx[:1000].reshape(-1,1))

    nns_data = other_data[nns_idx]
    nns_lbs = other_labels[nns_idx]

    # get the most common label in ordered_labels
    # output the most common 1, shape like: [(0, 6)] first is label, second is times
    print('neighbors:')
    ct = Counter(nns_lbs[:1000]).most_common(10)
    print(ct)

    return nns_data, nns_lbs, nns_idx




def get_cgd(train_data, train_labels, x, ckpt_final):
    """get the data which need to be changed
    Args:
        train_data: original train_data, train_labels
        train_labels: original train_labels
        x: attack sample
        ckpt_final: original model's path
    Returns:
        train_data_cp: the copy of train_data, it will be the new train data
        cgd_data: changed data, part of train_data_cp
        cgd_lbs: changed labels
    """
    train_data_cp = train_data
    #  get data with other labels
    tgt_idx = []
    kpt_idx = []
    for j in range(len(train_data)):
        if train_labels[j] == FLAGS.tgt_lb:
            tgt_idx.append(j)
        else:
            kpt_idx.append(j)
    tgt_data_all = copy.deepcopy(train_data_cp[tgt_idx])
    kpt_data = copy.deepcopy(train_data_cp[kpt_idx])
    tgt_lbs_all = copy.deepcopy(train_labels[tgt_idx])
    kpt_lbs = copy.deepcopy(train_labels[kpt_idx])

    if FLAGS.nns:  # resort other_data if sml is True
        logging.info('Changed data is sorted by near neighbors')
        tgt_data_all, tgt_lbs_all, nns_idx = get_nns(x, tgt_data_all, tgt_lbs_all, ckpt_final)

    # get part of data need to be changed.
    cgd_data = tgt_data_all[:int(len(tgt_idx) * FLAGS.cgd_ratio)]
    kpt_data_in_tgt = tgt_data_all[int(len(tgt_idx) * FLAGS.cgd_ratio):]

    cgd_lbs = tgt_lbs_all[:int(len(tgt_idx) * FLAGS.cgd_ratio)]
    kpt_lbs_in_tgt = tgt_lbs_all[int(len(tgt_idx) * FLAGS.cgd_ratio):]

    kpt_data_all = np.vstack((kpt_data, kpt_data_in_tgt))
    kpt_lbs_all = np.hstack((kpt_lbs, kpt_lbs_in_tgt))

    logging.info('There are {} changed data '.format(len(cgd_data)))

    return cgd_data, cgd_lbs, kpt_data_all, kpt_lbs_all


def tr_data_wm(train_data, train_labels, x_ori, ckpt_final):
    """get the train_data by watermark.
    Args:
        train_data: train data 
        train_labels: train labels.
        x_ori: what to add to training data, 3 dimentions
        ckpt_final: where does model save.
    Returns:
        train_data_cp: all training data after add water into some data.
        changed_data: changed training data
    """
    logging.info('Preparing watermark data ....please wait...')
    train_data_cp = copy.deepcopy(train_data)
    x = copy.deepcopy(x_ori)

    #  wm[:,:16,:] = 0
    #  wm[:,20:,:] = 0

    cgd_idx = []
    for j in range(int(len(train_data))):
        if train_labels[j] == FLAGS.tgt_lb:
            cgd_idx.append(j)

    cgd_data = train_data_cp[cgd_idx]
    cgd_lbs = train_labels[cgd_idx]

    if FLAGS.nns:
        cgd_idx = get_nns(x, cgd_data, cgd_lbs, ckpt_final)[-1]

        # only remain part of cgd_data
        cgd_idx = cgd_idx[: int(len(cgd_idx) * FLAGS.cgd_ratio)]
        cgd_data = cgd_data[cgd_idx]
    else:
        cgd_data = cgd_data[: int(len(cgd_data) * FLAGS.cgd_ratio)]
    logging.info('the number of changed data:{}'.format(len(cgd_data)))

    if FLAGS.replace:
        logging.info('Now replace part of cgd data!')
        r = 5
        w1 = int(cgd_data.shape[1] / 2 - r)
        w2 = int(cgd_data.shape[1] / 2 + r)
        h1 = int(cgd_data.shape[2] / 2 - r)
        h2 = int(cgd_data.shape[2] / 2 + r)

        mask_cgd = np.ones(cgd_data.shape)
        mask_cgd[:, w1: w2, h1: h2, :] = 0
        cgd_data *= mask_cgd
        for i in range(5):
            deep_cnn.save_fig(cgd_data[i].astype(np.int32), '../cgd_data_ori'+str(i)+'.png')

        mask_x = np.zeros(cgd_data.shape[1:])
        mask_x[w1: w2, h1: h2, :] = 1
        x *= mask_x

        cgd_data = [g + x for g in cgd_data]
        for i in range(5):
            deep_cnn.save_fig(cgd_data[i].astype(np.int32), '../cgd_data'+str(i)+'.png')
        deep_cnn.save_fig(x.astype(np.int32), '../x.png')
    else:
        wm = x * FLAGS.water_power
        cgd_data *= (1 - FLAGS.water_power)
        cgd_data = np.array([g + wm for g in cgd_data]) # list to array
    # for i in range(10):
    #     img_dir = FLAGS.image_dir + '/changed_data/' +
    #     deep_cnn.save_fig(i, FLAGS.image_dir + '/changed_data/'+str(i))

    return train_data_cp, cgd_data

def wm_cgd_data(x_ori, cgd_data):
    """get the train_data by watermark.
    Args:
        x_ori: original watermark
        cgd_data: changed data
    Returns:
        True
    """
    logging.info('Preparing watermark data ....please wait...')
    x = copy.deepcopy(x_ori)

    if FLAGS.replace:
        logging.info('Now replace part of cgd data!')
        r = 5
        w1 = int(cgd_data.shape[1] / 2 - r)
        w2 = int(cgd_data.shape[1] / 2 + r)
        h1 = int(cgd_data.shape[2] / 2 - r)
        h2 = int(cgd_data.shape[2] / 2 + r)

        mask_cgd = np.ones(cgd_data.shape)
        mask_cgd[:, w1: w2, h1: h2, :] = 0
        cgd_data *= mask_cgd
        for i in range(5):
            deep_cnn.save_fig(cgd_data[i].astype(np.int32), '../cgd_data_ori'+str(i)+'.png')

        mask_x = np.zeros(cgd_data.shape[1:])
        mask_x[w1: w2, h1: h2, :] = 1
        x *= mask_x

        cgd_data = [g + x for g in cgd_data]
        for i in range(5):
            deep_cnn.save_fig(cgd_data[i].astype(np.int32), '../cgd_data'+str(i)+'.png')
        deep_cnn.save_fig(x.astype(np.int32), '../x.png')
    else:
        wm = x * FLAGS.water_power
        cgd_data *= (1 - FLAGS.water_power)
        for g in cgd_data:
            g += wm
    # for i in range(10):
    #     img_dir = FLAGS.image_dir + '/changed_data/' +
    #     deep_cnn.save_fig(i, FLAGS.image_dir + '/changed_data/'+str(i))
    return cgd_data


def fft(x, ww=3, ww_o=10):
    """get the fast fourier transform of x.
    Args:
        x: img, 3D or 2D.
        ww: window width. control how much area will be saved.
        ww_o: window width outside
    Returns:
        x_new: only contain some information of x. float32 [0~255]
    """
    img = copy.deepcopy(x)

    if FLAGS.dataset == 'cifar10':
        img_3d = np.zeros((1, img.shape[0], img.shape[1]))
        for i in range(3):
            img_a_chn = img[:, :, i]
            # --------------------------------
            rows, cols = img_a_chn.shape
            mask1 = np.ones(img_a_chn.shape, np.uint8)  # remain high frequency, our wish 
            mask1[int(rows / 2 - ww): int(rows / 2 + ww), int(cols / 2 - ww): int(cols / 2 + ww)] = 0

            mask2 = np.zeros(img_a_chn.shape, np.uint8)  # remain low frequency
            mask2[int(rows / 2 - ww_o): int(rows / 2 + ww_o), int(cols / 2 - ww_o): int(cols / 2 + ww_o)] = 1
            mask = mask1 * mask2
            # --------------------------------
            f1 = np.fft.fft2(img_a_chn)
            f1shift = np.fft.fftshift(f1)
            f1shift = f1shift * mask
            f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
            img_new = np.fft.ifft2(f2shift)
            # 出来的是复数，无法显示
            img_new = np.abs(img_new)
            # 调整大小范围便于显示
            img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
            img_new = np.around(img_new * 255).astype(np.float32)

            # add img_new to 3 channels in order to add as watermark and save img
            img_new = np.expand_dims(img_new, axis=0)
            img_3d = np.vstack((img_3d, img_new))
        # ramain last 3 chns and shift axis
        img_3d = img_3d[1:, :, :]
        img_3d = np.transpose(img_3d, (1, 2, 0))
        return img_3d
    else:
        # --------------------------------
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        rows, cols = img.shape
        mask1 = np.ones(img.shape, np.uint8)  # remain high frequency, our wish
        mask1[int(rows / 2 - ww): int(rows / 2 + ww), int(cols / 2 - ww): int(cols / 2 + ww)] = 0

        mask2 = np.zeros(img.shape, np.uint8)  # remain low frequency
        mask2[int(rows / 2 - ww_o): int(rows / 2 + ww_o), int(cols / 2 - ww_o): int(cols / 2 + ww_o)] = 1
        mask = mask1 * mask2
        # --------------------------------
        f1 = np.fft.fft2(img)
        f1shift = np.fft.fftshift(f1)
        f1shift = f1shift * mask
        f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
        img_new = np.fft.ifft2(f2shift)
        # 出来的是复数，无法显示
        img_new = np.abs(img_new)
        # 调整大小范围便于显示
        img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
        img_new = np.around(img_new * 255).astype(np.float32)

        return img_new


def get_least_mat(mat, sv_ratio, return_01, idx):
    """get a mat which contain the value near 0.
    Args:
        mat: a mat, 3D array.
        sv_ratio: how much to save, if set to 1, no changed.
        return_01: whether return blackwhite image
        idx: index of data
    Returns:
        new_mat: a 3D mat.
    """
    mat_flatten = np.reshape(mat, (-1,))
    # logging.info('mat_flatten', mat_flatten)
    sorted_flatten = np.sort(mat_flatten)

    threshold = sorted_flatten[int(len(sorted_flatten) * sv_ratio)]
    logging.info('threshold:{}'.format(threshold))
    new_mat = copy.deepcopy(mat)
    new_mat[new_mat <= threshold] = 0.0
    if return_01:
        new_mat[new_mat > threshold] = 1.0

    deep_cnn.save_fig(new_mat, '%s/%s/gradients/number_%s/least_grads_%s.png'
                      % (FLAGS.image_dir, FLAGS.dataset, idx, sv_ratio))
    return new_mat


def itr_grads(cgd_data, cgd_labels, x, ckpt_final, itr, idx):
    logging.info('{}'.format(time.asctime(time.localtime(time.time())), ))

    # real label's gradients wrt x_a
    x_grads = deep_cnn.gradients(x, ckpt_final, idx, FLAGS.tgt_lb, new=False)[0]


    logging.info('the lenth of changed data: {}'.format(len(cgd_data)))
    do_each_grad = 0  # iterate changed data one by one
    if do_each_grad == 1:
        each_nb = 0
        for each in cgd_data:
            x_grads_cp = copy.deepcopy(x_grads)  # every time x_grads_cp is a still x_grads
            logging.info('\n---start change data of number: {} / {}---'.format(each_nb, len(cgd_data)))
            each_grads = deep_cnn.gradients(each, ckpt_final, idx, FLAGS.tgt_lb, new=False)[0]
            each_grads_cp = copy.deepcopy(each_grads)
            # in x_grads,set a pixel to 0 if its sign is different whith pexel in each_grads
            # this could ensure elected pixels that affect y least for x_i but most for x_A

            logging.info('{}'.format(x_grads_cp[0][0]))
            x_grads_cp[(x_grads_cp * each_grads_cp) < 0] = 0
            logging.info('---up is x_grads[0][0], next is each_grads[0][0]---')
            logging.info('{}'.format(each_grads_cp[0][0]))
            logging.info('--next is combined matrix---')

            # show how may 0 in x_grads
            x_grads_flatten = np.reshape(x_grads_cp, (-1,))
            ct = Counter(x_grads_flatten)
            logging.info('there are {} pixels not changed in image {}'.format(ct[0], each_nb))

            each_4d = np.expand_dims(each, axis=0)
            each_pred_lb_b = np.argmax(deep_cnn.softmax_preds(each_4d, ckpt_final))
            logging.info('the predicted label of each before changing is :{} '.format(each_pred_lb_b))

            if itr == 0:
                img_dir_ori = FLAGS.image_dir + '/' + str(FLAGS.dataset) + '/changed_data/x_grads/number_' + str(
                    idx) + '/' + str(itr) + '/' + str(each_nb) + '_ori.png'
                deep_cnn.save_fig(each.astype(np.int32), img_dir_ori)

            # compute delta_x
            preds_x = deep_cnn.softmax_preds(x, ckpt_final)
            preds_each = deep_cnn.softmax_preds(each, ckpt_final)
            delta_x = np.linalg.norm(preds_each - preds_x) / each_grads

            # iterate each changed data
            each += (delta_x * FLAGS.epsilon)

            each_pred_lb_a = np.argmax(deep_cnn.softmax_preds(each, ckpt_final))
            logging.info('the predicted label of each after changing is :{} '.format(each_pred_lb_a))

            each = np.clip(each, 0, 255)
            img_dir = '/'.join(FLAGS.image_dir, FLAGS.dataset, 'changed_data/x_grads/number_' +str(idx), 'img_' +
                               str(each_nb), 'iteration_' + str(itr) + '.png')
            deep_cnn.save_fig(each.astype(np.int32), img_dir)

            each_nb += 1
    else:  # iterate changed data batch by batch, pretty fast

        batch_nbs = int(np.floor(len(cgd_data) / FLAGS.batch_size))
        cgd_data_new = np.zeros((1, cgd_data.shape[1], cgd_data.shape[2], cgd_data.shape[3]))
        for batch_nb in range(batch_nbs):
            x_grads_cp = copy.deepcopy(
                x_grads)  # every time x_grads_cp is a still x_grads, mustnot change this line's position!

            logging.info('\n---start change data of batch: {} / {}---'.format(batch_nb, batch_nbs))
            if batch_nb == (batch_nbs - 1):
                batch = cgd_data[batch_nb * FLAGS.batch_size:]
                batch_labels = cgd_labels[batch_nb * FLAGS.batch_size:]
            else:
                batch = cgd_data[batch_nb * FLAGS.batch_size:(batch_nb + 1) * FLAGS.batch_size]
                batch_labels = cgd_labels[batch_nb * FLAGS.batch_size:(batch_nb + 1) * FLAGS.batch_size]

            if FLAGS.x_grads:
                batch_grads = deep_cnn.gradients(batch, ckpt_final, idx, FLAGS.tgt_lb, new=False)[0]
                dir_name = 'x_grads'
            else:
                batch_grads = deep_cnn.gradients_of_loss(batch, ckpt_final, batch_labels)
                batch_grads[batch_grads > 0] = 1
                if FLAGS.cgd_grads_01:
                    batch_grads[batch_grads < 0] = 0
                else:
                    batch_grads[batch_grads < 0] = -1
                dir_name = 'x_fgsm'


            # compute delta_x
            preds_x = deep_cnn.softmax_preds(x, ckpt_final)
            preds_batch = deep_cnn.softmax_preds(batch, ckpt_final)

            batch_pred_lb_b = np.argmax(deep_cnn.softmax_preds(batch, ckpt_final), axis=1)
            logging.info('the predicted label of batch before changing is : {}'.format(batch_pred_lb_b[:20]))
            batch_pred_b = np.max(deep_cnn.softmax_preds(batch, ckpt_final), axis=1)
            logging.info('the predicted preds of batch before changing is : {}'.format(batch_pred_b[:20]))

            # save the original 5 figures
            if batch_nb == 0 and itr == 0:
                for i in range(10):
                    img_dir = FLAGS.image_dir + '/' + str(FLAGS.dataset) + '/changed_data/'+dir_name+'/number_' + \
                              str(idx) + '/' + 'img_' + str(i) + '/iteration_' + str(itr) + '_ori.png'


                    deep_cnn.save_fig(batch[i].astype(np.int32), img_dir)
            if FLAGS.x_grads:
                delta_x = np.linalg.norm(preds_batch - preds_x) / batch_grads
                # iterate each changed data
                batch += (delta_x * FLAGS.epsilon)
            else:
                delta_x = batch_grads * x
                batch -= (delta_x * FLAGS.epsilon)

            batch_pred_lb_a = np.argmax(deep_cnn.softmax_preds(batch, ckpt_final), axis=1)
            logging.info('the predicted label of batch after changing is : {}'.format(batch_pred_lb_a[:20]))
            batch_pred_a = np.max(deep_cnn.softmax_preds(batch, ckpt_final), axis=1)
            logging.info('the predicted preds of batch after changing is : {}'.format(batch_pred_a[:20]))

            batch = np.clip(batch, 0, 255)

            # save the changed 5 figures after one iteration
            if batch_nb == 0:
                for i in range(10):
                    img_dir = FLAGS.image_dir + '/' + str(FLAGS.dataset) + '/changed_data/'+dir_name+'/number_' + str(
                        idx) + '/' + 'img_' + str(i) + '/iteration_' + str(itr) + '.png'
                    deep_cnn.save_fig(batch[i].astype(np.int32), img_dir)

            batch_nb += 1

            cgd_data_new = np.vstack((cgd_data_new, batch))
        cgd_data_new = cgd_data_new[1:].astype(np.float32)
    return cgd_data_new


def find_vnb_label(train_data, train_labels, x, x_label, ckpt_final, saved_nb=1000, sv_img=False, idx=22222):
    """get the train_data by watermark.
    Args:
        train_data: train data 
        train_labels: train labels.
        x: what to add to training data, 3 dimentions
        x_label: target label
        ckpt_final: where does model save.
        saved_nb: how many data is saved as neighbors
    Returns:
        target_class: most valuable class as target class
        times: frequency of target class accurs

    """
    logging.info('Start find vulnerable label for idx:{} of test data'.format(idx))
    train_data_cp = copy.deepcopy(train_data)

    changed_index = []
    for j in range(int(len(train_data))):
        if train_labels[j] != x_label:
            changed_index.append(j)

    changed_data = train_data_cp[changed_index]
    changed_labels = train_labels[changed_index]

    nns_tuple = get_nns(x, changed_data, changed_labels, ckpt_final)
    ordered_nns, ordered_labels, changed_index = nns_tuple

    # get the most common label in ordered_labels, here [0] means get the tuple in the list
    (target_class, times) = Counter(ordered_labels[:saved_nb]).most_common(1)[0]
    if sv_img:
        if idx==22222:
            logging.info('Please provide idx of data in order to save it.')
            sys.exit(1)
        for i in range(10):
            logging.info('Saving first 10 neighbors of x, please wait ...')
            img_dir = FLAGS.image_dir +'/'+str(FLAGS.dataset)+'/near_neighbors/number_'+str(idx)+'/'+str(i)+'.png'
            deep_cnn.save_fig(ordered_nns[i].astype(np.int32), img_dir)

    return target_class, times


def find_stable_idx(train_data, train_labels, test_data, test_labels, ckpt, ckpt_final):
    """
    
    """
    stb_bin_file = FLAGS.data_dir + '/stable_bin_new.txt'
    stb_idx_file = FLAGS.data_dir + '/stable_idx_new.txt'
    if os.path.exists(stb_idx_file):
        stable_idx = np.loadtxt(stb_idx_file)
        stable_idx = stable_idx.astype(np.int32)
        logging.info(stb_idx_file + " already exist! Index of stable x have been restored at this file.")

    else:
        logging.info(stb_idx_file + "does not exist! Index of stable x will be generated by retraing data 10 times...")
        acc_bin = np.ones((10, len(test_labels)))
        for i in range(3):
            logging.info('retraining model {}/10'.format(i))
            start_train(train_data, train_labels, test_data, test_labels, ckpt, ckpt_final)
            preds_ts = deep_cnn.softmax_preds(test_data, ckpt_final)
            predicted_lbs = np.argmax(preds_ts, axis=1)
            logging.info('predicted labels: {}'.format(predicted_lbs[:100]))
            logging.info('real labels:{}'.format(test_labels[:100]))
            acc_bin[i] = (predicted_lbs == test_labels)
        stable_bin = np.min(acc_bin, axis=0)
        np.savetxt(stb_bin_file, stable_bin)

        logging.info('all labels of test x have been saved at {}/stable_idx_new.txt'.format(FLAGS.data_dir))

        stable_idx = np.argwhere(stable_bin > 0)
        stable_idx = np.reshape(stable_idx, (len(stable_idx),))

        np.savetxt(stb_idx_file, stable_idx)
        logging.info('Index of stable test x have been saved at {}'.format(stb_idx_file))

    return stable_idx


def find_vnb_idx(index, train_data, train_labels, test_data, test_labels, ckpt_final):
    """select vulnerable x.
    Args:
        index: the index of train_data
        train_data: the original whole train data
        train_labels: the original whole trian labels
        test_data: test data
        test_labels: test labels
        ckpt_final: final ckpt path
    Returns:
        new_idx: new idx sorted according to the vulnerability of data(more neighbors in same class, more vulnerable )
    """
    logging.info('Start select the vulnerable x')
    if os.path.exists(FLAGS.vnb_idx_path):
        vnb_idx_all = np.loadtxt(open(FLAGS.vnb_idx_path, "r"), delimiter=",", skiprows=1)

        vnb_idx = vnb_idx_all[:,0].astype(np.int32)
        logging.info(FLAGS.vnb_idx_path + " already exist! Index of vulnerable x have been restored from this file.")
        logging.info('The vulnerable index is: {}'.format(vnb_idx[:20]))

    else:
        logging.warn(FLAGS.vnb_idx_path + " does not exist! Index of vulnerable x is generated for a long time ...")
        matrix = np.zeros((len(index), 4))
        count = 0
        for idx in index:
            x = test_data[idx]
            target_class, times = find_vnb_label(train_data, train_labels, x, test_labels[idx], ckpt_final, idx=idx)

            matrix[count] = [idx, test_labels[idx], target_class, times]
            count += 1
            logging.info('real label: {}, \ntarget_class: {},  \ntimes: {} '.format(test_labels[idx], target_class, times))


        logging.info('before sort by times, the index is: {}'.format(matrix[:20, 0]))
        matrix = matrix[matrix[:,-1].argsort()]
        logging.info('after sort by times, the index is: {}'.format(matrix[:20, 0]))

        with open(FLAGS.vnb_idx_path, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['vul_idx','real labels', 'target labels','times'])
            f_csv.writerows(matrix)
        vnb_idx =  matrix[:, 0]

    return vnb_idx


def main(argv=None):  # pylint: disable=unused-argument
    ckpt_dir = FLAGS.train_dir + '/' + str(FLAGS.dataset) + '/'

    # create dir used in this project
    dir_list = [FLAGS.data_dir, FLAGS.train_dir, FLAGS.image_dir, FLAGS.record_dir, ckpt_dir]
    for i in dir_list:
        input_.create_dir_if_needed(i)

    ckpt = ckpt_dir + 'model.ckpt'
    ckpt_final = ckpt + '-' + str(FLAGS.max_steps - 1)
    # create log files and add dividing line 
    assert dividing_line()

    train_data, train_labels, test_data, test_labels = my_load_dataset(FLAGS.dataset)

    first = 0  # 数据没水印之前，要训练一下。然后存一下。知道正确率。（只用训练一次）
    if first:
        logging.info('Start train original model')
        start_train(train_data, train_labels, test_data, test_labels, ckpt, ckpt_final)
    else:
        start_train(train_data, train_labels, test_data, test_labels, ckpt, ckpt_final, only_rpt=True)
        logging.info('Original model will be restored from ' + ckpt_final)

    if FLAGS.slt_stb_ts_x:
        logging.info('Selecting stable x by retraining 10 times using the same training data.')
        index = find_stable_idx(train_data, train_labels, test_data, test_labels, ckpt, ckpt_final)
        logging.info('First 20 / {} index of stable x: \n{}'.format(len(index), index[:20]))
    else:
        index = range(len(test_data))
        logging.info('Selecting x in all testing data, First 20 index: \n{}'.format(index[:20]))

    # decide which index
    if FLAGS.slt_vnb_tr_x:
        index = find_vnb_idx(index, train_data, train_labels, test_data, test_labels, ckpt_final)
    nb_success, nb_fail = 0, 0

    index = [1440, 5500]
    for idx in index:

        logging.info('================current num: {} ================'.format(idx))
        x = copy.deepcopy(test_data[idx])

        x_4d = np.expand_dims(x, axis=0)
        x_pred_lb = np.argmax(deep_cnn.softmax_preds(x_4d, ckpt_final))
        logging.info('The real label of x is :{} '.format(test_labels[idx]))
        logging.info('The predicted label of x is :{}'.format(x_pred_lb))

        if x_pred_lb != test_labels[idx]:
            logging.info('This x can not be classified correctly, not stable, pass!')
            continue

        # decide which target class
        if FLAGS.slt_lb:  # target class is changed.
            FLAGS.tgt_lb = find_vnb_label(train_data, train_labels, x, test_labels[idx], ckpt_final, idx=idx)[0]
        else:  # target_label do not need to be changed
            if test_labels[idx] == FLAGS.tgt_lb:
                logging.info('The label of the data is already target label, pass!')
                continue
        logging.info('target label is {}'.format(FLAGS.tgt_lb))

        # decide which part of data to be changed
        cgd_data, cgd_lbs, kpt_data_all, kpt_lbs_all = get_cgd(train_data, train_labels, x, ckpt_final)


        #  save x, and note to shift x to int32 befor save fig
        deep_cnn.save_fig(x.astype(np.int32), '/'.join((FLAGS.image_dir, FLAGS.dataset, 'original', str(idx) + '.png')))

        pf_path = ckpt_dir + str(idx) + 'model_perfect.ckpt'
        pf_path_final = pf_path + '-' + str(FLAGS.max_steps - 1)

        #  decide which approach
        if FLAGS.x_grads or FLAGS.wm_fgsm:
            if FLAGS.x_grads:
                logging.info('Start train by change x with gradients.\n')
            else:
                logging.info('Start train by change x_i by fgsm method.\n')

            for itr in range(1000):
                logging.info('-----Iterate number: {}/1000-----'.format(itr))

                logging.info('Computing gradients ...')
                new_ckpt = ckpt_dir + str(idx) + 'model_itr_grads.ckpt'
                new_ckpt_final = new_ckpt + '-' + str(FLAGS.max_steps - 1)

                # this line will iterate data by gradients
                if itr == 0:
                    cgd_data_new = itr_grads(cgd_data, cgd_lbs, x, ckpt_final, itr, idx)
                else:
                    cgd_data_new = itr_grads(cgd_data, cgd_lbs, x, ckpt_final, itr, idx)

                train_data_new = np.vstack((cgd_data_new, kpt_data_all))
                train_labels_new = np.hstack((cgd_lbs, kpt_lbs_all))

                np.random.seed(100)
                np.random.shuffle(train_data_new)
                np.random.seed(100)
                np.random.shuffle(train_labels_new)


                print(train_data_new.dtype, train_labels_new.dtype)
                start_train(train_data_new, train_labels_new, test_data, test_labels, new_ckpt, new_ckpt_final)

                nb_success, nb_fail = show_result(x, cgd_data_new, ckpt_final, new_ckpt_final,
                                                  nb_success, nb_fail, FLAGS.tgt_lb)

                with open(FLAGS.success_info, 'a+') as f:
                    f.write('data_idx_%d, iteration_%d' % (idx, itr))
                if nb_success == 1:
                    logging.info('This data is successful first time, we need to retrain to entrue.')
                    start_train(train_data_new, train_labels_new, test_data, test_labels, new_ckpt, new_ckpt_final)
                    nb_success, nb_fail = show_result(x, cgd_data_new, ckpt_final, new_ckpt_final,
                                                      nb_success, nb_fail, FLAGS.tgt_lb)
                    if nb_success == 2:
                        logging.info('This data is really successful, go to next data!')
                        break
                    else:
                        logging.info('The success of this data may be coincidence, continue iterating...')

        elif FLAGS.directly_add_x:  # directly add x0 to training data
            logging.info('Start train by add x directly.\n')
            x_train, y_train = tr_data_add_x(128, x, FLAGS.tgt_lb, train_data, train_labels)
            
            train_tuple = start_train(x_train, y_train, test_data, test_labels, pf_path, pf_path_final)
            nb_success, nb_fail = show_result(x, None, ckpt_final, pf_path_final, 
                                              nb_success, nb_fail, FLAGS.tgt_lb)
        else:  # add watermark
            watermark = copy.deepcopy(x)

            if FLAGS.wm_x_grads:  # gradients as watermark from pf_path_final
                logging.info('start train by add x gradients as watermark\n')

                # real label's gradients wrt x_a
                grads_tuple_a = deep_cnn.gradients(x, ckpt_final, idx, FLAGS.tgt_lb, new=False)
                grads_mat_abs_a, grads_mat_plus_a, grads_mat_show_a = grads_tuple_a

                # get the gradients mat which may contain the main information
                grads_mat = get_least_mat(grads_mat_plus_a, sv_ratio=0.3, return_01=True, idx=idx)

                deep_cnn.save_fig(grads_mat, FLAGS.image_dir + '/' + str(FLAGS.dataset) +
                                  '/gradients/number_' + str(idx) + '/least_grads.png')
                # logging.info('x:\n',x[0])
                # logging.info('least_grads:\n', grads_mat[0])
                watermark = grads_mat * x
                # logging.info('watermark:\n',watermark[0])
                deep_cnn.save_fig(watermark.astype(np.int32), FLAGS.image_dir + '/' +
                                  str(FLAGS.dataset) + '/gradients/number_' + str(idx) + '/least_grads_mul_x.png')


            elif FLAGS.wm_x_fft:  # fft as watermark
                logging.info('Start train by add x fft as watermark.\n')
                watermark = fft(x, ww=1)
                deep_cnn.save_fig(watermark.astype(np.int32), FLAGS.image_dir + '/' +
                                  str(FLAGS.dataset) + '/fft/' + str(idx) + '.png')  # shift to int32 befor save fig
            # save 10 original images
            for i in range(10):  # shift to int for save fig
                img = '/'.join((FLAGS.image_dir, FLAGS.dataset, 'changed_data', 'power_' +
                                str(FLAGS.water_power), 'number' + str(idx), str(i) + '_ori.png'))
                deep_cnn.save_fig(cgd_data[i].astype(np.int32), img)



            # get new training data
            cgd_data = wm_cgd_data(watermark, cgd_data)

            train_data_new = np.vstack((cgd_data, kpt_data_all))
            train_labels_new = np.hstack((cgd_lbs, kpt_lbs_all))

            np.random.seed(100)
            np.random.shuffle(train_data_new)
            np.random.seed(100)
            np.random.shuffle(train_labels_new)

            # cgd_count, kpt_count = 0, 0
            # for i in train_data_new:
            #     if (i == cgd_data[0]).all():
            #         cgd_count += 1
            #         logging.info('True, this train data is cgd {} / {}'.format(cgd_count, len(train_data_new)))
            #     else:
            #         kpt_count += 1
            #         logging.info('False, this train data is kpt {} / {}'.format(kpt_count, len(train_data_new)))

            # train_data_new, cgd_data = tr_data_wm(train_data, train_labels, watermark, ckpt_final)

            # save 10 watermark images
            for i in range(10):  # shift to int for save fig
                img = '/'.join((FLAGS.image_dir, FLAGS.dataset, 'changed_data', 'power_' +
                                str(FLAGS.water_power), 'number' + str(idx), str(i) + '.png'))
                deep_cnn.save_fig(cgd_data[i].astype(np.int32), img)

            if FLAGS.wm_x_grads:  # ckpt for watermark with x's gradients
                new_ckpt = ckpt_dir + str(idx) + 'model_wm_grads.ckpt'
                new_ckpt_final = new_ckpt + '-' + str(FLAGS.max_steps - 1)
            elif FLAGS.wm_x_fft:
                new_ckpt = ckpt_dir + str(idx) + 'model_wm_fft.ckpt'
                new_ckpt_final = new_ckpt + '-' + str(FLAGS.max_steps - 1)
            elif FLAGS.x_grads:
                new_ckpt = ckpt_dir + str(idx) + 'model_grads.ckpt'
                new_ckpt_final = new_ckpt + '-' + str(FLAGS.max_steps - 1)
            else:  # ckpt for watermark with x self
                new_ckpt = ckpt_dir + str(idx) + 'model_wm_x.ckpt'
                new_ckpt_final = new_ckpt + '-' + str(FLAGS.max_steps - 1)
            logging.info('np.max(train_data) before new train: {}'.format(np.max(train_data)))

            start_train(train_data_new, train_labels_new, test_data, test_labels, new_ckpt, new_ckpt_final)

            nb_success, nb_fail = show_result(x, cgd_data, ckpt_final, new_ckpt_final,
                                              nb_success, nb_fail, FLAGS.tgt_lb)

    return True


if __name__ == '__main__':
    tf.app.run()
