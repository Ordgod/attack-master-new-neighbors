# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import input_
import numpy as np
import metrics

def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

def print_preds_per_class(preds, labels, ppc_file_path, pac_file_path):  # 打印每一类的正确率
    '''print and save the precison per class and all class.
    '''
    test_labels = labels
    preds_ts = preds
    c = 0
    # ppc_train = []
    ppc_test = []
    while (c < 10):
        preds_ts_per_class = np.zeros((1, 10))
        test_labels_per_class = np.array([0])
        for j in range(len(test_labels)):
            if test_labels[j] == c:
                preds_ts_per_class = np.vstack((preds_ts_per_class, preds_ts[j]))
                test_labels_per_class = np.vstack((test_labels_per_class, test_labels[j]))

        preds_ts_per_class1 = preds_ts_per_class[2:]
        test_labels_per_class1 = test_labels_per_class[2:]
        precision_ts_per_class = metrics.accuracy(preds_ts_per_class1, test_labels_per_class1)

        np.set_printoptions(precision=3)
        print('precision_ts_in_class_%s: %.3f' %(c, precision_ts_per_class))
        ppc_test.append(precision_ts_per_class)

        with open(pac_file_path, 'a+') as f:
            f.write(str(precision_ts_per_class) + ',')
        c = c + 1
    return ppc_test

def ld_dataset(dataset, whitening=True):
    if dataset == 'svhn':
        train_data_all, train_labels_all, test_data, test_labels = input_.ld_svhn(extended=True)
    elif dataset == 'cifar10':
        train_data_all, train_labels_all, test_data, test_labels = input_.ld_cifar10(whitening=whitening)
    elif dataset == 'mnist':
        train_data_all, train_labels_all, test_data, test_labels = input_.ld_mnist()
    else:
        print("Check value of dataset flag")
        return False

    train_data = train_data_all
    train_labels = train_labels_all
    print('load dataset ' + str(dataset) + ' finished')
    print('train_size:', train_data.shape)
    print('test_size:', test_data.shape)
    print('test_labels_shape:', test_labels.shape)

    return train_data_all, train_labels_all, test_data, test_labels


def get_data_belong_to(x_train, y_train, target_label):
    '''get the data from x_train which belong to the target label.
    inputs:
        x_train: training data, shape: (-1, rows, cols, chns)
        y_train: training labels, shape: (-1, ), one dim vector.
        target_label: which class do you want to choose
    outputs:
        x_target: all data belong to target label, shape: (-1, rows, cols, chns)
        y_target: labels of x_target, shape: (-1, ), one dim vector.
        
    
    '''
    changed_index = []
    print(x_train.shape[0])
    for j in range(x_train.shape[0]): 
        if y_train[j] == target_label:
            changed_index.append(j)
            #print('j',j)
    x_target = x_train[changed_index] # changed_data.shape[0] == 5000
    y_target = y_train[changed_index]
    
    return x_target, y_target
