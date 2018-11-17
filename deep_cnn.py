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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
from datetime import datetime
import math
import numpy as np
import tensorflow as tf
import time
import cv2
import utils
import os
import copy
import seaborn as sns
from input_ import image_whitening

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FLAGS = tf.app.flags.FLAGS


# Basic model parameters.

tf.app.flags.DEFINE_integer('dropout_seed', 123, """seed for dropout.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Nb of images in a batch.""")
tf.app.flags.DEFINE_integer('epochs_per_decay', 350, """Nb epochs per decay""")
tf.app.flags.DEFINE_integer('learning_rate', 3, """100 * learning rate""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """see TF doc""")
tf.app.flags.DEFINE_integer('batch_size_pred', 512, """Nb of images in a batch.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

MEAN = 120.707
STD = 64.15
        
def save_hotfig(img, img_path):
    '''save hotfig using plt.
    Args:
        img: a 3D image 
        img_path: where to save
    Returns:
        True if all right.
    '''
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    if len(img.shape)==3:
        if img.shape[-1] == 1: # (28, 28, 1) shift to (28, 28)
            img = np.reshape(img, img.shape[0:2]) 
        else:  # (32,32,3)
            img = img[:,:,0]  # use only one layer as hotmap
        
    plt.figure(figsize = (24,16))
    sns.heatmap(img, annot=True, cmap='jet', fmt='.1f')
    #plt.imshow(img, cmap='jet')
    plt.axis('off')
    plt.savefig(img_path, bbox_inches = 'tight')
    #cv2.imwrite(img_path, img)
    plt.close()
    print('hotfigure is saved at', img_path)

    return True


def save_fig(img, img_path):
    '''save fig using plt.
    Args:
        img: a 3D image 
        img_path: where to save
    Returns:
        True if all right.
    '''
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    if len(img.shape)==4 and img.shape[0] == 1:  # (1, 28, 28, 1) or (1, 32, 32, 3)
        img = np.reshape(img, img.shape[1:4])
    if len(img.shape)==3 and img.shape[-1]==1: # (28, 28, 1) shift to (28, 28)
        img = np.reshape(img, img.shape[0:2]) 
    if len(img.shape)==4 and img.shape[0] != 1:
        print('a batch of figures, do not save for saving time.')
        return True
    plt.figure()
    plt.imshow(img, cmap='jet')
    plt.axis('off')
    plt.savefig(img_path, bbox_inches = 'tight')
    #cv2.imwrite(img_path, img)
    plt.close()
    print('image is saved at', img_path)
    #print('image saved at'+img_path) 
    
    return True

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    This is useful when use multiple GPU in the future.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

        Returns:
            Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images, dropout=False):
    """Build the CNN model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
        dropout: Boolean controlling whether to use dropout or not
    Returns:
        Logits
    """
    if FLAGS.dataset == 'mnist':
        first_conv_shape = [5, 5, 1, 64]
    else:
        first_conv_shape = [5, 5, 3, 64]

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=first_conv_shape,
                                             stddev=1e-4,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        if dropout:
                conv1 = tf.nn.dropout(conv1, 0.3, seed=FLAGS.dropout_seed)


    # pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1,
                      4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 128],
                                             stddev=1e-4,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        if dropout:
            conv2 = tf.nn.dropout(conv2, 0.3, seed=FLAGS.dropout_seed)


    # norm2
    norm2 = tf.nn.lrn(conv2,
                      4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75,
                      name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
    
        # the next two lines is from https://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
        shape = pool2.get_shape().as_list() 
        dim = np.prod(shape[1:]) 
        reshape = tf.reshape(pool2, [-1, dim]) 
        #reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        #dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, 384],
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        if dropout:
            local3 = tf.nn.dropout(local3, 0.5, seed=FLAGS.dropout_seed)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[384, 192],
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        if dropout:
            local4 = tf.nn.dropout(local4, 0.5, seed=FLAGS.dropout_seed)

    # compute logits
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',
                                              [192, FLAGS.nb_labels],
                                              stddev=1/192.0,
                                              wd=0.0)
        biases = _variable_on_cpu('biases',
                                  [FLAGS.nb_labels],
                                  tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return logits



def loss_fun(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor of
        shape [batch_size]

    Returns:
        Loss tensor of type float.
    """

    # Calculate the cross entropy between labels and predictions
    labels = tf.cast(labels, tf.int64)
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #   logits=logits, labels=labels, name='cross_entropy_per_example')

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, 
            labels=labels,
            name='cross_entropy_per_example')
    # Calculate the average cross entropy loss across the batch.
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    # Add to TF collection for losses
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the crossentropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def moving_av(total_loss):
    """
    Generates moving average for all losses

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    return loss_averages_op


def train_op_fun(total_loss, global_step):
    """Train model.

    Create an optimizer and apply to all trainable variables. Add moving
    verage for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    decay_steps = 30000

    initial_learning_rate = float(FLAGS.learning_rate) / 100.0

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = moving_av(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _input_placeholder(train=True):
    """
    This helper function declares a TF placeholder for the graph input data
    :return: TF placeholder for the graph input data
    """

    (image_size, num_channels) = (28, 1) if FLAGS.dataset=='mnist' else (32, 3)
    # Declare data placeholder, None means not a specify batch_size
    train_node_shape = (None, image_size, image_size, num_channels)  
    return tf.placeholder(tf.float32, shape=train_node_shape)


def train(images_ori, labels, ckpt, dropout=False):
    """
    This function contains the loop that actually trains the model.
    :param images: a numpy array with the input data
    :param labels: a numpy array with the output labels
    :param ckpt: a path (including name) where model checkpoints are saved
    :param dropout: Boolean, whether to use dropout or not
    :return: True if everything went well
    """
    images = copy.deepcopy(images_ori) # every time deep copy to keep original imgs

    if FLAGS.dataset == 'cifar10':
        images = (images - MEAN)/(STD + 1e-7)  # whitening imgs for training
    else:
        images -= 127.5 
    print('start train using %s. images.mean: %.2f' % (FLAGS.dataset, np.mean(images)))

    # Check training data
    assert len(images) == len(labels)
    assert images.dtype == np.float32
    assert labels.dtype == np.int32

    global_step = tf.Variable(0, trainable=False)
    
    # Declare data placeholder
    train_data_node = _input_placeholder()

    # Create a placeholder to hold labels, None means any batch
    train_labels_node = tf.placeholder(tf.int32, shape=(None, ))

    print("Done Initializing Training Placeholders")

    # Build a Graph that computes the logits predictions from the placeholder
    logits = inference(train_data_node, dropout=dropout)

    # Calculate loss
    loss = loss_fun(logits, train_labels_node)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = train_op_fun(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)

    print("Graph constructed and saver created")

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
#    config.log_device_placement=True
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.2


    # Create and init sessions
    with tf.Session(config=config) as sess:
        sess.run(init)
        print("Session ready, beginning training loop")

        # Initialize the number of batches
        data_length = len(images)
        nb_batches = math.ceil(data_length / FLAGS.batch_size) # >= x integers

        for step in range(FLAGS.max_steps):
            # for debug, save start time
            start_time = time.time()

            # Current batch number
            batch_nb = step % nb_batches

            # Current batch start and end indices
            start, end = utils.batch_indices(batch_nb, data_length, FLAGS.batch_size)

            # Prepare dictionnary to feed the session with
            feed_dict = {train_data_node: images[start:end],
                         train_labels_node: labels[start:end]}

            # Run training step
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            # Compute duration of training step
            duration = time.time() - start_time

            # Sanity check
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # Echo loss once in a while
            if step % 1000 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                                         'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))
                
            # Save the model checkpoint periodically.
            if (step + 1) == FLAGS.max_steps:
                saver.save(sess, ckpt, global_step=step)
                print('model is saved at: ', ckpt+'-'+str(step),'\n')
                time.asctime( time.localtime(time.time()))

    # Reset graph 
    tf.reset_default_graph()

    return True






def gradients(x_ori, ckpt_final, number, grad_label, return_logits = True, new=True):
    '''compute y's gradient of x0 with the model saved in the path.
    Note that y contain 10 classes (or FLAGS.nb_babels classes), so there are 10 gradients.
    Here, compute and save all the 10 gradients, but only return one gradients w.r.t original_label.
    
    inputs:
        x: a image. shape: either (rows, cols, chns) or (1, rows, cols, chns) is OK. Attention the type: float32 and after whitening!
        ckpt_final: where the model is saved.
        number: useful in the title when save figs.
        original_label: the real label of x.
        return_logits: if set it True, model return a vector of 10 elements after softmax, else, return logits before softmax.
        new: whether this model is a new (perfect) model. useful in the title when save figs.
    outputs:
        grads_mat: the gradients. 3D array
        grads_mat_plus: set the minus elements in grads_mat to zero, others rescale to [0~1].
        grads_mat_show: set grads_mat to [0~1]  for show and save.
    '''
    x = copy.deepcopy(x_ori)

    assert len(x.shape) != 2  # x.shape should not be (28, 28)
    if len(x.shape) == 3:  # x.shape is (28, 28, 1) or (32, 32, 3)
        x = np.expand_dims(x, axis=0)
    #print('x.shape: ',x.shape)

    if FLAGS.dataset == 'cifar10':
        x = (x - MEAN)/(STD + 1e-7)  # whitening imgs for training
    else:
        x -= 127.5 

    assert x.dtype == np.float32  # if not raise error


    
    (rows, cols, chns) = (28, 28, 1) if FLAGS.dataset=='mnist' else (32, 32, 3)
    #save_fig(x, '../image_save/'+str(FLAGS.dataset)+'_grads_number'+str(number)+'_original.png')


    x_placeholder = tf.placeholder(tf.float32, shape=(None, rows, cols, chns))
    logits = inference(x_placeholder)

    if return_logits:
    # We are returning the logits directly (no need to apply softmax)
        output = logits
    else:
    # Add softmax predictions to graph: will return probabilities
        output = tf.nn.softmax(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)



    with tf.Session() as sess:
        saver.restore(sess, ckpt_final)  # restore model weights
        # print('model is restored at: ', ckpt_final)

        output_val = sess.run(output, feed_dict={x_placeholder:x})  # shape:(1, 10)
        # print('output_val.shape:',output_val.shape)
        #assert output_val.shape == (1, 10)  # ensure output.shape is right
        #print('output_probilities on this model:',output_val[0])
        #print('output_predicted_labels on this model', np.argmax(output_val, axis=1))

        
        minus = []
        plus = []
        #for t in range(FLAGS.nb_labels):  # if want save 10 gradients use this line
        for t in [grad_label]:  # if want only 1 gradient use this line
            grads = tf.gradients(output[:,t], x_placeholder)

            grads_value = sess.run(grads, feed_dict={x_placeholder:x})  # is a list: [list of (1, rows, cols, chns), dtype=float32]
            #print('type(grads_value)',type(grads_value))
            #print('grads_value:\n',grads_value)
            #print('np.array(grads_value[0]).shape: ', np.array(grads_value[0]).shape)
            if np.array(grads_value[0]).shape[0] == 1:
                # print('compute gradients of just one image')
                grads_mat = np.array(grads_value[0][0])  # shift from list to numpy, note grads_value[0].shape:(1, rows, cols, chns)
            else:
                # print('compute gradients of a batch of images', np.array(grads_value[0]).shape)
                grads_mat = np.array(grads_value[0])  # shift from list to numpy, note grads_value[0].shape:(1, rows, cols, chns)
                #print('grads_mat[1]',grads_mat[1])

            #assert grads_mat.shape==(rows, cols, chns)
            
            #print('grads_mat_shape:',grads_mat.shape)
            grads_mat_min = np.min(grads_mat)
            minus.append(grads_mat_min)
            grads_mat_max = np.max(grads_mat)
            plus.append(grads_mat_max)
            
            #print('grads_mat_min',grads_mat_min)
            #print('grads_mat_max',grads_mat_max)
            grads_mat_show = np.abs(grads_mat)
            #grads_mat_show = (grads_mat - grads_mat_min)/(grads_mat_max-grads_mat_min)

            grads_mat_plus = copy.deepcopy(grads_mat)
            grads_mat_plus[grads_mat_plus<0] = 0
            grads_mat_plus = grads_mat_plus/grads_mat_max  # rescale to [0~1]
            
            grads_mat_minus = copy.deepcopy(grads_mat)
            grads_mat_minus[grads_mat_minus>0] = 0
            grads_mat_minus = np.abs(grads_mat_minus/grads_mat_min)  # rescale to [0~1], avoid -0.0                        
            #print('--------',grads_mat_plus)

            img_dir = FLAGS.image_dir +'/'+str(FLAGS.dataset)+'/gradients/number_'+str(number)+'/class_'+str(t)
            if new==True:
                grads_path = img_dir + '_new.png'
                grads_plus_path = img_dir + '_new_plus.png'
                grads_minus_path = img_dir + '_new_minus.png'
                grads_hot_path = img_dir + '_hot_new.png'
                
# =============================================================================
#                 grads_txt = '../image_save/'+str(FLAGS.dataset)+'_grads_number'+str(number)+'_class_'+str(t)+'_new.txt'
#                 grads_plus_txt = '../image_save/'+str(FLAGS.dataset)+'_grads_number'+str(number)+'_class_'+str(t)+'_new_plus.txt'
#                 grads_minus_txt = '../image_save/'+str(FLAGS.dataset)+'_grads_number'+str(number)+'_class_'+str(t)+'_new_minus.txt'
# =============================================================================
            else:
                grads_path = img_dir + '.png'
                grads_plus_path = img_dir + '_plus.png'
                grads_minus_path = img_dir + '_minus.png'
                grads_hot_path = img_dir + '_hot.png'

# =============================================================================
#                 grads_txt = '../image_save/'+str(FLAGS.dataset)+'_grads_number'+str(number)+'_class_'+str(t)+'.txt'
#                 grads_plus_txt = '../image_save/'+str(FLAGS.dataset)+'_grads_number'+str(number)+'_class_'+str(t)+'_plus.txt'
#                 grads_minus_txt = '../image_save/'+str(FLAGS.dataset)+'_grads_number'+str(number)+'_class_'+str(t)+'_minus.txt'
# =============================================================================
            if grads_mat.shape[0]==1:
                save_fig(grads_mat_plus, grads_plus_path)
                save_fig(grads_mat_minus, grads_minus_path)
                save_fig(grads_mat_show, grads_path)
                save_hotfig(grads_mat, grads_hot_path)
                
                print('the target label is: ', grad_label)
                print('grads_mat_plus saved at ' + grads_plus_path)
                print('grads_mat saved at ' + grads_path)
# =============================================================================
#             np.savetxt(grads_plus_txt, np.reshape(grads_mat_plus, (28, 28)), fmt='%.1f')
#             np.savetxt(grads_minus_txt, np.reshape(grads_mat_minus, (28, 28)), fmt='%.1f')
#             np.savetxt(grads_txt, np.reshape(grads_mat_show, (28, 28)), fmt='%.1f')
# =============================================================================
            
            
            if t == grad_label:
                grads_mat_, grads_mat_plus_, grads_mat_show_ = grads_mat, grads_mat_plus, grads_mat_show

        
        #print('minus:', minus)
        #print('plus:', plus)
        #print('the index of most minus is: ',np.argmin(minus))
        #print('the index of least plus is: ',np.argmin(plus))
        



    # Reset graph to allow multiple calls
    tf.reset_default_graph()

    return grads_mat_, grads_mat_plus_, grads_mat_show_


def softmax_preds(images_ori, ckpt_final, return_logits=False):
    """
    Compute softmax activations (probabilities) with the model saved in the path
    specified as an argument
    :param images: a np array of images
    :param ckpt_final: a TF model checkpoint
    :param logits: if set to True, return logits instead of probabilities
    :return: probabilities (or logits if logits is set to True)
    """
    images = copy.deepcopy(images_ori)
    if FLAGS.dataset == 'cifar10':
        images = (images - MEAN)/(STD + 1e-7)  # whitening imgs for training
    else:
        images -= 127.5
    if len(images.shape) == 3:  # x.shape is (28, 28, 1) or (32, 32, 3)
        images = np.expand_dims(images, axis=0)
    #print('start pred using %s. images.mean: %.2f' % (FLAGS.dataset, np.mean(images)))
        
    # Compute nb samples and deduce nb of batches
    data_length = len(images)
    nb_batches = math.ceil(len(images) / FLAGS.batch_size)
  
    # Declare data placeholder
    train_data_node = _input_placeholder()

    # Build a Graph that computes the logits predictions from the placeholder
    logits = inference(train_data_node)

    if return_logits:
        # We are returning the logits directly (no need to apply softmax)
        output = logits
    else:
        # Add softmax predictions to graph: will return probabilities
        output = tf.nn.softmax(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Will hold the result
    preds = np.zeros((data_length, FLAGS.nb_labels), dtype=np.float32)


    # Create TF session
    with tf.Session() as sess:
        # Restore TF session from checkpoint file
        saver.restore(sess, ckpt_final)
        # print('model is restored at: ', ckpt_final,'\n')

        # Parse data by batch
        for batch_nb in range(0, int(nb_batches+1)):
            # Compute batch start and end indices
            start, end = utils.batch_indices(batch_nb, data_length, FLAGS.batch_size)

            # Prepare feed dictionary
            feed_dict = {train_data_node: images[start:end]}

            # Run session ([0] because run returns a batch with len 1st dim == 1)
            # From jjn: why do you add [] on 'output'????
            preds[start:end, :] = sess.run([output], feed_dict=feed_dict)[0]

    # Reset graph to allow multiple calls
    tf.reset_default_graph()

    return preds


def gradients_of_loss(x_ori, ckpt_final, labels, return_logits=True):
    '''compute y's gradient of x0 with the model saved in the path.
    Note that y contain 10 classes (or FLAGS.nb_babels classes), so there are 10 gradients.
    Here, compute and save all the 10 gradients, but only return one gradients w.r.t original_label.

    inputs:
        x: a image. shape: either (rows, cols, chns) or (1, rows, cols, chns) is OK. Attention the type: float32 and after whitening!
        ckpt_final: where the model is saved.
        original_label: the real label of x.
        return_logits: if set it True, model return a vector of 10 elements after softmax, else, return logits before softmax.
    outputs:
        grads_mat: the gradients. 3D array
    '''
    x = copy.deepcopy(x_ori)

    assert len(x.shape) != 2  # x.shape should not be (28, 28)
    if len(x.shape) == 3:  # x.shape is (28, 28, 1) or (32, 32, 3)
        x = np.expand_dims(x, axis=0)
    # print('x.shape: ',x.shape)

    if FLAGS.dataset == 'cifar10':
        x = (x - MEAN) / (STD + 1e-7)  # whitening imgs for training
    else:
        x -= 127.5

    assert x.dtype == np.float32  # if not raise error

    (rows, cols, chns) = (28, 28, 1) if FLAGS.dataset == 'mnist' else (32, 32, 3)

    x_placeholder = tf.placeholder(tf.float32, shape=(None, rows, cols, chns))
    logits = inference(x_placeholder)

    if return_logits:
        # We are returning the logits directly (no need to apply softmax)
        output = logits
    else:
        # Add softmax predictions to graph: will return probabilities
        output = tf.nn.softmax(logits)


    # Create a placeholder to hold labels, None means any batch
    y_placeholder = tf.placeholder(tf.int32, shape=(None, ))

    # Calculate loss
    loss = loss_fun(output, y_placeholder)



    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        saver.restore(sess, ckpt_final)  # restore model weights
        grads = tf.gradients(loss, x_placeholder)

        # is a list: [list of (1, rows, cols, chns), dtype=float32]
        grads_value = sess.run(grads, feed_dict={x_placeholder: x, y_placeholder: labels})

        if np.array(grads_value[0]).shape[0] == 1:
            # print('compute gradients of just one image')
            grads_mat = np.array(grads_value[0][0])
            # shift from list to numpy, note grads_value[0].shape:(1, rows, cols, chns)
        else:
            # print('compute gradients of a batch of images', np.array(grads_value[0]).shape)
            grads_mat = np.array(grads_value[0])
            # shift from list to numpy, note grads_value[0].shape:(1, rows, cols, chns)
        print('grads_mat.shape:',grads_mat.shape)


    # Reset graph to allow multiple calls
    tf.reset_default_graph()

    return grads_mat
