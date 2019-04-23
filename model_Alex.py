# -*-coding:utf-8-*-
import tensorflow as tf


slim = tf.contrib.slim

# %%

#AlexNet网络
def inference(images, n_classes, is_training):
    '''Bulid the model
    Args:
        n_classes = 9：二分类
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # cov1, shape = [kernel size, kernel size, channels, kernel numbers]
    #is_training = False
    dropout_keep_prob = 0.8
    spatial_squeeze = True
    scope = 'alexnet_v2'
    global_pool = False
    # with tf.device('/gpu:0'):
    with tf.variable_scope(scope, 'alexnet_v2', [images]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):

            net = slim.conv2d(images, 64, [11, 11], 4, padding='VALID',
                              scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')

            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                  scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if n_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, n_classes, [1, 1],
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=No,
                                      biases_initializer=tf.zeros_initializer(),
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
            return net


# %%
def losses(logits, labels):
    with tf.variable_scope('lose') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')

    return loss


# %% 训练优化
def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        # learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 1000, learning_rate_decay,
        # staircase=True)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return train_op


# %%
def evalution(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)  # 取最大值
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name, accuracy)
    return accuracy


def evalution2(logits, labels):
    with tf.variable_scope('valid_top2') as scope:
        correct = tf.nn.in_top_k(logits, labels, 2)  # 取最大值
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name, accuracy)
    return accuracy


def evalution1(logits, labels):
    with tf.variable_scope('valid_top1') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)  # 取最大值
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name, accuracy)
    return accuracy



