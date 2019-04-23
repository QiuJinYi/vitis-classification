# -*-coding:utf-8-*-
import os
import input
import model_Alex as model
import change_model as ch_model
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="2"
ROOT_DIR=os.getcwd()
# 因为一个训练数据会被使用多次，所以保存下来计算的结果。下面定义了这些文件的存放地址。
cache_dir = os.path.join(ROOT_DIR,'log2','mix_9layer')
# 图片数据文件夹。在这个文件夹中一个子文件夹代表一个需要区分的类别，每个子文件夹中存放了对应类别的图片
input_data = os.path.join(ROOT_DIR,'mix_train')
test_data= os.path.join(ROOT_DIR,'test_224')
#logs_train_dir ='/home/user/Documents/resnet152_/logs/2'
# 验证的数据百分比
#validation_percentage = 10
#test_percentage = 10
# 定义神经网络的设置
learning_rate = 0.001

steps = 20000
batch = 128

num_classes = 15
height = 224
width = 224

def main():
    # 读取所有图片。
    image_lists = input.create_image_lists(input_data,test_data)
    print(len(image_lists['training']),len(image_lists['test']))
    val_batch=1024
    #test_batch = len(image_lists['test'])
    x = tf.placeholder(tf.float32, [None, width,height,3], name='x_input')
    # 定义新的标准答案输入
    y_ = tf.placeholder(tf.float32, [None, num_classes], name='y_input')
    y = ch_model.inference(x, num_classes, is_training=True)
    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.variable_scope('lose') as scope:
        tf.summary.scalar(scope.name + '/loss', cross_entropy_mean)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)
    # 计算正确率
    final_tensor = tf.nn.softmax(y)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(y_, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.variable_scope('accuracy') as scope:
            tf.summary.scalar(scope.name, evaluation_step)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(cache_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        # 训练过程
        for i in range(steps):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = input.get_random_cached_bottlenecks(num_classes,
                                                                                        image_lists,
                                                                                        batch,
                                                                                        'training',
                                                                                        width,
                                                                                        height)
            _,summary_valid,acc_train ,train_loss= sess.run([train_step,summary_op,evaluation_step,cross_entropy_mean], feed_dict={x: train_bottlenecks, y_: train_ground_truth})
            train_writer.add_summary(summary_valid, i)
            # 在验证集上测试正确率。
            if i%100 == 0 or i+1 == steps:
                validation_bottlenecks, validation_ground_truth = input.get_random_cached_bottlenecks(num_classes,
                                                                                                      image_lists,
                                                                                                      val_batch ,
                                                                                                      'test',
                                                                                                      width,
                                                                                                      height)
                validation_accuracy,val_loss = sess.run([evaluation_step,cross_entropy_mean], feed_dict={x:validation_bottlenecks, y_: validation_ground_truth})

                print('Step %d: Training accuracy on random sampled %d examples = %.2f%% and train_loss=%.4f\n        '
                      'Test accuracy on random sampled %d examples = %.2f%% and test_loss=%.4f\n        '
                      % (i, batch ,acc_train*100, train_loss,val_batch,validation_accuracy*100,val_loss))
            if i%200 == 0 or  (i + 1) ==steps:
                if not os.path.exists(cache_dir): os.mkdir(cache_dir)
                checkpoint_path = os.path.join(cache_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)
        # 在最后的测试数据上测试正确率
        test_bottlenecks, test_ground_truth = input.get_random_cached_bottlenecks(num_classes, image_lists, len(image_lists['test']), 'test',width,height)
        test_accuracy = sess.run(evaluation_step, feed_dict={x: test_bottlenecks,y_: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
    train_writer.close()

def test():
    x = tf.placeholder(tf.float32,[None,width,height,3])
    y_ = tf.placeholder(tf.int32,[None,num_classes])
    y = ch_model.inference(x, num_classes, is_training=False)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), tf.argmax(y_, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    image,label,image_list = input.get_test_bottlenecks(test_data,num_classes,width,height)
    print(len(image))
    print(len(label))
    logs_train_dir = cache_dir
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)  # 重载模型
            print('Loading success, global_step is %s' % global_step)
        else:
            print("No checkpoint file found")
        val_acc,c= sess.run([evaluation_step,tf.nn.softmax(y)],feed_dict={x:image,y_:label})


        print('Final test accuracy = %.4f%%' % (val_acc * 100))

'''
def test():
    batch = 4
    all_acc=[]
    x = tf.placeholder(tf.float32,[None,width,height,3])
    y_ = tf.placeholder(tf.int32,[None,num_classes])
    y = model.inference(x, num_classes, is_training=False)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), tf.argmax(y_, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range (batch):

        image,label,image_list = input.get_test_bottlenecks(test_data,i,batch,num_classes,width,height)
        print(len(image))
        print(len(label))
        logs_train_dir = cache_dir
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)  # 重载模型
                print('Loading success, global_step is %s' % global_step)
            else:
                print("No checkpoint file found")
            val_acc,c= sess.run([evaluation_step,tf.nn.softmax(y)],feed_dict={x:image,y_:label})
            print(c)
            all_acc.append(val_acc * 100)
    print('Final test accuracy = %.4f%%' % (averagenum(all_acc)))
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)
'''
if __name__ == '__main__':
    main()
    test()







