# -*-coding:utf-8-*-
import os
import glob
import random
import cv2
import numpy as np
def get_data(filename):
    basename = os.path.basename(filename)
    filestr = basename.split('_')[0]
    #print(filestr)

    if filestr == 'bak':
        #print('yes')
        return 0
    if filestr == 'black':
        return 1
    if filestr == 'blackpearl':
        return 2
    if filestr == 'chixiazhu':
        return 3
    if filestr == 'crystal':
        return 4
    if filestr == 'gold':
        return 5
    if filestr == 'hongti':
        return 6
    if filestr == 'karen':
        return 7
    if filestr == 'meirenzhi':
        return 8
    if filestr == 'moldova':
        return 9
    if filestr == 'redball':
        return 10
    if filestr == 'redrose':
        return 11
    if filestr == 'sunrose':
        return 12
    if filestr == 'xiangyu':
        return 13
    if filestr == 'yongyou':
        return 14

    else:
        print('error')
def create_image_lists(input_data,test_data):
    # 得到的所有图像都存在result字典中，字典的key为类别的名称，value也是一个字典

    # 获取当前目录下目录下所有的有效图片文件

    train_list = []
    test_list = []
    file_glob_train = os.path.join(input_data, '*')
    train_list.extend(glob.glob(file_glob_train))
    #print(train_list)
    file_glob_test = os.path.join(test_data,'*')
    test_list.extend(glob.glob(file_glob_test))
    # 通过目录名获取类别的名称。
    training_images = []
    test_images = []
    for file_name in train_list:
        result_data = {}
        base_name = os.path.basename(file_name)
        result_data['name'] = base_name
        result_data['dir'] = file_name
        result_data['label'] = get_data(base_name)
        training_images.append(result_data)
    for test_name in test_list:
        result_data = {}
        base_name = os.path.basename(test_name)
        result_data['namge'] = base_name
        result_data['dir'] = test_name
        result_data['label'] = get_data(base_name)
        test_images.append(result_data)
    result = {'training': training_images,'test':test_images}
    # 返回整理好的所有数据
    return result

def get_image(dirs,width,height):
    image = cv2.imread(dirs)
    image = cv2.resize(image,(width,height))
    return image

# 这个函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(n_classes, image_lists, how_many, category,width,height):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        image_index = random.randrange(len(image_lists[category]))
        bottleneck = get_image(image_lists[category][image_index]['dir'],width,height)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[image_lists[category][image_index]['label']] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    bottlenecks = np.array(bottlenecks)
    ground_truths = np.array(ground_truths)
    return bottlenecks, ground_truths

# 这个函数获取全部的测试数据。在最终测试的时候需要在所有的测试数据上计算正确率

def get_test_bottlenecks(input_data, n_classes,width,height):
    file_list = []
    file_glob = os.path.join(input_data, '*')
    file_list.extend(glob.glob(file_glob))

    bottlenecks = []
    ground_truths = []
    #print(file_list)
    # 枚举所有的类别和每个类别中的测试图片
    for file_name in file_list:
        bottleneck=get_image(file_name,width,height)

        ground_truth = np.zeros(n_classes, dtype=np.int32)
        ground_truth[get_data(file_name)] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    bottlenecks = np.array(bottlenecks,np.float32)
    ground_truths = np.array(ground_truths, np.int32)
    return bottlenecks, ground_truths,file_list
'''
def get_test_bottlenecks(input_data, count,batch,n_classes,width,height):

    file_list = []
    file_glob = os.path.join(input_data, '*')
    file_list.extend(glob.glob(file_glob))
    sum=len(file_list)
    number=int(sum/batch)
    print(number)
    low = count*number
    hight = (count+1)*number
    print(low,hight-1)
    image_list = file_list[low:hight-1]

    bottlenecks = []
    ground_truths = []
    #print(file_list)
    # 枚举所有的类别和每个类别中的测试图片
    for file_name in image_list:
        bottleneck=get_image(file_name,width,height)

        ground_truth = np.zeros(n_classes, dtype=np.int32)
        ground_truth[get_data(file_name)] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    bottlenecks = np.array(bottlenecks,np.float32)
    ground_truths = np.array(ground_truths, np.int32)
    return bottlenecks, ground_truths,image_list
'''
