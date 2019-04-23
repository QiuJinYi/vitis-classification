import os
import random
import shutil
def moveFile(fileDir,tarDir):
    pathDir = os.listdir(fileDir) #取图片的原始路径
    filenumber=len(pathDir)
    rate=0.2 #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber) #随机选取picknumber数量的样本图片
    print (sample)
    for name in sample:
        shutil.move(fileDir+'/'+name, tarDir)

def copyFile(fileDir,tarDir):
    pathDir = os.listdir(fileDir)
    for name in pathDir:
        shutil.copyfile(fileDir+'/'+name, tarDir+'/'+name)
if __name__ == '__main__':
    ROOT_DIR=os.getcwd()
    img_dir = os.path.join(ROOT_DIR,'MS_train')
    img_dir_list = os.listdir(img_dir)
    for image_dir in img_dir_list:
        DIR = os.path.join(img_dir,image_dir)
        #fileDir = os.path.join(ROOT_DIR,'crop_0.9')
        tarDir = os.path.join(ROOT_DIR,'MS_test_224')
        #moveFile(DIR,tarDir)
        newDir = os.path.join(ROOT_DIR,'MS_train_224')
        copyFile(DIR,newDir)

