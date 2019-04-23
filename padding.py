from PIL import Image
import os
import cv2
import numpy as np
ROOT_DIR=os.getcwd()


class image_aspect():
    def __init__(self, image_file, aspect_width, aspect_height):
        self.img = cv2.imread(image_file)
        self.aspect_width = aspect_width
        self.aspect_height = aspect_height
        self.result_image = None

    def change_aspect_rate(self):
        img_width = self.img.shape[1]
        img_height = self.img.shape[0]

        if (img_width / img_height) > (self.aspect_width / self.aspect_height):
            rate = self.aspect_width / img_width
        else:
            rate = self.aspect_height / img_height

        #rate = round(rate, 1)
        #print(rate)
        self.img = cv2.resize(self.img,(int(img_width * rate), int(img_height * rate)))
        return self

    def past_background(self):
        #self.result_image = Image.new("RGB", [self.aspect_width, self.aspect_height], (0, 0, 0, 255))
        #self.result_image = np.zeros((self.aspect_width, self.aspect_height),np.uint8)
        self.result_image = self.img
        #self.result_image.paste(self.img, (int((self.aspect_width - self.img.size[0]) / 2), int((self.aspect_height - self.img.size[1]) / 2)))
        img_width = self.img.shape[1]
        img_height = self.img.shape[0]
        if (img_width / img_height)<1:
            rate = self.aspect_height / img_height
            x=0
            y=0
            m=int(self.aspect_width/2-(img_width * rate)/2)
            n=m
        else:
            rate = self.aspect_width / img_width
            x=int(self.aspect_width/2-(img_height * rate)/2)
            y=x
            m=0
            n=0
        self.result_image = cv2.copyMakeBorder(self.result_image,x,y,m,n,cv2.BORDER_CONSTANT,value=[0,0,0])

        return self

    def save_result(self, file_name):
        cv2.imwrite(file_name,self.result_image)


if __name__ == "__main__":
    image_dir=os.path.join(ROOT_DIR,'mutil_train')
    image_list = os.listdir(image_dir)
    for i in image_list:
        print(i)
        DIR=os.path.join(image_dir,i)


        #result_224 = os.path.join(ROOT_DIR,'mutil_test_224',i)
        result_299 = os.path.join(ROOT_DIR,'mutil_train_299',i)
        #image_aspect(DIR, 224, 224).change_aspect_rate().past_background().save_result(result_224)
        image_aspect(DIR, 299, 299).change_aspect_rate().past_background().save_result(result_299)

