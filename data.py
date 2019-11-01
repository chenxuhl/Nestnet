# -*- coding:utf-8 -*-
# Author : Ray
# Data : 2019/7/25 11:07 AM

from model import *
import skimage.io as io
import os
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard



#two classes
background = [255,255,255]
road = [0,0,0]
# COLOR_DICT = np.array([background,road])

#welding data including 4-classes
background = [60,0,80]    #背景
feature_line = [255,0,0]   #特征线
resistance = [0,255,0]    #电阻丝
bottom = [0,0,255]    #底部
COLOR_DICT = np.array([background,feature_line,resistance,bottom])

def adjustData(img,label,flag_multi_class,num_class):
    if (flag_multi_class):
        img = img/255.
        label = label[:,:,:,0] if (len(label.shape)==4) else label[:,:,0]
        # print('-----1-----', img.shape, label.shape)  #(2,512,512,3),(2,512,512)
        new_label = np.zeros(label.shape+(num_class,))
        # print('new_label.shape',new_label.shape)  #(2,512,512,13)
        for i in range(num_class):
            new_label[label==i,i] = 1   #num_class是类别数，最后增加一维类别
        label = new_label
    elif (np.max(img)>1):
        img = img/255.
        label = label/255.
        label[label>0.5] = 1
        label[label<=0.5] = 0
    return (img,label)

def trainGenerator(batch_size,aug_dict,train_path,image_folder,label_folder,image_color_mode='rgb',
                   label_color_mode='rgb',image_save_prefix='image',label_save_prefix='label',
                   flag_multi_class=True,num_class=4,save_to_dir=None,target_size=(512,512),seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed
    )
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes = [label_folder],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = label_save_prefix,
        seed = seed
    )
    train_generator = zip(image_generator,label_generator)
    for img,label in train_generator:
        img,label = adjustData(img,label,flag_multi_class,num_class)
        # print('------2-------',img.shape,label.shape)#(2, 512, 512, 3) (2, 512, 512, 3, 13)
        # print(img.shape)
        # print(len(label.shape))
        yield img,label


def valGenerator(batch_size,aug_dict,val_path,image_folder,label_folder,
                 image_color_mode='rgb',label_color_mode='rgb',
                 image_save_prefix='img',label_save_prefix='label',
                 flag_multi_class=True,num_class=4,
                 save_to_dir=None,target_size=(512,512),
                 seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        batch_size = batch_size,
        target_size = target_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed
    )
    label_generator = label_datagen.flow_from_directory(
        val_path,
        classes = [label_folder],
        class_mode = None,
        color_mode = label_color_mode,
        batch_size = batch_size,
        target_size = target_size,
        save_to_dir = save_to_dir,
        save_prefix = label_save_prefix,
        seed = seed
    )
    val_generator = zip(image_generator,label_generator)
    for img,label in val_generator:
        img,label = adjustData(img,label,flag_multi_class,num_class)
        yield (img,label)

def testGenerator(test_path,target_size=(512,512),flag_multi_class=True,as_gray=False):
    filenames = os.listdir(test_path)
    for filename in filenames:
        img = io.imread(os.path.join(test_path,filename),as_gray=as_gray) #先将图片读出灰度图
        # img = img/255.
        img = trans.resize(img,target_size,mode = 'constant')
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img #img通过reshape成（512,512,1）
        img = np.reshape(img,(1,)+img.shape)  #img通过reshape成（1,512,512）这里的1可以理解为batchsize
        yield img


def saveResult(save_path,npyfile,flag_multi_class=True):
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = item
            img_out = np.zeros(img[:, :, 0].shape + (3,))   #将（512，512,1）变成（512，512,3）生成全零矩阵
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    index_of_class = np.argmax(img[row, col])
                    img_out[row, col] = COLOR_DICT[index_of_class]
            img = img_out.astype(np.uint8)
            io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
        else:
            img = item[:, :, 0]
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            img = img * 255.
            io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
