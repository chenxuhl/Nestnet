# -*- coding:utf-8 -*-
# Author : Ray
# Data : 2019/7/25 2:15 PM


from data import *
from model import *
import warnings

warnings.filterwarnings('ignore')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

aug_args = dict(
    rotation_range = 0.2,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    shear_range = 0.05,
    zoom_range = 0.05,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest'
)



#生成训练数据，返回迭代器
train_gene = trainGenerator(batch_size=2,aug_dict=aug_args,train_path='data/train/',
                        image_folder='train_img',label_folder='train_label',
                        image_color_mode='rgb',label_color_mode='rgb',
                        image_save_prefix='image',label_save_prefix='label',
                        flag_multi_class=True,num_class=4,save_to_dir=None
                        )
val_gene = valGenerator(batch_size=2,aug_dict=aug_args,val_path='data/train/',
                       image_folder='val_img',label_folder='val_label',
                       image_color_mode='rgb',label_color_mode='rgb',
                       image_save_prefix='image',label_save_prefix='label',
                       flag_multi_class=True,num_class=4,save_to_dir=None
                       )

tensorboard = TensorBoard(log_dir='./log')


# model, loss_function = unet(num_class=4)
model, loss_function = nestnet(num_class=4,dropout_rate = 0.5)

model.compile(optimizer=Adam(lr = 8e-4),loss=loss_function,metrics=['accuracy'])
model.summary()




model_checkpoint = ModelCheckpoint('welding_nestnet_v1_4.hdf5',monitor='val_loss',verbose=1,save_best_only=True)

history = model.fit_generator(train_gene,
                              steps_per_epoch=63,
                              epochs=10,
                              verbose=1,
                              callbacks=[model_checkpoint,tensorboard],
                              validation_data=val_gene,
                              validation_steps=1   #validation/batchsize
                              )


