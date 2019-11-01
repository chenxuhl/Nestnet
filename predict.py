# - * - coding: utf-8 - * -
# Author: JoeyChen
# Data: 2019/11/01


from data import *
from model import *
import warnings


warnings.filterwarnings('ignore')

# model, _ = unet(num_class=4)

model, _ = nestnet(num_class=4)

model.load_weights('welding_nestnet_v1_4.hdf5')
test_gene = testGenerator(test_path='data/test/img')
results = model.predict_generator(test_gene,30,verbose=1)
saveResult('data/test/pred_nn/',results)