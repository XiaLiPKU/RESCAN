import os
import logging

channel = 24
stage_num = 4
depth = 7 # >=3
use_se = True 
uint = 'RNN' # ['Conv', 'RNN', 'GRU', 'LSTM']
frame = 'Add' # ['Add', 'Full']

aug_data = False # Set as False for fair comparison

batch_size = 64
patch_size = 64
lr = 5e-3

data_dir = 'path/to/dataset/folder'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest')
save_steps = 400

num_workers = 8
num_GPU = 1
device_id = 3

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


