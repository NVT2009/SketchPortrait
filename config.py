from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Config:
    # counts gpu
    num_gpu = 1
    lr = 1e-5
    train_batch_size = 1
    eposch_size =1000
    optimizer = 'adam'

    # export CUDA_VISIBLE_DEVICES=1
    gpu_ids = '0'
    nr_gpus = 1

    keep_checkpoint_every_n_hours = 1
    data_shape = 128

    # --------------        TRAINING     ---------------#
    path_data = r'E:\Project\Sketch_Portrait\Data\SketchTwo'
    train_cvs = r'E:\Programming Language\Python\SketchPortrait\data.cvs'




    #---------------------------------------------------#


