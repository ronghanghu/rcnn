"""
Make a pure Caffe R-CNN model by transplanting the SVMs into an FC layer.

n.b. the MODEL_MAT file must be made from the MATLAB model files and needs
to have the vars W, b, and feat_norm_mean.
"""
import numpy as np, scipy.io
import caffe

# the net containing fine-tuned fc layers
# OLD_PROTOTXT = '(The network proto in fine-tuning fc layers, remember to remove WindowDataLayer and LossLayer and Accuracy Layer)'
# MODIFY
OLD_PROTOTXT = 'pascal_finetune_trainval.prototxt'
# OLD_WEIGHTS = '(The network binary in fine-tuning fc layers, make sure it weights are fc or sppfc)'
# MODIFY
OLD_WEIGHTS = './finetune_voc_2007_spp_trainval_iter_300000'

# the net containing original conv1~conv5 layers
# MODIFY
NEW_PROTOTXT = '../../model-defs/spp_output_fc7.prototxt'
NEW_WEIGHTS = '../../data/caffe_nets/spp_zf_iter_315000'

# MODIFY
OUT_FILE = '../../data/caffe_nets/finetune_voc_2007_spp_zf_trainval_iter_300000_linked'

# load architecture for pure Caffe net and the fine-tuned model
old_net = caffe.Net(OLD_PROTOTXT, OLD_WEIGHTS)
new_net = caffe.Net(NEW_PROTOTXT, NEW_WEIGHTS)
old_net.set_mode_cpu()
new_net.set_mode_cpu()

# fc6, change name from sppfc6 to fc6
# MODIFY
new_net.params["fc6"][0].data[...] = old_net.params["fc6"][0].data[...]
new_net.params["fc6"][1].data[...] = old_net.params["fc6"][1].data[...]

# fc7, change name from sppfc7 to fc7
# MODIFY
new_net.params["fc7"][0].data[...] = old_net.params["fc7"][0].data[...]
new_net.params["fc7"][1].data[...] = old_net.params["fc7"][1].data[...]

# # fc8_pascal
# new_net.params["fc8_pascal"][0].data[...] = old_net.params["fc8_pascal"][0].data[...]
# new_net.params["fc8_pascal"][1].data[...] = old_net.params["fc8_pascal"][1].data[...]

# save
new_net.save(OUT_FILE)
