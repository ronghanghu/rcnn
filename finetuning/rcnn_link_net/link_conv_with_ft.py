"""
Make a pure Caffe R-CNN model by transplanting the SVMs into an FC layer.

n.b. the MODEL_MAT file must be made from the MATLAB model files and needs
to have the vars W, b, and feat_norm_mean.
"""
import numpy as np, scipy.io
import caffe

# the net containing fine-tuned fc layers
OLD_PROTOTXT = '(The network proto in fine-tuning fc layers, remember to remove WindowDataLayer and LossLayer and Accuracy Layer)'
OLD_WEIGHTS = '(The network binary in fine-tuning fc layers, make sure it weights are fc or sppfc)'

# the net containing original conv1~conv5 layers
NEW_PROTOTXT = 'spp_rcnn_output_fc8.prototxt'
NEW_WEIGHTS = '../spp-rcnn-feat-cache/spp_rcnn_output_spp5.bin'

OUT_FILE = 'finetune_voc_2007_spp_trainval_iter_20000'

# load architecture for pure Caffe net and the fine-tuned model
old_net = caffe.Net(OLD_PROTOTXT, OLD_WEIGHTS)
new_net = caffe.Net(NEW_PROTOTXT, NEW_WEIGHTS)
old_net.set_mode_cpu()
new_net.set_mode_cpu()

# fc6, change name from sppfc6 to fc6
new_net.params["fc6"][0].data[...] = old_net.params["sppfc6"][0].data[...]
new_net.params["fc6"][1].data[...] = old_net.params["sppfc6"][1].data[...]

# fc7, change name from sppfc7 to fc7
new_net.params["fc7"][0].data[...] = old_net.params["sppfc7"][0].data[...]
new_net.params["fc7"][1].data[...] = old_net.params["sppfc7"][1].data[...]

# fc8_pascal
new_net.params["fc8_pascal"][0].data[...] = old_net.params["fc8_pascal"][0].data[...]
new_net.params["fc8_pascal"][1].data[...] = old_net.params["fc8_pascal"][1].data[...]

# save
new_net.save(OUT_FILE)
