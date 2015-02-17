load 10k_training/classes_7k.mat;

net_model = '/home/ronghang/workspace/mhex_verify/imagenet_10k_det/exp_mhex_10k_leafonly_det_whole.prototxt';
net_weights = '/home/ronghang/workspace/mhex_verify/imagenet_10k_det/caffemodel/exp_mhex_10k_leafonly_det_whole.caffemodel';

rcnn_model_10k = rcnn_create_model(net_model, net_weights);
rcnn_model_10k = rcnn_load_model(rcnn_model_10k, true);
rcnn_model_10k.training_opts.feat_norm_mean = 20;
rcnn_model_10k.classes = classes_7k;

rcnn_model_10k.detectors.W = [-ones(1, 7404); eye(7404)];
rcnn_model_10k.detectors.B = zeros(1, 7404);
