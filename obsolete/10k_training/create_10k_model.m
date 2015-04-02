meta_7k = load('external/mhex_graph/+imagenet/meta_7k.mat');
synsets_7k = meta_7k.synsets_7k;
classes_7k = cell(7404, 1);
for v = 1:7404
  words_split = strsplit(synsets_7k(v).words, ',');
  classes_7k{v} = words_split{1};
end

net_model = '/home/ronghang/workspace/mhex_verify/imagenet_10k_det/exp_mhex_10k_leafonly_det_whole.prototxt';
net_weights = '/home/ronghang/workspace/mhex_verify/imagenet_10k_det/caffemodel/exp_mhex_10k_leafonly_det_whole.caffemodel';

rcnn_model = rcnn_create_model(net_model, net_weights);
rcnn_model = rcnn_load_model(rcnn_model, true);
rcnn_model.training_opts.feat_norm_mean = 20;
rcnn_model.classes = classes_7k;

rcnn_model.detectors.W = [-ones(1, 7404); eye(7404)];
rcnn_model.detectors.B = zeros(1, 7404);