net_proto_file = './model-defs/spp_rcnn_output_spp5.prototxt';
net_binary_file = './data/caffe_nets/finetune_voc_2007_spp_trainval_iter_20000';
save_as_mat = false;
gpu_id = 1;

% cache trainval
imdb_trainval = imdb_from_voc('datasets/VOCdevkit2007', 'trainval', '2007');
output_dir = '/x/ronghang/voc2007/trainval_mat';
% rcnn_make_window_file(imdb_trainval, 'external/caffe/examples/pascal-finetuning');
rcnn_make_spp_cache(imdb_trainval, output_dir, net_proto_file, net_binary_file, save_as_mat, gpu_id);

% cache test
% imdb_test = imdb_from_voc('datasets/VOCdevkit2007', 'test', '2007');
% output_dir = '/x/ronghang/voc2007/test_mat';
% rcnn_make_window_file(imdb_test, 'external/caffe/examples/pascal-finetuning');
% rcnn_make_spp_cache(imdb_test, output_dir, net_proto_file, net_binary_file, save_as_mat, gpu_id);
