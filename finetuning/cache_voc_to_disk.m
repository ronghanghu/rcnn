net_proto_file = './model-defs/spp_output_pool5.prototxt';
net_binary_file = './data/caffe_nets/spp_zf_iter_315000';
save_as_mat = false;
gpu_id = 0;

% make imdb
imdb_trainval = imdb_from_voc('datasets/VOCdevkit2007', 'trainval', '2007');
imdb_test = imdb_from_voc('datasets/VOCdevkit2007', 'test', '2007');

% make window data file
rcnn_make_window_file(imdb_trainval, 'external/caffe/examples/pascal-finetuning-spp');
rcnn_make_window_file(imdb_test, 'external/caffe/examples/pascal-finetuning-spp');

% cache trainval
output_dir = '/x/ronghang/voc2007/trainval';
rcnn_make_spp_cache(imdb_trainval, output_dir, net_proto_file, net_binary_file, save_as_mat, gpu_id);

% cache test
output_dir = '/x/ronghang/voc2007/test';
rcnn_make_spp_cache(imdb_test, output_dir, net_proto_file, net_binary_file, save_as_mat, gpu_id);
