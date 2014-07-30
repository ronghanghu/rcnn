net_proto_file = './model-defs/spp_zf_output_imagelevel_fc7.prototxt';
net_binary_file = './data/caffe_nets/spp_zf_iter_315000';
save_as_mat = false;
gpu_id = 0;

% cache trainval
imdb_trainval = imdb_from_voc('datasets/VOCdevkit2007', 'trainval', '2007');
output_dir = '/x/ronghang/voc2007/trainval';
rcnn_make_spp_cache(imdb_trainval, output_dir, net_proto_file, net_binary_file, save_as_mat, gpu_id);

% cache test
% imdb_test = imdb_from_voc('datasets/VOCdevkit2007', 'test', '2007');
% output_dir = '/x/ronghang/voc2007/test';
% rcnn_make_spp_cache(imdb_test, output_dir, net_proto_file, net_binary_file, save_as_mat, gpu_id);
