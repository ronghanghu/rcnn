net_proto_file = 'external/caffe/examples/spp-rcnn-feat-cache/spp_rcnn_output_spp5.prototxt';
net_binary_file = 'external/caffe/examples/spp-rcnn-feat-cache/spp_rcnn_output_spp5.bin';
gpu_id = 0;

% % cache trainval
% imdb_trainval = imdb_from_voc('datasets/VOCdevkit2007', 'trainval', '2007');
% output_dir = '/x/ronghang/voc2007/trainval';
% rcnn_make_window_file(imdb_trainval, 'external/caffe/examples/pascal-finetuning');
% rcnn_make_spp_cache(imdb_trainval, output_dir, net_proto_file, net_binary_file, gpu_id);

% cache test
imdb_test = imdb_from_voc('datasets/VOCdevkit2007', 'test', '2007');
output_dir = '/x/ronghang/voc2007/test';
% rcnn_make_window_file(imdb_test, 'external/caffe/examples/pascal-finetuning');
rcnn_make_spp_cache(imdb_test, output_dir, net_proto_file, net_binary_file, gpu_id);
