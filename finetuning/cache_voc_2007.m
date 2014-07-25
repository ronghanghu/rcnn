net_proto_file = 'external/caffe/examples/spp-rcnn-feat-cache/spp_rcnn_output_spp5.prototxt';
net_binary_file = 'external/caffe/examples/spp-rcnn-feat-cache/spp_rcnn_output_spp5.bin';
gpu_id = 1;

imdb_trainval = imdb_from_voc('datasets/VOCdevkit2007', 'trainval', '2007');
output_dir = '/x/ronghang/voc2007_training';

rcnn_make_spp_cache(imdb_trainval, output_dir, net_proto_file, net_binary_file, gpu_id);