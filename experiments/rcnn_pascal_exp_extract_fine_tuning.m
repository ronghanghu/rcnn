function rcnn_pascal_exp_extract_fine_tuning()

% -------------------- CONFIG --------------------
net_proto_file = './model-defs/spp_output_pool5.prototxt';
net_binary_file = './data/caffe_nets/spp_zf_iter_315000';
cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';

% SPPWindowDataLayer
spp_feat_cache_param.feat_dim = 12800;
spp_feat_cache_param.batch_per_file = 20;
spp_feat_cache_param.batch_size = 128;
spp_feat_cache_param.fg_fraction = 0.25;
spp_feat_cache_param.fg_overlap_max = 1.05; % larger than 1
spp_feat_cache_param.fg_overlap_min = 0.5;
spp_feat_cache_param.bg_overlap_max = 0.5;
spp_feat_cache_param.bg_overlap_min = 0.1;
spp_feat_cache_param.extension = '.feat_cache';
% ------------------------------------------------

% make imdb
imdb_trainval = imdb_from_voc('datasets/VOCdevkit2007', 'trainval', '2007');
imdb_test = imdb_from_voc('datasets/VOCdevkit2007', 'test', '2007');

% cache trainval
spp_feat_cache_param.cache_dir = '/x/ronghang/voc2007/trainval';
rcnn_pascal_make_spp_cache(imdb_trainval, ...
    net_proto_file, net_binary_file, cache_name, spp_feat_cache_param);

% cache test
spp_feat_cache_param.cache_dir = '/x/ronghang/voc2007/test';
rcnn_pascal_make_spp_cache(imdb_test, ...
    net_proto_file, net_binary_file, cache_name, spp_feat_cache_param);