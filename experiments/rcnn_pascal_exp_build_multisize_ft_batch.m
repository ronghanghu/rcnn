function rcnn_pascal_exp_build_multisize_ft_batch()

% -------------------- CONFIG --------------------
net_proto_file = './model-defs/spp_output_pool5.prototxt';
net_binary_file = './data/caffe_nets/spp_zf_iter_315000';
cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
storage_dir = '/home/ronghang/workspace/external/voc2007_multisize';

feat_dim_inds_cell = {1:12800, 12801:25600, 25601:38400};
extension_cell = {'.feat_cache_IM', '.feat_cache_1X', '.feat_cache_2X'};

% SPPWindowDataLayer
spp_feat_cache_param.feat_dim = 12800;
spp_feat_cache_param.batch_per_file = 20;
spp_feat_cache_param.batch_size = 128;
spp_feat_cache_param.fg_fraction = 0.25;
spp_feat_cache_param.fg_overlap_max = 1.05; % larger than 1
spp_feat_cache_param.fg_overlap_min = 0.5;
spp_feat_cache_param.bg_overlap_max = 0.5;
spp_feat_cache_param.bg_overlap_min = 0.1;
% ------------------------------------------------

% make imdb
imdb_trainval = imdb_from_voc('datasets/VOCdevkit2007', 'trainval', '2007');
imdb_test = imdb_from_voc('datasets/VOCdevkit2007', 'test', '2007');

% make multisize cache
assert(length(feat_dim_inds_cell) == length(extension_cell));
size_num = length(feat_dim_inds_cell);
for size = 1:size_num
spp_feat_cache_param.feat_dim_inds = feat_dim_inds_cell{size};
spp_feat_cache_param.extension = extension_cell{size};

% cache trainval
spp_feat_cache_param.cache_dir = fullfile(storage_dir, 'trainval');
rcnn_pascal_make_spp_cache(imdb_trainval, ...
    net_proto_file, net_binary_file, cache_name, spp_feat_cache_param);

% cache test
spp_feat_cache_param.cache_dir = fullfile(storage_dir, 'test');
rcnn_pascal_make_spp_cache(imdb_test, ...
    net_proto_file, net_binary_file, cache_name, spp_feat_cache_param);
end