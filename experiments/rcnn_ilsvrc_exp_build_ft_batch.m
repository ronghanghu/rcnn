function rcnn_ilsvrc_exp_build_ft_batch()

error('this function is out of date. it needs to be modified according to corresponding pascal version');

% -------------------- CONFIG --------------------
net_proto_file = './model-defs/spp_output_pool5.prototxt';
net_binary_file = './data/caffe_nets/spp_zf_iter_315000';
cache_name   = 'v1_finetune_val1+train1k_iter_50k';
max_train_pos_num = 1000;
split_num = 1000;

% SPPWindowDataLayer
spp_feat_cache_param.feat_dim = 12800;
spp_feat_cache_param.batch_per_file = 20;
spp_feat_cache_param.batch_size = 128;
spp_feat_cache_param.fg_fraction = 0.25;
spp_feat_cache_param.fg_overlap_max = 1.05; % larger than 1
spp_feat_cache_param.fg_overlap_min = 0.5;
spp_feat_cache_param.bg_overlap_max = 0.5;
spp_feat_cache_param.bg_overlap_min = 0.1;
spp_feat_cache_param.extension = '.multisize_feat_cache';
% ------------------------------------------------

devkit = './datasets/ILSVRC13';

% cache train1k_val1
spp_feat_cache_param.cache_dir = '/x/ronghang/ilsvrc13/train1k_val1';
imdb_cell_train1k_val1 = cell(201, 1);
inds_to_sample_cell_train1k_val1 = cell(201, 1);
% add train_pos_1 to train_pos_200
for i = 1:200
    fprintf('loading training imdb %d/%d\n', i, 200);
    imdb_train = imdb_from_ilsvrc13(devkit, ['train_pos_' num2str(i)]);
    inds = find(imdb_train.is_blacklisted == false);
    num = min(length(inds), max_train_pos_num);
    inds_to_sample = inds(1:num);
    imdb_cell_train1k_val1{i} = imdb_train;
    inds_to_sample_cell_train1k_val1{i} = inds_to_sample;
end
% add val1
imdb_val1 = imdb_from_ilsvrc13(devkit, 'val1');
imdb_cell_train1k_val1{201} = imdb_val1;
inds_to_sample_cell_train1k_val1{201} = find(imdb_val1.is_blacklisted == false);
rcnn_ilsvrc_make_spp_cache(imdb_cell_train1k_val1, ...
    inds_to_sample_cell_train1k_val1, split_num, ...
    net_proto_file, net_binary_file, cache_name, spp_feat_cache_param);

% cache val2
spp_feat_cache_param.cache_dir = '/x/ronghang/ilsvrc13/val2';
imdb_val2 = imdb_from_ilsvrc13(devkit, 'val2');
inds_to_sample = find(imdb_val2.is_blacklisted == false);
rcnn_ilsvrc_make_spp_cache({imdb_val2}, {inds_to_sample}, split_num, ...
    net_proto_file, net_binary_file, cache_name, spp_feat_cache_param);
