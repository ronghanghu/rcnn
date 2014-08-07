function rcnn_make_spp_cache(imdb, net_proto_file, net_binary_file, ...
    spp_feat_cache_param)
% rcnn_make_window_file(imdb, out_dir)
%   Makes a window file that can be used by the caffe WindowDataLayer
%   for finetuning.
%
%   The window file format contains repeated blocks of:
%
%     # image_index
%     img_path
%     channels
%     height
%     width
%     num_windows
%     class_index overlap x1 y1 x2 y2
%     <... num_windows-1 more windows follow ...>

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

feat_dim = spp_feat_cache_param.feat_dim;
batch_per_file = spp_feat_cache_param.batch_per_file;
batch_size = spp_feat_cache_param.batch_size;
fg_fraction = spp_feat_cache_param.fg_fraction;
fg_overlap_max = spp_feat_cache_param.fg_overlap_max; % large than 1
fg_overlap_min = spp_feat_cache_param.fg_overlap_min;
bg_overlap_max = spp_feat_cache_param.bg_overlap_max;
bg_overlap_min = spp_feat_cache_param.bg_overlap_min;
extension = spp_feat_cache_param.extension;
cache_dir = spp_feat_cache_param.cache_dir;

load rcnn_model_spp.mat;
roidb = imdb.roidb_func(imdb);
if ~exist(cache_dir, 'dir')
  mkdir(cache_dir);
end

% initialize caffe
caffe('init', net_proto_file, net_binary_file);
caffe('set_phase_test');

fg_per_batch = round(batch_size * fg_fraction);
bg_per_batch = batch_size - fg_per_batch;
fg_cache = [];
bg_cache = [];
fg_label_cache = [];

file_id = 0;
for i = 1:length(imdb.image_ids)
  fprintf('------------------------------------------------------------\n');
  fprintf('caching feature: %d/%d\n', i, length(imdb.image_ids));
  
  % extract features
  th1 = tic();
  img_path = imdb.image_at(i);
  roi = roidb.rois(i);
  im = imread(img_path);
  % the roi.boxes are [x1 y1 x2 y2], 1-indexed
  feat = spp_features(im, roi.boxes, rcnn_model_spp);
  fprintf('[Extracting feature: %f]\n', toc(th1));
  
  % put features into fg cache and bg cache
  [gt_overlap, label] = max(roi.overlap, [], 2);
  is_fg = (gt_overlap >= fg_overlap_min) & (gt_overlap <= fg_overlap_max);
  is_bg = (gt_overlap >  bg_overlap_min) & (gt_overlap <  bg_overlap_max);
  fg_cache = cat(1, fg_cache, feat(is_fg, :));
  bg_cache = cat(1, bg_cache, feat(is_bg, :));
  fg_label_cache = cat(1, fg_label_cache, label(is_fg));
  
  % store features to disk, when
  % case 1: the number of feature in a cache hits limit
  % case 2: at the end there are still residue enough to build at least 1
  % batch
  fg_num = size(fg_cache, 1);
  bg_num = size(bg_cache, 1);
  has_residue_at_end = (i == length(imdb.image_ids)) ...
      && (fg_num >= fg_per_batch) ...
      && (bg_num >= bg_per_batch); 
  hits_limit = (fg_num >= fg_per_batch * batch_per_file) ...
      && (bg_num >= bg_per_batch * batch_per_file);
  if has_residue_at_end || hits_limit
    th2 = tic();
    fprintf('------------------------------------------------------------\n');
    fprintf('saving file %d%s\n', file_id, extension);
    fprintf('\t#fg in cache: %d\n', fg_num);
    fprintf('\t#bg in cache: %d\n', bg_num);
    
    % calculate the actual number of batches in this file
    actual_batch_num = ...
        min(floor(fg_num / fg_per_batch), floor(bg_num / bg_per_batch));
    if i < length(imdb.image_ids)
      actual_batch_num = min(actual_batch_num, batch_per_file);
    end
    if actual_batch_num ~= batch_per_file
      fprintf('Notice: there are %d batches in this file\n', actual_batch_num);
    end

    % subsample background windows
    if bg_num > bg_per_batch * actual_batch_num
      bg_keep = randsample(1:bg_num, bg_per_batch * actual_batch_num);
      bg_cache = bg_cache(bg_keep, :);
    end

    % write batches to disk
    file_name = sprintf('%d%s', file_id, extension);
    file_path = fullfile(cache_dir, file_name);
    fid = fopen(file_path, 'w');
    fwrite(fid, [actual_batch_num, batch_size, feat_dim], 'single');
    for b = 1:actual_batch_num
      fg_cache_index = ((b-1)*fg_per_batch+1):(b*fg_per_batch);
      bg_cache_index = ((b-1)*bg_per_batch+1):(b*bg_per_batch);
      % write labels (bg labels are zeros)
      feat_labels = zeros(batch_size, 1, 'single');
      feat_labels(1:fg_per_batch) = fg_label_cache(fg_cache_index);
      fwrite(fid, feat_labels, 'single');

      % write feature, 12800-D spp5 feature is the fastest dimension
      feat_batch = zeros(feat_dim, batch_size, 'single');
      feat_batch(:, 1:fg_per_batch) = ...
        single(permute(fg_cache(fg_cache_index, :), [2, 1]));
      feat_batch(:, (fg_per_batch+1):end) = ...
        single(permute(bg_cache(bg_cache_index, :), [2, 1]));
      fwrite(fid, feat_batch, 'single');
    end
    fclose(fid);

    % prune caches
    fg_cache = fg_cache((fg_per_batch * actual_batch_num+1):end, :);
    bg_cache = [];
    file_id = file_id + 1;
    fprintf('[Saving feature: %f]\n', toc(th2));
  end

end
