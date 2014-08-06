function rcnn_make_spp_cache(imdb, out_dir, net_proto_file, net_binary_file, ...
    spp_window_data_param)
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

spp5_dim = spp_window_data_param.spp5_dim;
batch_per_file = spp_window_data_param.batch_per_file;
batch_size = spp_window_data_param.batch_size;
fg_fraction = spp_window_data_param.fg_fraction;
fg_overlap_max = spp_window_data_param.fg_overlap_max; % large than 1
fg_overlap_min = spp_window_data_param.fg_overlap_min;
bg_overlap_max = spp_window_data_param.bg_overlap_max;
bg_overlap_min = spp_window_data_param.bg_overlap_min;

load rcnn_model_spp.mat;
roidb = imdb.roidb_func(imdb);
if ~exist(out_dir, 'dir')
  mkdir(out_dir);
end

% initialize caffe
caffe('init', net_proto_file, net_binary_file);
caffe('set_phase_test');

fg_per_batch = round(batch_size * fg_fraction);
bg_per_batch = batch_size - fg_per_batch;
fg_limit = fg_per_batch * batch_per_file;
bg_limit = bg_per_batch * batch_per_file;
fg_cache = [];
bg_cache = [];

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
  fprintf('\n[Extracting feature: %f]\n', toc(th1));
  
  % put features into fg cache and bg cache
  is_fg = (roi.overlap >= fg_overlap_min) & (roi.overlap <= fg_overlap_max);
  is_bg = (roi.overlap >  bg_overlap_min) & (roi.overlap <  bg_overlap_max);
  fg_cache = cat(1, fg_cache, feat(is_fg, :));
  bg_cache = cat(1, bg_cache, feat(is_bg, :));
  
  % store features to disk
  if size(fg_cache, 1) >= fg_limit
    th2 = tic();
    fg_num = size(fg_cache, 1);
    bg_num = size(bg_cache, 1);
    fprintf('------------------------------------------------------------\n');
    fprintf('saving file id %d', file_id);
    fprintf('\t#fg in cache: %d\n', fg_num);
    fprintf('\t#bg in cache: %d\n', bg_num);
    % assume that background windows are much more than foreground windows
    assert(bg_num >= bg_limit);
    % subsample background windows
    bg_keep = sort(randsample(1:bg_num, bg_limit));
    bg_cache = bg_cache(bg_keep, :);
    
    % 12800-D spp5 feature is the fastest dimension
    feat_batches = zeros(spp5_dim, batch_size, batch_per_file, 'single');
    for b = 1:batch_per_file
      fg_cache_index = ((b-1)*fg_per_batch+1):(b*fg_per_batch);
      bg_cache_index = ((b-1)*bg_per_batch+1):(b*bg_per_batch);
      feat_batches(:, 1:fg_per_batch, b) = ...
          single(permute(fg_cache(fg_cache_index, :), [2, 1]));
      feat_batches(:, (fg_per_batch+1):end, b) = ...
          single(permute(bg_cache(bg_cache_index, :), [2, 1]));
    end
    % write batches to disk
    file_name = sprintf('%d.spp_cache', file_id);
    file_path = fullfile(out_dir, file_name);
    fid = fopen(file_path, 'w');
    fwrite(fid, feat_batches, 'single');
    fclose(fid);
    
    % prune caches
    fg_cache = fg_cache((fg_limit+1):end, :);
    bg_cache = [];
    file_id = file_id + 1;
    fprintf('[Saving feature: %f]\n', toc(th2));
  end

end
