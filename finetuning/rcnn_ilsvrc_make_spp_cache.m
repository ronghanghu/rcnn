function rcnn_ilsvrc_make_spp_cache(imdb_cell, inds_to_sample_cell, ...
  split_num, ...
  net_proto_file, net_binary_file, cache_name, spp_feat_cache_param)
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

% load roidb for each imdb and
% calculate how many images in each imdb to be put into batches each time
assert(iscell(imdb_cell) && iscell(inds_to_sample_cell));
imdb_num = length(imdb_cell);
assert(imdb_num == length(inds_to_sample_cell));
roidb_cell = cell(size(imdb_cell));
imdb_interval_cell = cell(size(imdb_cell));
for n = 1:imdb_num
  fprintf('loading roidb %d/%d\n', n, imdb_num);
  roidb_cell{n} = imdb_cell{n}.roidb_func(imdb_cell{n});
  image_num = length(inds_to_sample_cell{n});
  imdb_interval_cell{n} = round(linspace(0, image_num, split_num+1));
end

if ~exist(cache_dir, 'dir')
  mkdir(cache_dir);
end

% fix seed for repeatability
seed_rand();

% initialize caffe
rcnn_model = rcnn_create_model(net_proto_file, net_binary_file);
rcnn_model = rcnn_load_model(rcnn_model);

fg_per_batch = round(batch_size * fg_fraction);
bg_per_batch = batch_size - fg_per_batch;
fg_cache = [];
bg_cache = [];
fg_label_cache = [];

file_id = 0;
for s = 1:split_num
  for n = 1:imdb_num
    ii_first = imdb_interval_cell{n}(s) + 1;
    ii_last = imdb_interval_cell{n}(s+1);
    if ii_first > ii_last
      continue
    end
    imdb = imdb_cell{n};
    roidb = roidb_cell{n};
    for ii = ii_first:ii_last
      fprintf('------------------------------------------------------------\n');
      fprintf('caching feature: split %d/%d\n\timdb %s %d/%d\n', s, ...
          split_num, imdb.name, ii, length(inds_to_sample_cell{n}));
      i = inds_to_sample_cell{n}(ii);
      % extract features
      th1 = tic();
      save_file = ['./feat_cache/' cache_name '/' imdb.name '/' ...
          imdb.image_ids{i} '.mat'];
      roi = roidb.rois(i);
      if exist(save_file, 'file')
        fprintf('loading existing feature from mat file on feat cache\n');
        d = load(save_file);
        feat = d.feat;
        assert(size(feat, 1) == size(roi.boxes, 1));
      else
        fprintf('extracting feature from image\n');
        img_path = imdb.image_at(i);
        im = imread(img_path);
        % the roi.boxes are [x1 y1 x2 y2], 1-indexed
        feat = spp_features(im, roi.boxes, rcnn_model);
      end
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
      has_residue_at_end = ...
        (s == split_num) && (n == imdb_num) && (ii == ii_last) ...
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
        if ~has_residue_at_end
          actual_batch_num = min(actual_batch_num, batch_per_file);
        end
        if actual_batch_num ~= batch_per_file
          fprintf('Notice: there are %d batches in this file\n', ...
              actual_batch_num);
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
        fg_label_cache = ...
            fg_label_cache((fg_per_batch * actual_batch_num+1):end);
        file_id = file_id + 1;
        fprintf('[Saving feature: %f]\n', toc(th2));
      end
    end
  end
end