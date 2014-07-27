function rcnn_make_spp_cache(imdb, out_dir, net_proto_file, net_binary_file, save_as_mat, gpu_id)
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

load rcnn_model_spp.mat;
roidb = imdb.roidb_func(imdb);

% initialize caffe
caffe('init', net_proto_file, net_binary_file);
caffe('set_phase_test');
caffe('set_mode_gpu');
caffe('set_device', gpu_id);

for i = 1:length(imdb.image_ids)
  th0 = tic();
  fprintf('------------------------------------------------------------\n');
  fprintf('caching feature: %d/%d\n', i, length(imdb.image_ids));
  
  % extract features
  th1 = tic();
  img_path = imdb.image_at(i);
  roi = roidb.rois(i);
  roi.boxes = roi.boxes(1, :); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  num_boxes = size(roi.boxes, 1);
  im = imread(img_path);
  % the roi.boxes are [x1 y1 x2 y2], 1-indexed
  feat = spp_features_5_scale(im, roi.boxes, rcnn_model_spp);
  fprintf('\n[Extracting feature: %f]\n', toc(th1));
  
  % store features to disk
  th2 = tic();
  im_id = i - 1;
  mkdir(fullfile(out_dir, num2str(im_id)));
  for j = 1:num_boxes
    bbox = roi.boxes(j,:)-1; % bbox is [x1 y1 x2 y2], 0-indexed
    % file name: $(im_id)/$x1_$y1_$x2_$y2.spp5feat
    if save_as_mat
      file_name = sprintf('%d_%d_%d_%d.mat', bbox(1), bbox(2), bbox(3), bbox(4));
      file_path = fullfile(out_dir, num2str(im_id), file_name);
      spp5 = feat(j, :)';
      save(file_path, 'spp5');
    else
      file_name = sprintf('%d_%d_%d_%d.spp5feat', bbox(1), bbox(2), bbox(3), bbox(4));
      file_path = fullfile(out_dir, num2str(im_id), file_name);
      fid = fopen(file_path, 'w');
      fwrite(fid, feat(j, :)', 'single');
      fclose(fid);
    end
  end
  fprintf('[Saving feature: %f]\n', toc(th2));
  fprintf('[Total: %f]\n', toc(th0));
end
