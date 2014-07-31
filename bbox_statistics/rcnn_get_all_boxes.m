function [all_boxes, all_overlaps] = rcnn_get_all_boxes(imdb)
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

roidb = imdb.roidb_func(imdb);

all_boxes = [];
all_overlaps = [];

for i = 1:length(imdb.image_ids)
  tic_toc_print('running statistics: %d/%d\n', i, length(imdb.image_ids));
  roi = roidb.rois(i);
  all_boxes = cat(1, all_boxes, roi.boxes);
  all_overlaps = cat(1, all_overlaps, max(roi.overlap, [], 2));  
end
