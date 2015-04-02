function roidb = roidb_from_imagenet10k_cls(imdb)
% create imdb from ImageNET CLS 10K dataset
% using whole image as bounding box
% the resulting class indices are 1~10447
% ---------------------------------------------------------

% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

cache_file = ['./imdb/cache/roidb_' imdb.name];
try
  load(cache_file);
catch  
  roidb.name = imdb.name;
  wnid2label_map = ...
    containers.Map({imdb.details.meta_det.synsets.WNID}, ...
    1:length(imdb.details.meta_det.synsets));
  
  num_classes = length(imdb.details.meta_det.synsets);
  roidb.details.wnid2label_map = wnid2label_map;

  for i = 1:length(imdb.image_ids)
    tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
    
    % get class index
    WNID = get_wnid(imdb.image_ids{i});
    class_id = wnid2label_map(WNID);
    
    % size format: [height, width]
    im_size = imdb.sizes(i, :);

    roidb.rois(i) = attach_proposals_whole_im(class_id, im_size, num_classes);
  end

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function rec = attach_proposals_whole_im(class_id, im_size, num_classes)
% ------------------------------------------------------------------------

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]

% size format: [height, width]
% box format: [x1 y1 x2 y2], 1-indexed
h = im_size(1);
w = im_size(2);
box_whole_im = [1 1 w h];

rec.gt = true;
rec.overlap = zeros(1, num_classes, 'single');
rec.overlap(1, num_classes) = 1;
rec.overlap = sparse(rec.overlap);
rec.boxes = single(box_whole_im);
rec.feat = [];
rec.class = uint8(class_id);


% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
