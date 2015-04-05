function roidb = roidb_from_ilsvrc12_loc(imdb)
% create roidb from ILSVRC LOC 1K dataset
% using bounding box annotation
% the resulting class indices are 1~1000
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
  addpath(fullfile(imdb.details.devkit_path, 'evaluation')); 

  roidb.name = imdb.name;
  roidb.details.wnid2label_map = ...
    containers.Map({imdb.details.meta_det.synsets.WNID}, ...
    1:length(imdb.details.meta_det.synsets));
  
  is_train = strcmp(imdb.name, 'ilsvrc12_loc_train');
  regions.boxes = cell(length(imdb.image_ids), 1);

  hash = make_hash(imdb.details.meta_det.synsets);

  for i = 1:length(imdb.image_ids)
    tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
    if is_train
      WNID = get_wnid(imdb.image_ids{i});
      anno_file = fullfile(imdb.details.bbox_path, ...
          WNID, [imdb.image_ids{i} '.xml']);
    else
      anno_file = fullfile(imdb.details.bbox_path, ...
          [imdb.image_ids{i} '.xml']);
    end

    try
      rec = VOCreadrecxml(anno_file, hash);
    catch
      rec = [];
    end
    roidb.rois(i) = attach_proposals(rec, regions.boxes{i}, WNID, roidb.details.wnid2label_map, anno_file);
  end

  rmpath(fullfile(imdb.details.devkit_path, 'evaluation')); 

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function rec = attach_proposals(ilsvrc_rec, boxes, WNID, wnid2label_map, anno_file)
% ------------------------------------------------------------------------

num_classes = 1000;
assert(isempty(boxes));

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
if ~isempty(boxes)
  boxes = boxes(:, [2 1 4 3]);
end

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(ilsvrc_rec, 'objects') && length(ilsvrc_rec.objects) > 0
  gt_boxes = cat(1, ilsvrc_rec.objects(:).bbox);
  all_boxes = cat(1, gt_boxes, boxes);
  gt_classes = cat(1, ilsvrc_rec.objects(:).label);
  num_gt_boxes = size(gt_boxes, 1);
  
  if isempty(gt_classes)
    % just assign image-level labels to each object box
    try
      gt_classes = zeros(num_gt_boxes, 1);
      label = wnid2label_map(WNID);
      assert(1 <= label && label <= num_classes);
      gt_classes(:) = label;
    catch
      fprintf('serious issue occurred within image %s, please check\n', anno_file);
      keyboard
    end
    % rec.gt = [];
    % rec.is_difficult = [];
    % rec.overlap = [];
    % rec.boxes = [];
    % rec.feat = [];
    % rec.class = [];
    % return
  end
else
  gt_boxes = [];
  all_boxes = boxes;
  gt_classes = [];
  num_gt_boxes = 0;
end
num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.is_difficult = false(num_gt_boxes + num_boxes, 1);
rec.overlap = zeros(num_gt_boxes+num_boxes, num_classes, 'single');
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.overlap = sparse(rec.overlap);
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));


% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
