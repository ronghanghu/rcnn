function show_det_boxes(im, dets, classes, score_thresh, inclass_nms_thresh, across_nms_thresh, color, save_file)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes, out)
%
%   If out is given, a pdf of the image is generated (requires export_fig).

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan
%
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

if ~exist('score_thresh', 'var')
  score_thresh = 0;
end
if ~exist('inclass_nms_thresh', 'var')
  inclass_nms_thresh = 0.3;
end
if ~exist('across_nms_thresh', 'var')
  across_nms_thresh = 1.1;
end
if ~exist('color', 'var')
  color = 'r';
end

all_bboxes = [];
all_bboxclass = [];
for i = 1:length(dets)
  above_thresh = (dets{i}(:, 5) >= score_thresh);
  boxes = dets{i}(above_thresh, :);
  keep = nms(boxes, inclass_nms_thresh);
  boxes = boxes(keep, :);
  det_num = sum(size(boxes, 1));
  all_bboxes = cat(1, all_bboxes, boxes);
  all_bboxclass = cat(1, all_bboxclass, i * ones(det_num, 1));
end

% across category nms
keep = nms(all_bboxes, across_nms_thresh);
all_bboxes = all_bboxes(keep, :);
all_bboxclass = all_bboxclass(keep);

if exist('save_file', 'var')
  h = figure('visible', 'off');
  cwidth = 1;
  image(im);
else
  h = figure;
  cwidth = 1;
  image(im);
end

axis image;
axis off;
set(h, 'Color', 'white');

if ~isempty(all_bboxes)
  numfilters = size(all_bboxes,1); %floor(size(boxes, 2)/4);
  
  % draw the boxes with the detection window on top (reverse order)
  for i = numfilters:-1:1
    x1 = all_bboxes(i,1);%boxes(:,1+(i-1)*4);
    y1 = all_bboxes(i,2);%boxes(:,2+(i-1)*4);
    x2 = all_bboxes(i,3);%boxes(:,3+(i-1)*4);
    y2 = all_bboxes(i,4);%boxes(:,4+(i-1)*4);
    % remove unused filters
    del = find(((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0)) == 1);
    x1(del) = [];
    x2(del) = [];
    y1(del) = [];
    y2(del) = [];
    c = color;
    s = '-';
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', c, 'linewidth', cwidth, 'linestyle', s);
    ss = regexp(classes{all_bboxclass(i)}, ',', 'split');
    x_t = double(max(5, x1-5)); y_t = double(min(y2+5, size(im, 1)-15));
    text(x_t,y_t,sprintf('%s: %2.1f', ss{1}, all_bboxes(i,5)),...
      'BackgroundColor', [0.7 0.9 0.7], 'FontSize', 20, 'Color', c);
  end
end

if exist('save_file', 'var')
  set(h, 'PaperPosition', [0, 0, 4, 3]);
  saveas(h, save_file);
end