function feat = spp_multisize_features(im, boxes, rcnn_model)
%   Compute Spatial Pyramid Pooling features on a set of boxes.
%
%   im is an image in RGB order as returned by imread
%   boxes are in [x1 y1 x2 y2] format with one box per row
%   rcnn_model specifies the CNN Caffe net file to use.

% local scales
window_sizes = [1 2];
window_size_num = length(window_sizes);

% concatenate all boxes together
% multiscale_boxes(1:end-1, :) are local boxes
% multiscale_boxes(end, :) is global box
multiscale_boxes = [];

% get multiscale boxes
box_num = size(boxes, 1);
box_centers = (boxes(:, [1 2 1 2]) + boxes(:, [3 4 3 4])) / 2;
for s = 1:window_size_num
  scaled_boxes = (boxes - box_centers) * window_sizes(s) + box_centers;
  multiscale_boxes = cat(1, multiscale_boxes, scaled_boxes);
end

% get the box of the whole image
[h, w, ~] = size(im);
box_whole_image = [1 1 w h];
multiscale_boxes = cat(1, multiscale_boxes, box_whole_image);

% remove these constrains and allow boxes to expand outside image region
% percise localization of box center is more important
% % make sure the boxes fit into the image region
% multiscale_boxes(:, 1) = max(1, multiscale_boxes(:, 1));
% multiscale_boxes(:, 2) = max(1, multiscale_boxes(:, 2));
% multiscale_boxes(:, 3) = min(w, multiscale_boxes(:, 3));
% multiscale_boxes(:, 4) = min(h, multiscale_boxes(:, 4));

% use spp_features to extract raw features
raw_feat = spp_features(im, multiscale_boxes, rcnn_model);

% get global feature for the whole image
global_feat = repmat(raw_feat(end, :), [box_num 1]);

% get local feature for each box
raw_local_feat = raw_feat(1:end-1, :);
assert(size(raw_local_feat, 1) == window_size_num * box_num);
local_feat = [];
for s = 1:window_size_num
  ind = ((s-1)*box_num+1):(s*box_num);
  local_feat = cat(2, local_feat, raw_local_feat(ind, :));
end

feat = cat(2, global_feat, local_feat);

end
