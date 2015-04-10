% fix roidb errors:
%   remove all boxes that are outside the image

function roidb = roidb_fix_errors(roidb)
  % check every image and remove those bboxes outside the image
  wrong_bbox_count = 0;
  for i = 1:length(imdb.image_ids)
    % size format: [height, width]
    im_size = imdb.sizes(i, :);
    h = im_size(1);
    w = im_size(2);
    % box format: [x1 y1 x2 y2], 1-indexed
    
    roi = roidb.rois(i);
    boxes = roi.boxes;
    % x1 < x2, y1 < y2, x2 <= w, y2 <= h
    valid = (boxes(:, 1) < boxes(:, 3)) & (boxes(:, 2) < boxes(:, 4)) ...
      & (boxes(:, 3) <= w) & (boxes(:, 4) <= h);
    wrong_bbox_count = wrong_bbox_count + sum(~valid);
    
    roi.gt = roi.gt(valid, :);
    roi.is_difficult = roi.is_difficult(valid, :);
    roi.overlap = roi.overlap(valid, :);
    roi.boxes = roi.boxes(valid, :);
    if ~isempty(roi.feat)
      roi.feat = roi.feat(valid, :);
    end
    roi.class = roi.class(valid, :);
  end
  fprintf('removed %d wrong boxes in roidb %s\n', wrong_bbox_count, imdb.name);
end