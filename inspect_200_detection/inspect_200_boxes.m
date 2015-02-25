% load imdb and roidb
imdb_name = 'ilsvrc13_val2';
[imdb, roidb] = load_imdb_roidb(imdb_name);

% ground-truth boxes
aboxes_gt = load_gt_from_roidb(imdb, roidb);

% load detection boxes
% bbox_dir = './cachedir/step_2v1_train_100_det_iter_50000.caffemodel';
% aboxes = load_dets_from_cache(bbox_dir, imdb);

% show and save detection boxes
boxes_set = aboxes_gt;
suffix = '_gt';
extension = '.png';
color = 'b';
save_dir = '~/Downloads';

score_thresh = 0;
inclass_nms_thresh = 0.3;
across_nms_thresh = 1.1;

start_at = 1;
end_at = 100;
for im_id = start_at:end_at;
  fprintf('%d / %d\n', im_id, end_at);
  im = imread(imdb.image_at(im_id));
  dets = boxes_set(im_id, :);
  save_path = fullfile(save_dir, [imdb.image_ids{im_id} suffix extension]);
  show_det_boxes(im, dets, imdb.classes, score_thresh, ...
    inclass_nms_thresh, across_nms_thresh, ...
    color, save_path);
end