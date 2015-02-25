%% load imdb and roidb

imdb_name = 'ilsvrc13_val2';
load(['imdb/cache/imdb_' imdb_name '.mat']);
load(['imdb/cache/roidb_' imdb_name '.mat']);
classes = imdb.classes;
num_classes = length(classes);
num_im = length(roidb.rois);

%% ground-truth boxes
agtboxes = cell(num_im, num_classes);
for i = 1:num_im
  gtboxes = roidb.rois(i).boxes(roidb.rois(i).gt, :);
  gtboxes = cat(2, gtboxes, 1000 * ones(size(gtboxes, 1), 1));
  gtclasses = roidb.rois(i).class(roidb.rois(i).gt);
  assert(size(gtboxes, 1) == length(gtclasses));
  for c = 1:num_classes
    agtboxes{i, c} = gtboxes(gtclasses == c, :);
  end
end

%% load detection boxes
bbox_dir = './cachedir/step_2v1_train_100_det_iter_50000.caffemodel';
aboxes = cell(num_im, num_classes);
for c = 1:num_classes
  fprintf('loading box class %d / %d\n', c, num_classes);
  cache_name = [classes{c} '_boxes_' imdb_name];
  load(fullfile(bbox_dir, imdb_name, cache_name));
  aboxes(:, c) = boxes;
end

%% show and save detection boxes
%% TODO the two lines below this before using
&&&&&&&& boxes_set = aboxes;
&&&&&&&& suffix = '_det_v1';

score_thresh = 0;
inclass_nms_thresh = 0.3;
across_nms_thresh = 1.1;
color = 'r';
save_dir = '~/Downloads';

start_at = 1;
end_at = 100;
for im_id = start_at:end_at;
  fprintf('%d / %d\n', im_id, end_at);
  im = imread(imdb.image_at(im_id));
  dets = boxes_set(im_id, :);
  save_path = fullfile(save_dir, [imdb.image_ids{im_id} suffix '.png']);
  show_det_boxes(im, dets, classes, score_thresh, inclass_nms_thresh, ...
    across_nms_thresh, color, save_path);
end
