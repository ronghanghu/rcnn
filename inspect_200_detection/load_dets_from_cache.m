function aboxes = load_dets_from_cache(bbox_dir, imdb, roidb)

num_classes = length(imdb.classes);
num_im = length(roidb.rois);

aboxes = cell(num_im, num_classes);

for c = 1:num_classes
  fprintf('loading box class %d / %d\n', c, num_classes);
  cache_name = [imdb.classes{c} '_boxes_' imdb.name];
  boxes = load(fullfile(bbox_dir, imdb.name, cache_name));
  boxes = boxes.boxes;
  aboxes(:, c) = boxes;
end

end

