function aboxes = load_gt_from_roidb(imdb, roidb)

num_classes = length(imdb.classes);
num_im = length(roidb.rois);

aboxes = cell(num_im, num_classes);

for i = 1:num_im
  gtboxes = roidb.rois(i).boxes(roidb.rois(i).gt, :);
  gtboxes = cat(2, gtboxes, 1000 * ones(size(gtboxes, 1), 1));
  gtclasses = roidb.rois(i).class(roidb.rois(i).gt);
  assert(size(gtboxes, 1) == length(gtclasses));
  for c = 1:num_classes
    aboxes{i, c} = gtboxes(gtclasses == c, :);
  end
end

end

