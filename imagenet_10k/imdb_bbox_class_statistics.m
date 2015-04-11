function bbox_num_per_cls = imdb_bbox_class_statistics(imdb, numclasses)

bbox_num_per_cls = zeros(numclasses, 1);

for ii = 1:length(imdb)
  roidb = imdb(ii).roidb_func(imdb(ii));
  assert(length(roidb.rois) == length(imdb(ii).image_ids));
  for i = 1:length(roidb.rois)
    class_labels = roidb.rois(i).class;
    gt = roidb.rois(i).gt;
    class_labels = class_labels(gt);
    for n = 1:length(class_labels)
      label = class_labels(n);
      bbox_num_per_cls(label) = bbox_num_per_cls(label) + 1;
    end
  end
end

end