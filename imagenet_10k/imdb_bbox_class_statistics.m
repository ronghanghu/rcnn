function [classes, bbox_num_per_cls] = imdb_bbox_class_statistics(imdb)

classes = imdb(1).classes;
numclasses = length(classes);
bbox_num_per_cls = zeros(numclasses, 1);

for ii = 1:length(imdb)
  roidb = imdb(ii).roidb_func(imdb(ii));
  assert(length(roidb.rois) == length(imdb(ii).image_ids));
  for i = 1:length(roidb.rois)
    class_labels = roidb.rois(i).class;
    bbox_num_per_cls(class_labels) = bbox_num_per_cls(class_labels) + 1;
  end
end

end