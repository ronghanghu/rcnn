% statistics on  trainval
imdb_trainval = imdb_from_voc('datasets/VOCdevkit2007', 'trainval', '2007');
[all_boxes, all_overlaps] = rcnn_get_all_boxes(imdb_trainval);
save bbox_statistics/trainval_boxes.mat all_boxes all_overlaps;

% statistics on test
