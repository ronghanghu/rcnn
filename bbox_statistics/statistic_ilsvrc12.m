load roidb_ilsvrc13_val2;
img_num = size(roidb.rois, 2);
all_pos_num = 0;
all_neg_num = 0;

for n = 1:img_num
  fprintf('%d / %d\n', n, img_num);
  max_overlap = max(roidb.rois(1,n).overlap, [], 2);
  pos_num = sum(max_overlap > 0.5);
  neg_num = sum(max_overlap < 0.5 & max_overlap > 0.1);
  all_pos_num = all_pos_num + pos_num;
  all_neg_num = all_neg_num + neg_num;
end
