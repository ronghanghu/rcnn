function feat = average_pool(feat_map, boxes, scales, stride, proposal_num)
boxes = boxes';

boxes(:, [1 2]) = floor(boxes(:, [1 2]) / stride) + 1;
boxes(:, [3 4]) = ceil(boxes(:, [3 4]) / stride) + 1;

class_num = size(feat_map, 3);

feat = zeros(proposal_num, class_num, 'single');

for n = 1:proposal_num
  x1 = boxes(n, 1);
  y1 = boxes(n, 2);
  x2 = boxes(n, 3);
  y2 = boxes(n, 4);
  area = (x2 - x1 + 1) * (y2 - y1 + 1);
  scale = scales(n);
  % keyboard;
  scores = zeros(1, class_num);
  for c = 1:class_num
    score_sum = sum(sum(feat_map(y1:y2, x1:x2, c, scale + 1)));
    scores(c) = score_sum / area;
  end
  feat(n, :) = scores;
end

end