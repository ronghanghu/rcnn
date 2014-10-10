function feat = proposal_features(im, boxes, rcnn_model)

fixed_sizes = [640, 768, 917, 1152, 1600]';
% fixed_sizes = [917]';
max_proposal_num = 10000;

% get channel mean
image_mean = rcnn_model.cnn.image_mean;
channel_mean = [mean2(image_mean(:,:,1)), ...
  mean2(image_mean(:,:,2)), ...
  mean2(image_mean(:,:,3))];
  
% input size is the size of image used for network input
input_size = max(fixed_sizes);
scale_num = size(fixed_sizes, 1);
proposal_num = size(boxes, 1);
assert(proposal_num <= max_proposal_num);

% calculate zooming factor
[image_h, image_w, ~] = size(im);
image_l = max(image_h, image_w);
zoom_factors = fixed_sizes / image_l;
fixed_hs = round(min(image_h * zoom_factors, fixed_sizes));
fixed_ws = round(min(image_w * zoom_factors, fixed_sizes));

% match the boxes onto optimal scale, on which the mapped boxes is
% closest to 227*227
desired_area = 227*227;
boxes_area = (boxes(:, 4) - boxes(:, 2) + 1) .* (boxes(:, 3) - boxes(:, 1) + 1);
zoomed_area = boxes_area * (zoom_factors.^2)';
area_dif = abs(zoomed_area - desired_area);
[~, scale] = min(area_dif, [], 2);
bottom_scale = single(scale - 1); % convert 1-indexed scale to 0-indexed scale

% calculate the multiscale image data and conv5 windows
bottom_data = zeros(input_size, input_size, 3, scale_num, 'single');
bottom_box = zeros(4, max_proposal_num, 'single');
for scale = 1:scale_num
  % resize the image to a fixed scale, turn off antialiasing to make it
  % similar to imresize in OpenCV
  resized_im = imresize(im, [fixed_hs(scale), fixed_ws(scale)], 'bilinear', ...
      'antialiasing', false);
  % convert from RGB channels to BGR channels
  image_data = single(resized_im(:, :, [3, 2, 1]));
  % mean subtraction
  for c = 1:3
    image_data(:, :, c) = image_data(:, :, c) - channel_mean(c);
  end
  % set width to be the fastest dimension
  image_data = permute(image_data, [2, 1, 3]);
  bottom_data(1:fixed_ws(scale), 1:fixed_hs(scale), :, scale) = ...
      image_data;
  
  % boxes do not need to be divided by stride here
  % (it is done in DenseWindowLossLayer)
  resized_boxes = single(round((boxes - 1) * zoom_factors(scale)));

  % set width to be the fastest dimension
  resized_boxes = permute(resized_boxes, [2, 1]);
  is_matched = (scale - 1 == bottom_scale);
  bottom_box(:, is_matched) = resized_boxes(:, is_matched);
end

% in testing, just set all labels to be zero (background)
bottom_window_num = single([proposal_num, 0, proposal_num]');
bottom_scale = cat(1, bottom_scale, ...
    zeros(max_proposal_num - proposal_num, 1, 'single'));
bottom_label = zeros(max_proposal_num, 1, 'single');

bottom = ...
    {bottom_data; bottom_window_num; bottom_box; bottom_scale; bottom_label};

% forward the network
top = caffe('forward', bottom);

% get results
feat = permute(squeeze(top{2}), [2, 1]);
feat = feat(1:proposal_num, :);

end
