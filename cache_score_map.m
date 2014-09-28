function score_map = cache_score_map(im, rcnn_model)
%   Compute Spatial Pyramid Pooling features on a set of boxes.
%
%   im is an image in RGB order as returned by imread
%   boxes are in [x1 y1 x2 y2] format with one box per row
%   rcnn_model specifies the CNN Caffe net file to use.

% PARAMETER OF THE NETWORK
% TODO: move these parameters into rcnn_model
% NOTE: if you change any of these parameters, you must also change the
% corresponding network prototext file
% 5 Scale
% fixed_sizes = [640, 768, 917, 1152, 1600]';
% 1 Scale
fixed_sizes = [917]';

% extract features
score_map = spp_features_forward(im, rcnn_model, fixed_sizes);

end

function score_map = spp_features_forward(im, rcnn_model, fixed_sizes)
% extract SPP features

% make sure that caffe has been initialized for this model
if rcnn_model.cnn.init_key ~= caffe('get_init_key')
  error('You probably need to call rcnn_load_model');
end

% get the channel (BGR) mean
image_mean = rcnn_model.cnn.image_mean;
channel_mean = [mean2(image_mean(:,:,1)), ...
  mean2(image_mean(:,:,2)), ...
  mean2(image_mean(:,:,3))];

% input size is the size of image used for network input
input_size = max(fixed_sizes);
scale_num = size(fixed_sizes, 1);

% calculate zooming factor
[image_h, image_w, ~] = size(im);
image_l = max(image_h, image_w);
zoom_factors = fixed_sizes / image_l;
fixed_hs = round(min(image_h * zoom_factors, fixed_sizes));
fixed_ws = round(min(image_w * zoom_factors, fixed_sizes));

% calculate the multiscale image data and conv5 windows
multiscale_image_data = zeros(input_size, input_size, 3, scale_num, 'single');
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
  multiscale_image_data(1:fixed_ws(scale), 1:fixed_hs(scale), :, scale) = ...
      image_data;
end

% forward image data, conv5 windows and conv5 scales into caffe to get
% features
% split the windows into batches when window_num exceeds max_window_num
score_map = caffe('forward', {multiscale_image_data});
score_map = squeeze(score_map{1});
fg_score = score_map(:,:,2:end);
bg_score = score_map(:,:,1);
score_map = bsxfun(@minus, fg_score, bg_score);
score_map = permute(score_map, [2, 1, 3]);

end
