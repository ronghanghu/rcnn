function feat = spp_features(im, boxes, rcnn_model)
% feat = spp_features(im, boxes, rcnn_model)
%   Compute Spatial Pyramid Pooling features on a set of boxes.
%
%   im is an image in RGB order as returned by imread
%   boxes are in [x1 y1 x2 y2] format with one box per row
%   rcnn_model specifies the CNN Caffe net file to use.

% % make sure that caffe has been initialized for this model
% if rcnn_model.cnn.init_key ~= caffe('get_init_key')
%     error('You probably need to call rcnn_load_model');
% end

% PARAMETER OF THE NETWORK
% TODO: move these parameters into rcnn_model
% NOTE: if you change any of these parameters, you must also change the
% corresponding network prototext file
fixed_sizes = [480, 576, 688, 864,1200]';
conv5_sizes = [ 29,  35,  42,  53,  74]';
max_proposal_num = 2500;
conv5_edge_on_image = 18; % Alex net
% conv5_edge_on_image = 17; % Zeiler & Fergus net
conv5_stride = 16;

% calculate the channel (BGR) mean from image mean
image_mean = rcnn_model.cnn.image_mean;
channel_mean = [mean2(image_mean(:,:,1)), ...
  mean2(image_mean(:,:,2)), ...
  mean2(image_mean(:,:,3))];

% input size is the size of image used for network input
input_size = max(fixed_sizes);
scale_num = size(fixed_sizes, 1);
proposal_num = size(boxes, 1);

% calculate zooming factor
[image_h, image_w, ~] = size(im);
image_l = max(image_h, image_w);
zoom_factors = fixed_sizes / image_l;
fixed_hs = round(min(image_h * zoom_factors, fixed_sizes));
fixed_ws = round(min(image_w * zoom_factors, fixed_sizes));

% match the boxes onto optimal scale, on which the mapped boxes is
% closest to 227*227
desired_area = 51529; % 227*227
boxes_area = (boxes(:, 4) - boxes(:, 2) + 1) .* (boxes(:, 3) - boxes(:, 1) + 1);
zoomed_area = boxes_area * (zoom_factors.^2)';
area_dif = abs(zoomed_area - desired_area);
[~, scale] = min(area_dif, [], 2);
conv5_scales = single(scale - 1); % convert 1-indexed scale to 0-indexed scale

% calculate the multiscale image data and conv5 windows
multiscale_image_data = zeros(input_size, input_size, 3, scale_num, 'single');
multiscale_conv5_windows = zeros(4, proposal_num, 'single');
for scale = 1:scale_num
  % resize the image to a fixed scale, turn off antialiasing to make it
  % similar to imresize in OpenCV
  resized_im = imresize(im, [fixed_hs(scale), fixed_ws(scale)], 'bilinear', 'antialiasing', false);
  % convert from RGB channels to BGR channels
  image_data = single(resized_im(:, :, [3, 2, 1]));
  % mean subtraction
  for channel = 1:3
    image_data(:, :, channel) = image_data(:, :, channel) - channel_mean(channel);
  end
  % set width to be the fastest dimension
  image_data = permute(image_data, [2, 1, 3]);
  multiscale_image_data(1:fixed_ws(scale), 1:fixed_hs(scale), :, scale) = image_data;

  % resize the boxes and calculate the conv5 windows ([y1 x1 y2 x2],
  % 0-indexed)
  resized_boxes = (boxes - 1) * zoom_factors(scale) + 1;
  conv5_windows = single(round((resized_boxes(:, [2 1 4 3]) - conv5_edge_on_image) / conv5_stride));
  % add 1 to the ends
  conv5_windows(:, [3, 4]) = conv5_windows(:, [3, 4]) + 1;
  % make sure the windows fit into the conv5 maps
  conv5_windows = min(max(conv5_windows, 0), conv5_sizes(scale));
  % set width to be the fastest dimension
  conv5_windows = permute(conv5_windows, [2, 1]);
  is_matched = (scale - 1 == conv5_scales);
  multiscale_conv5_windows(:, is_matched) = conv5_windows(:, is_matched);
end

% forward image data, conv5 windows and conv5 scales into caffe to get
% features
% split the windows into batches when window_num exceeds max_window_num
feat = [];
for start_id = 1:max_proposal_num:proposal_num
  end_id = min(proposal_num, start_id + max_proposal_num - 1);
  conv5_windows_batch = zeros(4, max_proposal_num, 'single');
  conv5_windows_batch(:, 1:end_id-start_id+1) = multiscale_conv5_windows(:, start_id:end_id);
  conv5_scales_batch = zeros(max_proposal_num, 1, 'single');
  conv5_scales_batch(1:end_id-start_id+1) = conv5_scales(start_id:end_id);
  batch = {multiscale_image_data; conv5_windows_batch; conv5_scales_batch};
  caffe_output = caffe('forward', batch);
  feat = cat(1, feat, squeeze(caffe_output{1})');
end
feat = feat(1:proposal_num, :);

end