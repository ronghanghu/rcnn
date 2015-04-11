function [images, boxes] = selective_search_boxes_imdb(imdb)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

images = imdb.image_ids;

im_width = 500;
fast_mode = true;
mean_num = 0;
mean_time = 0;

numimages = length(imdb.image_ids);
result = cell(numimages, 1);
for i = 1:numimages
  fprintf('%d/%d (%s) ...', i, numimages, imdb.image_ids{i});
  try
    im = imread(imdb.image_at(i));
  catch lerr
    if strcmp(lerr.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')
      result{i} = [];
    else
      warning(lerr.message);
    end
    result{i} = [];
  end
  % convert gray-scale image to color image
  if size(im, 3) == 1
    im = repmat(im, [1, 1, 3]);
  end
  th = tic();
  result{i} = selective_search_boxes(im, fast_mode, im_width);
  t = toc(th);

  mean_num = (mean_num * (i-1) + size(result{i}, 1))/i;
  mean_time = (mean_time * (i-1) + t)/i;
  fprintf('%.2fs...%d boxes (means: %.2fs %.1f boxes)\n', t, ...
      size(result{i}, 1), mean_time, mean_num);
end