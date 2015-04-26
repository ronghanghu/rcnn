function selective_search_boxes_imdb(imdb, save_path, start_id, end_id)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

im_width = 500;
fast_mode = true;

total_time = 0;
total_num = 0;
count = 0;

if ~exist('start_id', 'var')
  start_id = 1;
end
if ~exist('end_id', 'var')
  end_id = length(imdb.image_ids);
end

for i = start_id:end_id
  fprintf('%s: cache selective search boxes (s) %d/%d\n', procid(), i, end_id);
  image_id = imdb.image_ids{i};
  save_file = fullfile(save_path, [image_id '.mat']);
  if exist(save_file, 'file') ~= 0
    fprintf(' [already exists]\n');
    continue;
  end
  count = count + 1;

  tot_th = tic;
  
  try
    im = imread(imdb.image_at(i));
  catch lerr
    if ~strcmp(lerr.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')
      warning(lerr.message);
    end
  end
  % convert gray-scale image to color image
  if size(im, 3) == 1
    im = repmat(im, [1, 1, 3]);
  end

  th = tic;
  boxes = selective_search_boxes(im, fast_mode, im_width);
  fprintf(' [selective search: %.3fs]\n', toc(th));
  
  th = tic;
  save(save_file, 'image_id', 'boxes');
  fprintf(' [saving:   %.3fs]\n', toc(th));

  total_num = total_num + size(boxes, 1);
  total_time = total_time + toc(tot_th);
  fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
    total_time/count, total_time);
  fprintf(' [avg num: %.3fs]\n', total_num/count);
end