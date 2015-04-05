error('this file is obsolete. do not use it');

% build a image list for 10k training
% For those classes that are in ImageNET 200 or ImageNET 1K, use bounding
% box. For other classes, use the whole image

load external/mhex_graph/+imagenet/meta_7k.mat;
load external/mhex_graph/+imagenet/meta_1k.mat;
load external/mhex_graph/+imagenet/meta_200.mat;

%% create map vector for 200 and 1k
map_vec_200_to_10k = zeros(200, 1);
for v = 1:200
  WNID = synsets_200(v).WNID;
  try
    label = wnid2label_7k(WNID);
  catch
    label = -1;
  end
  map_vec_200_to_10k(v) = label;
end
% insert background at beginning
map_vec_200_to_10k = [0; map_vec_200_to_10k];

map_vec_1k_to_10k = zeros(1000, 1);
for v = 1:1000
  WNID = synsets_1k(v).WNID;
  try
    label = wnid2label_7k(WNID);
  catch
    label = -1;
  end
  map_vec_1k_to_10k(v) = label;
end
% insert background at beginning
map_vec_1k_to_10k = [0; map_vec_1k_to_10k];

%% create 10k map vec
% find 10k classes that are not in 200 or 1K, and create an image list of
% them
class_num_10k = length(synsets_7k);
is_in_200_or_1k = true(class_num_10k, 1);
for v = 1:class_num_10k
  WNID = synsets_7k(v).WNID;
  try
    wnid2label_200(WNID);
    wnid2label_1k(WNID);
    is_in_200_or_1k(v) = true;
  catch err_msg
    if strcmp(err_msg.identifier, 'MATLAB:Containers:Map:NoKey')
      is_in_200_or_1k(v) = false;
    else
      error(err_msg);
    end
  end
end
map_vec_10k_to_10k = (1:class_num_10k)';
map_vec_10k_to_10k(is_in_200_or_1k) = -1;
% insert background at beginning
map_vec_10k_to_10k = [0; map_vec_10k_to_10k];

%% create 200 & 1k window files

out_dir = '../';

% load 200 imdb
load imdb/cache/all_ilsvrc.mat;
imdb_ilsvrc13_det = [imdb_ilsvrc_val1 imdb_ilsvrc_train];

% load 1k imdb
imdb_ilsvrc12_loc = load('imdb/cache/imdb_ilsvrc12_loc_train.mat');
imdb_ilsvrc12_loc = imdb_ilsvrc12_loc.imdb;
assert(strcmp(imdb_ilsvrc12_loc.name, 'ilsvrc12_loc_train'));

% load 10k imdb
imdb_imagenet10k_cls = load('imdb/cache/imdb_imagenet10k_cls_train.mat');
imdb_imagenet10k_cls = imdb_imagenet10k_cls.imdb;
assert(strcmp(imdb_imagenet10k_cls.name, 'imagenet10k_cls_train'));

% write 200 window file
num_to_sample = 1000;
map_label = true(201, 1);
whole_im = false(201, 1);
ending_index = rcnn_make_mapped_window_file(imdb_ilsvrc13_det, out_dir, ...
    'mapped_200_to_10k', ...
    num_to_sample, map_label, whole_im, map_vec_200_to_10k, 0);
disp(ending_index);

% write 1k window file
map_label = true;
whole_im = false;
ending_index = rcnn_make_mapped_window_file(imdb_ilsvrc12_loc, out_dir, ...
    'mapped_1k_to_10k', ...
    num_to_sample, map_label, whole_im, map_vec_1k_to_10k, ending_index);
disp(ending_index);

% write 10k window file
map_label = true;
whole_im = false;
ending_index = rcnn_make_mapped_window_file(imdb_imagenet10k_cls, out_dir, ...
    'mapped_10k_to_10k', ...
    num_to_sample, map_label, whole_im, map_vec_10k_to_10k, ending_index);
disp(ending_index);