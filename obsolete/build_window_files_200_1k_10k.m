% build a image list for 10k training
% For those classes that are in ImageNET 200 or ImageNET 1K, use bounding
% box. For other classes, use the whole image

% output directory
output_dir = '/home/ronghang/imagenet_10k_window_files';
if exist(output_dir, 'dir') == 0
  error('output directory %s does not exist', output_dir);
end
fprintf('building window files to directory %s\n', output_dir);

%% -----------------------------------------------------------------------------
% create map vector for 200 and 1k
load external/mhex_graph/+imagenet/meta_7k.mat;
load external/mhex_graph/+imagenet/meta_1k.mat;
load external/mhex_graph/+imagenet/meta_200.mat;

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

%% -----------------------------------------------------------------------------
% create 10k map vec (remove those classes existing in 200 or 1k)
% find 10k classes that are not in 200 or 1K, and create an image list of
% them. i.e. we only use whole image for a class when there is no bounding
% box available

% Note: those overlapping with 3k are still kept
class_num_10k = length(synsets_7k);
is_in_200_or_1k = ...
  wnid2label_200.isKey({synsets_7k.WNID}) | ...
  wnid2label_1k.isKey({synsets_7k.WNID});
map_vec_10k_to_10k = (1:class_num_10k)';
map_vec_10k_to_10k(is_in_200_or_1k) = -1;

% insert background at beginning
map_vec_10k_to_10k = [0; map_vec_10k_to_10k];

%% -----------------------------------------------------------------------------
% create 200 & 1k & 10k window files

% load 200 imdb
fprintf('loading imdbs (this may take a while)...');
imdb_ilsvrc_val1 = imdb_from_ilsvrc13('./datasets/ILSVRC13', 'val1');
fprintf('.');
for n = 1:200
  imdb_ilsvrc_train(n, 1) = imdb_from_ilsvrc13('./datasets/ILSVRC13', ...
    ['train_pos_' num2str(n)]);
  fprintf('.');
end
% load 1k imdb
imdb_ilsvrc12_loc = imdb_from_ilsvrc12_loc('./datasets/ILSVRC13', 'train');
assert(strcmp(imdb_ilsvrc12_loc.name, 'ilsvrc12_loc_train'));
fprintf('.');
% load 10k imdb
imdb_imagenet10k_cls = imdb_from_imagenet10k_cls('./datasets/imagenet_10k', 'train');
assert(strcmp(imdb_imagenet10k_cls.name, 'imagenet10k_cls_train'));
fprintf('.');
fprintf('done\n');

% concatenate all imdbs together
imdb_ilsvrc_val1 = rmfield(imdb_ilsvrc_val1, 'eval_func');
imdb_ilsvrc_train = rmfield(imdb_ilsvrc_train, 'eval_func');
imdb_all = [
  imdb_ilsvrc_val1 % ImageNET 200 val1 (1 imdb)
  imdb_ilsvrc_train % ImageNET 200 train (200 imdbs)
  imdb_ilsvrc12_loc % ImageNET 1K train (1 imdb)
  imdb_imagenet10k_cls % ImageNET 10K train (imdb)
]';

% set up label mapping
label_map_flag = [
  true
  true(200, 1)
  true
  true
];

label_map_cell = [
  repmat({map_vec_200_to_10k}, 1+200, 1)
  {map_vec_1k_to_10k; map_vec_10k_to_10k}
];

% set to 1000 as in rcnn. it only affects imagenet 200 train
num_to_sample = 1000;

% write window file
rcnn_make_window_file_map_labels(imdb_all, output_dir, ...
    'mapped_200_1k_10k_to_10k', num_to_sample, ...
    label_map_flag, label_map_cell);