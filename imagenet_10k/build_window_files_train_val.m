% build a image list for 10k training
% For those classes that are in ImageNET 200 or ImageNET 1K or 3K, use bounding
% box. For other classes, use the whole image

% output directory
output_dir = '..';
if exist(output_dir, 'dir') == 0
  error('output directory %s does not exist', output_dir);
end
fprintf('building window files to directory %s\n', output_dir);

%% -----------------------------------------------------------------------------
% create map vector for 200 and 1k
load external/mhex_graph/+imagenet/meta_extended.mat;
load external/mhex_graph/+imagenet/meta_1k.mat;
load external/mhex_graph/+imagenet/meta_200.mat;

map_vec_200_to_10k = zeros(200, 1);
for v = 1:200
  map_vec_200_to_10k(v) = wnid2label_extended(synsets_200(v).WNID);
end
% insert background at beginning
map_vec_200_to_10k = [0; map_vec_200_to_10k];

map_vec_1k_to_10k = zeros(1000, 1);
for v = 1:1000
  map_vec_1k_to_10k(v) = wnid2label_extended(synsets_1k(v).WNID);
end
% insert background at beginning
map_vec_1k_to_10k = [0; map_vec_1k_to_10k];

%% -----------------------------------------------------------------------------
% create 200 & 1k & 10k window files

if run_train
  fprintf('building window files for train\n');
  
  % load 200 imdb
  fprintf('loading 200 imdbs (this may take a while)');
  for n = 1:200
    imdb_200_train(n, 1) = imdb_from_ilsvrc13('./datasets/ILSVRC13', ...
      ['train_pos_' num2str(n)]);
    fprintf('.');
  end
  imdb_200_val1 = imdb_from_ilsvrc13('./datasets/ILSVRC13', 'val1');
  fprintf('.');
  fprintf('done\n');
  imdb_200_train = rmfield(imdb_200_train, 'eval_func');
  imdb_200_val1 = rmfield(imdb_200_val1, 'eval_func');
  
  % load 1k imdb
  fprintf('loading 1k imdbs (this may take a while)');
  imdb_1k_train = imdb_from_ilsvrc12_loc('./datasets/ILSVRC13', 'train');
  fprintf('.');
  fprintf('done\n');
  assert(strcmp(imdb_1k_train.name, 'ilsvrc12_loc_train'));
  
  % load 3k imdb
  fprintf('loading 3k imdbs (this may take a while)');
  imdb_3k_train = imdb_from_imagenet3k_loc('./datasets/imagenet_3k', 'train');
  fprintf('.');
  fprintf('done\n');
  assert(strcmp(imdb_3k_train.name, 'imagenet3k_loc_train'));
  
  % concatenate all imdbs together and set up label mapping
  imdb_all_train = [imdb_200_val1; imdb_200_train; imdb_1k_train; imdb_3k_train;]';
  label_map_flag_train = [true(1+200, 1); true; false];
  label_map_cell_train = ...
    [repmat({map_vec_200_to_10k}, 1+200, 1); {map_vec_1k_to_10k; []}];

  % set to 1000 as in rcnn. it only affects imagenet 200 train
  num_to_sample = 1000;
  
  % write window file
  rcnn_make_window_file_map_labels(imdb_all_train, output_dir, ...
    'mapped_200_1k_3k_to_10k_train', num_to_sample, ...
    label_map_flag_train, label_map_cell_train);

else
  fprintf('building window files for val\n');
  
  % load 200 imdb
  fprintf('loading 200 imdbs (this may take a while)');
  imdb_200_val2 = imdb_from_ilsvrc13('./datasets/ILSVRC13', 'val2');
  fprintf('.');
  fprintf('done\n');
  imdb_200_val2 = rmfield(imdb_200_val2, 'eval_func');
  
  % load 1k imdb
  fprintf('loading 1k imdbs (this may take a while)');
  imdb_1k_val   = imdb_from_ilsvrc12_loc('./datasets/ILSVRC13', 'val');
  fprintf('.');
  fprintf('done\n');
  assert(strcmp(imdb_1k_val.name, 'ilsvrc12_loc_val'));
  
  % load 3k imdb
  fprintf('loading 3k imdbs (this may take a while)');
  imdb_3k_val = imdb_from_imagenet3k_loc('./datasets/imagenet_3k', 'val');
  fprintf('.');
  fprintf('done\n');
  assert(strcmp(imdb_3k_val.name, 'imagenet3k_loc_val'));
  
  % concatenate all imdbs together and set up label mapping
  imdb_all_val = [imdb_200_val2; imdb_1k_val; imdb_3k_val;]';
  label_map_flag_val = [true; true; false];
  label_map_cell_val = {map_vec_200_to_10k; map_vec_1k_to_10k; []};
  
  % set to 1000 as in rcnn. it only affects imagenet 200 train
  num_to_sample = 1000;
  
  rcnn_make_window_file_map_labels(imdb_all_val, output_dir, ...
    'mapped_200_1k_3k_to_10k_val', num_to_sample, ...
    label_map_flag_val, label_map_cell_val);

end