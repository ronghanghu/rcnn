function imdb = imdb_from_imagenet10k_cls_fast(root_dir, image_set)
% create imdb from ImageNET CLS 10K dataset
% using whole image as bounding box
% the resulting class indices are 1~10447
% ---------------------------------------------------------

% root_dir = '/work4/rbg/ILSVRC13';

% names
% imagenet10k_cls_train
% imagenet10k_cls_test

%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/ILSVRC/ILSVRC2013_DET_train/n02672831/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'n02672831_11478', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'accordian', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

cache_file = ['./imdb/cache/imdb_imagenet10k_cls_' image_set];
try
  load(cache_file);
catch
  NUM_CLS = 10447;
  
  % no bounding box for ImageNET 10K CLS
  % the directories are specified in im_list files
  im_path.train   = fullfile(root_dir, 'train');
  im_path.test    = fullfile(root_dir, 'test');
  im_list.train   = fullfile(root_dir, 'llc_10k-100-train.txt');
  im_list.test    = fullfile(root_dir, 'llc_10k-50-test.txt');
  
  % no devkit path
  % devkit_path     = '';
  meta_det        = load('./external/mhex_graph/+imagenet/meta_7k.mat');
  
  imdb.name = ['imagenet10k_cls_' image_set];
  imdb.extension = 'jpg';
  % no blacklist
  % is_blacklisted = containers.Map;
  
  IM_LENGTH = 256;
  
  imdb.image_dir = im_path.(image_set);
%   imdb.details.image_list_file = im_list.(image_set);
%   [imdb.image_ids, ~] = textread(imdb.details.image_list_file, '%s %d');
  iminfo_all = dir([imdb.image_dir '*.' imdb.extension]);
  % keep those valid images and remove extension
  keep = false(length(iminfo_all), 1);
  for n = 1:length(iminfo_all)
    iminfo_all(n).name = iminfo_all(n).name(1:end-4);
    keep(n) = (iminfo_all(n).bytes > 0);
  end
  iminfo_all = iminfo_all(keep);
  
  imdb.image_ids = {iminfo_all.name};
  
  % all classes are present
  imdb.classes = {meta_det.synsets_7k(1:NUM_CLS).words};
  imdb.num_classes = length(imdb.classes);
  imdb.class_to_id = ...
    containers.Map(imdb.classes, 1:imdb.num_classes);
  imdb.class_ids = 1:imdb.num_classes;
  
  imdb.image_at = @(i) ...
    fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
  
  imdb.details.blacklist_file = [];
  % no bbox path
  imdb.details.bbox_path = '';
  
  % private ILSVRC 2013 details
  imdb.details.meta_det    = meta_det;
  imdb.details.root_dir    = root_dir;
  % imdb.details.devkit_path = devkit_path;
  
  % VOC specific functions for evaluation and region of interest DB
  imdb.roidb_func = @roidb_from_imagenet10k_cls;
  
  % Some images are blacklisted due to noisy annotations
  imdb.is_blacklisted = false(length(imdb.image_ids), 1);
  
  % size set length to resized ones
  imdb.sizes = IM_LENGTH * ones(length(imdb.image_ids), 2);

  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
end