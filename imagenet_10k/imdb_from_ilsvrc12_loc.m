function imdb = imdb_from_ilsvrc12_loc(root_dir, image_set)
% create imdb from ILSVRC LOC 1K dataset
% using bounding box annotation
% the resulting class indices are 1~1000
% ---------------------------------------------------------

% root_dir = '/work4/rbg/ILSVRC13';

% names
% ilsvrc12_loc_train
% ilsvrc12_loc_val

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

cache_file = ['./imdb/cache/imdb_ilsvrc12_loc_' image_set];
try
  load(cache_file);
catch
  NUM_CLS = 1000;
  bbox_path.train = fullfile(root_dir, 'ILSVRC2012_bbox_train');
  bbox_path.val   = fullfile(root_dir, 'ILSVRC2012_bbox_val');
  im_path.train   = fullfile(root_dir, 'ILSVRC2012_train');
  im_path.val     = fullfile(root_dir, 'ILSVRC2012_val');
  devkit_path     = fullfile(root_dir, 'ILSVRC2013_devkit');
  meta_det        = load(fullfile(devkit_path, 'data', 'meta_clsloc.mat'));
  
  imdb.name = ['ilsvrc12_loc_' image_set];
  imdb.extension = 'JPEG';

  % no blacklist
  % is_blacklisted = containers.Map;

  imdb.image_dir = im_path.(image_set);
  imdb.details.image_list_file = ...
    fullfile(devkit_path, 'data', ['loc_list_' image_set '.txt']);
  [imdb.image_ids, ~] = textread(imdb.details.image_list_file, '%s %d');
  
  % all classes are present
  imdb.classes = {meta_det.synsets(1:NUM_CLS).words};
  imdb.num_classes = length(imdb.classes);
  imdb.class_to_id = ...
    containers.Map(imdb.classes, 1:imdb.num_classes);
  imdb.class_ids = 1:imdb.num_classes;
  
  if strcmp(imdb.name, 'ilsvrc12_loc_train')
    imdb.image_at = @(i) ...
      fullfile(imdb.image_dir, get_wnid(imdb.image_ids{i}), ...
      [imdb.image_ids{i} '.' imdb.extension]);
  elseif strcmp(imdb.name, 'ilsvrc12_loc_val')
    imdb.image_at = @(i) ...
      fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
  else
    error('unknown imdb %s', imdb.name);
  end
  
  imdb.details.blacklist_file = [];
  
  % private ILSVRC 2013 details
  imdb.details.meta_det    = meta_det;
  imdb.details.root_dir    = root_dir;
  imdb.details.devkit_path = devkit_path;
  
  % VOC specific functions for evaluation and region of interest DB
  imdb.roidb_func = @roidb_from_ilsvrc12_loc;
  
  % Some images are blacklisted due to noisy annotations
  imdb.is_blacklisted = false(length(imdb.image_ids), 1);
  
  for i = 1:length(imdb.image_ids)
    tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
    try
      im = imread(imdb.image_at(i));
      imdb.sizes(i, :) = [size(im, 1) size(im, 2)];
    catch
      lerr = lasterror;
      % gah, annoying data issues
      if strcmp(lerr.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')   
        warning('reading %s using imfinfo', imdb.image_at(i));
        info = imfinfo(imdb.image_at(i));
        assert(isscalar(info.Height) && info.Height > 0);
        assert(isscalar(info.Width) && info.Width > 0);
        imdb.sizes(i, :) = [info.Height info.Width];
      else
        warning(lerr.message);
        warning(imdb.image_at(i));
        % blacklist error images (for removal)
        imdb.is_blacklisted(i) = true;
      end
    end
  end
  % remove blacklisted images
  fprintf('Remove %d blacklisted images\n', sum(imdb.is_blacklisted));
  keep = ~imdb.is_blacklisted;
  imdb.image_ids(keep);
  imdb.sizes = imdb.sizes(keep, :);
  imdb.is_blacklisted = false(length(imdb.image_ids), 1);
  
  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
