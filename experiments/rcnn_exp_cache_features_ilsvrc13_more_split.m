function rcnn_exp_cache_features_ilsvrc13_more_split(chunk)

% -------------------- CONFIG --------------------
%net_file     = './data/caffe_nets/ilsvrc_2012_train_iter_310k';
%cache_name   = 'v1_caffe_imagenet_train_iter_310k';
%crop_mode    = 'warp';
%crop_padding = 16;

%net_file     = '/data1/ILSVRC13/finetune_ilsvrc13_val1_iter_50000';
%cache_name   = 'v1_finetune_val1_iter_50k';
%crop_mode    = 'warp';
%crop_padding = 16;

net_file     = './data/caffe_nets/finetune_ilsvrc13_val1+train1k_iter_50000';
cache_name   = 'v1_finetune_val1+train1k_iter_50k';
crop_mode    = 'warp';
crop_padding = 16;

% change to point to your VOCdevkit install
devkit = './datasets/ILSVRC13';
% ------------------------------------------------

imdb_val1 = imdb_from_ilsvrc13(devkit, 'val1');
imdb_val2 = imdb_from_ilsvrc13(devkit, 'val2');
imdb_test = imdb_from_ilsvrc13(devkit, 'test');

test_chunks = round(linspace(0, length(imdb_test.image_ids), 5));

switch chunk
  % ----------------------------------------------------------------------
  % Val set chunks
  % ----------------------------------------------------------------------
  case 'val1_1'
    end_at = ceil(length(imdb_val1.image_ids)/2);
    rcnn_cache_pool5_features(imdb_val1, ...
        'start', 1, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'val1_2'
    start_at = ceil(length(imdb_val1.image_ids)/2)+1;
    rcnn_cache_pool5_features(imdb_val1, ...
        'start', start_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'val2_1'
    start_at = 1;
    end_at = ceil(length(imdb_val2.image_ids)/4);
    rcnn_cache_pool5_features(imdb_val2, ...
        'start', start_at, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'val2_2'
    start_at = ceil(length(imdb_val2.image_ids)/4)+1;
    end_at = ceil(length(imdb_val2.image_ids)*2/4);
    rcnn_cache_pool5_features(imdb_val2, ...
        'start', start_at, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'val2_3'
    start_at = ceil(length(imdb_val2.image_ids)*2/4)+1;
    end_at = ceil(length(imdb_val2.image_ids)*3/4);
    rcnn_cache_pool5_features(imdb_val2, ...
        'start', start_at, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'val2_4'
    start_at = ceil(length(imdb_val2.image_ids)*3/4)+1;
    end_at = length(imdb_val2.image_ids);
    rcnn_cache_pool5_features(imdb_val2, ...
        'start', start_at, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);

  % ----------------------------------------------------------------------
  % Train set chunks
  % ----------------------------------------------------------------------
  case 'train1'
    for i = 1:100
      imdb_train = imdb_from_ilsvrc13(devkit, ['train_pos_' num2str(i)]);
      rcnn_cache_pool5_features(imdb_train, ...
          'crop_mode', crop_mode, ...
          'crop_padding', crop_padding, ...
          'net_file', net_file, ...
          'cache_name', cache_name);
    end
  case 'train2'
    for i = 101:200
      imdb_train = imdb_from_ilsvrc13(devkit, ['train_pos_' num2str(i)]);
      rcnn_cache_pool5_features(imdb_train, ...
          'crop_mode', crop_mode, ...
          'crop_padding', crop_padding, ...
          'net_file', net_file, ...
          'cache_name', cache_name);
    end

  % ----------------------------------------------------------------------
  % Test set chunks
  % ----------------------------------------------------------------------
  case 'test1'
    start_at = test_chunks(1)+1;
    end_at = test_chunks(2);
    rcnn_cache_pool5_features(imdb_test, ...
        'start', start_at, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'test2'
    start_at = test_chunks(2)+1;
    end_at = test_chunks(3);
    rcnn_cache_pool5_features(imdb_test, ...
        'start', start_at, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'test3'
    start_at = test_chunks(3)+1;
    end_at = test_chunks(4);
    rcnn_cache_pool5_features(imdb_test, ...
        'start', start_at, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'test4'
    start_at = test_chunks(4)+1;
    end_at = test_chunks(5);
    rcnn_cache_pool5_features(imdb_test, ...
        'start', start_at, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);

end
