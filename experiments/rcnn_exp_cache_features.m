function rcnn_exp_cache_features(chunk)

% -------------------- CONFIG --------------------
net_file     = './data/caffe_nets/pascal_cccp3_proposal.caffemodel';
cache_name   = 'cccp3_proposal';
crop_mode    = 'warp';
crop_padding = 16;

% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
% ------------------------------------------------

imdb_train = imdb_from_voc(VOCdevkit, 'train', '2007');
imdb_val   = imdb_from_voc(VOCdevkit, 'val', '2007');
imdb_test  = imdb_from_voc(VOCdevkit, 'test', '2007');
imdb_trainval = imdb_from_voc(VOCdevkit, 'trainval', '2007');

switch chunk
  case 'train'
    rcnn_cache_pool5_features(imdb_train, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
    link_up_trainval(cache_name, imdb_train, imdb_trainval);
  case 'val'
    rcnn_cache_pool5_features(imdb_val, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
    link_up_trainval(cache_name, imdb_val, imdb_trainval);
  case 'test_1'
    end_at = ceil(length(imdb_test.image_ids)/2);
    rcnn_cache_pool5_features(imdb_test, ...
        'start', 1, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'test_2'
    start_at = ceil(length(imdb_test.image_ids)/2)+1;
    rcnn_cache_pool5_features(imdb_test, ...
        'start', start_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
end


% ------------------------------------------------------------------------
function link_up_trainval(cache_name, imdb_split, imdb_trainval)
% ------------------------------------------------------------------------
cmd = {['mkdir -p ./feat_cache/' cache_name '/' imdb_trainval.name '; '], ...
    ['cd ./feat_cache/' cache_name '/' imdb_trainval.name '/; '], ...
    ['for i in `ls -1 ../' imdb_split.name '`; '], ... 
    ['do ln -s ../' imdb_split.name '/$i $i; '], ... 
    ['done;']};
cmd = [cmd{:}];
fprintf('running:\n%s\n', cmd);
system(cmd);
fprintf('done\n');
