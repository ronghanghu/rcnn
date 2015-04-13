%% load 200 imdb
fprintf('loading imdbs for 200');
imdb_ilsvrc_val1 = imdb_from_ilsvrc13('./datasets/ILSVRC13', 'val1');
fprintf('.');
for n = 1:200
  imdb_ilsvrc_train(n, 1) = imdb_from_ilsvrc13('./datasets/ILSVRC13', ...
    ['train_pos_' num2str(n)]);
  fprintf('.');
end
fprintf('done\n');

fprintf('running statistics...');
stat_200_val1 = imdb_bbox_class_statistics(imdb_ilsvrc_val1, 200);
stat_200_train = imdb_bbox_class_statistics(imdb_ilsvrc_train, 200);
% at most 1000 per class
stat_200 = stat_200_val1 + min(stat_200_train, 1000);
fprintf('done\n');
clear imdb_ilsvrc_val1 imdb_ilsvrc_train n;

%% load 1k imdb
fprintf('loading imdbs for 1k...');
imdb_ilsvrc12_loc = imdb_from_ilsvrc12_loc('./datasets/ILSVRC13', 'train');
assert(strcmp(imdb_ilsvrc12_loc.name, 'ilsvrc12_loc_train'));
fprintf('done\n');
fprintf('running statistics...');
stat_1k = imdb_bbox_class_statistics(imdb_ilsvrc12_loc, 1000);
fprintf('done\n');
clear imdb_ilsvrc12_loc;

%% load 3k imdb
fprintf('loading imdbs for 3k...');
imdb_imagenet3k_loc = imdb_from_imagenet3k_loc('./datasets/imagenet_3k', 'train');
assert(strcmp(imdb_imagenet3k_loc.name, 'imagenet3k_loc_train'));
fprintf('done\n');
fprintf('running statistics...');
stat_3k = imdb_bbox_class_statistics(imdb_imagenet3k_loc, 10447);
fprintf('done\n');
clear imdb_imagenet3k_loc;

%% load 10k imdb
fprintf('loading imdbs for 10k...');
imdb_imagenet10k_cls = imdb_from_imagenet10k_cls('./datasets/imagenet_10k', 'train');
assert(strcmp(imdb_imagenet10k_cls.name, 'imagenet10k_cls_train'));
fprintf('done\n');

fprintf('running statistics...');
stat_10k = imdb_bbox_class_statistics(imdb_imagenet10k_cls, 10447);
fprintf('done\n');
clear imdb_imagenet10k_cls;

%% post-process data
load external/mhex_graph/+imagenet/meta_extended.mat;
stat_all_loc = zeros(size(synsets_extended));
stat_all_cls = zeros(size(synsets_extended));
for n = 1:length(synsets_extended)
  imagenet_200_id = synsets_extended(n).imagenet_200_id;
  imagenet_1k_id = synsets_extended(n).imagenet_1k_id;
  imagenet_10k_id = synsets_extended(n).imagenet_10k_id;
  % add 200
  if ~isempty(imagenet_200_id)
    stat_all_loc(n) = stat_all_loc(n) + stat_200(imagenet_200_id);
  end
  % add 1k
  if ~isempty(imagenet_1k_id)
    stat_all_loc(n) = stat_all_loc(n) + stat_1k(imagenet_1k_id);
  end
  % add 3k and 10k
  if ~isempty(imagenet_10k_id)
    stat_all_loc(n) = stat_all_loc(n) + stat_3k(imagenet_10k_id);
    stat_all_cls(n) = stat_all_cls(n) + stat_10k(imagenet_10k_id);
  end
end
stat_all = stat_all_cls + stat_all_loc;
clear synsets_extended wnid2label_extended ...
  imagenet_200_id imagenet_1k_id imagenet_10k_id n

%% calculate duplicate rate
NUM_PER_CLS = 1000;
duplicate_rate_extended = ...
  ceil(max(NUM_PER_CLS - stat_all_loc, 1) ./ max(stat_all_cls, 1));

%% save results
save imagenet_10k/statistics.mat -v7.3