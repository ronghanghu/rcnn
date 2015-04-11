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
[~, stat_200_val1] = imdb_bbox_class_statistics(imdb_ilsvrc_val1);
[~, stat_200_train] = imdb_bbox_class_statistics(imdb_ilsvrc_train);
stat_200_train = min(stat_200_train, 1000); % at most 1000 per class
stat_200 = stat_200_val1 + stat_200_train;
fprintf('done\n');
clear imdb_ilsvrc_val1 imdb_ilsvrc_train n;

%% load 1k imdb
fprintf('loading imdbs for 1k...');
imdb_ilsvrc12_loc = imdb_from_ilsvrc12_loc('./datasets/ILSVRC13', 'train');
assert(strcmp(imdb_ilsvrc12_loc.name, 'ilsvrc12_loc_train'));
fprintf('done\n');
fprintf('running statistics...');
[~, stat_1k] = imdb_bbox_class_statistics(imdb_ilsvrc12_loc);
fprintf('done\n');
clear imdb_ilsvrc12_loc;

%% load 3k imdb
fprintf('loading imdbs for 3k...');
imdb_imagenet3k_loc = imdb_from_imagenet3k_loc('./datasets/imagenet_3k', 'train');
assert(strcmp(imdb_imagenet3k_loc.name, 'imagenet3k_loc_train'));
fprintf('done\n');
fprintf('running statistics...');
[~, stat_3k] = imdb_bbox_class_statistics(imdb_imagenet3k_loc);
fprintf('done\n');
clear imdb_imagenet3k_loc;

%% load 10k imdb
fprintf('loading imdbs for 10k...');
imdb_imagenet10k_cls = imdb_from_imagenet10k_cls('./datasets/imagenet_10k', 'train');
assert(strcmp(imdb_imagenet10k_cls.name, 'imagenet10k_cls_train'));
fprintf('done\n');

fprintf('running statistics...');
[~, stat_10k] = imdb_bbox_class_statistics(imdb_imagenet10k_cls);
fprintf('done\n');
clear imdb_imagenet10k_cls;

%% save results
save statistics.mat -v7.3