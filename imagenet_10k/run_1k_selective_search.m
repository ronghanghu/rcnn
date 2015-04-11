clear;

load imdb/cache/imdb/cache/imdb_ilsvrc12_loc_train.mat
[images, boxes] = selective_search_boxes_imdb(imdb);
fprintf('saving to file...\n');
save_path = './data/selective_search_data';
save(fullfile(save_path, imdb.name), 'boxes', 'images');
fprintf('done\n');

clear;

load imdb/cache/imdb/cache/imdb_ilsvrc12_loc_val.mat
[images, boxes] = selective_search_boxes_imdb(imdb);
fprintf('saving to file...\n');
save_path = './data/selective_search_data';
save(fullfile(save_path, imdb.name), 'boxes', 'images');
fprintf('done\n');

clear;