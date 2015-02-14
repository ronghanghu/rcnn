load imdb/cache/all_ilsvrc.mat;
load imdb/cache/imdb_ilsvrc12_loc_train.mat;
load 10k_training/map_vec_200_1k_to_7k.mat;

imdb_ilsvrc13_det = [imdb_ilsvrc_val1 imdb_ilsvrc_val2 imdb_ilsvrc_train];
imdb_ilsvrc12_loc = imdb;
assert(strcmp(imdb_ilsvrc12_loc.name, 'imdb_ilsvrc12_loc_train'));

out_dir = '../external';

num_to_sample = 1000;
map_label = true(202, 1);
whole_im = false;

ending_index = rcnn_make_mapped_window_file(imdb_ilsvrc13_det, out_dir, 'mapped_200_to_7k', ...
    num_to_sample, map_label, whole_im, map_vec_200_to_7k, 0);
disp(ending_index);

map_label = true;
whole_im = false;
ending_index = rcnn_make_mapped_window_file(imdb_ilsvrc12_loc, out_dir, 'mapped_1k_to_7k', ...
    num_to_sample, map_label, whole_im, map_vec_1k_to_7k, ending_index);
disp(ending_index);