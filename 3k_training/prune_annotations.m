clear;

load meta_all;
im_path = '/y/mrowca/3Kbboxes/extracted';

% load the list of all annotated image filenames
im_list = textread('./3k_training/annotation_list_3k_before_prune.txt', '%s');

% arrange image filenames in cells corresponding to each wnid
% in wnids_3k
im_cell = cell(length(wnids_3k), 1);
skipped_num = 0;
for n_im = 1:length(im_list)
  im_name = im_list{n_im};
  wnid = im_name(1:9);
  
  if exist(fullfile(im_path, wnid, im_name), 'file')
    label = wnid2label_3k(wnid);
    im_cell{label} = cat(1, im_cell{label}, {im_name});
  else
    skipped_num = skipped_num + 1;
  end
end
fprintf('skipped %d non-existence images\n', skipped_num);

% prune the wnids
% conditions of keeping
% 1. in 10K leaf or internal
% 2. not in 1K leaf
% 3. not in 200 leaf
in_200 = false(length(wnids_3k), 1);
in_1k = false(length(wnids_3k), 1);
in_10k = false(length(wnids_3k), 1);
for n_wnid = 1:length(wnids_3k)
  wnid = wnids_3k{n_wnid};
  
  % 1. in 10K leaf or internal
  try 
    label_in_10k = wnid2label_7k(wnid);
  catch
    label_in_10k = 0;
  end
  
  % 2. not in 1K leaf
  try
    label_in_1k = wnid2label_1k(wnid);
  catch
    label_in_1k = 0;
  end
  
  % 3. not in 200 leaf
  try
    label_in_200 = wnid2label_200(wnid);
  catch
    label_in_200 = 0;
  end
  
  in_200(n_wnid) = (label_in_200 > 0 && label_in_200 <= 200);
  in_1k(n_wnid) = (label_in_1k > 0 && label_in_1k <= 1000);
  in_10k(n_wnid) = (label_in_10k > 0);
end
keep = in_10k & ~in_200 & ~in_1k;
wnids_3k_kept = wnids_3k(keep);
im_cell_kept = im_cell(keep);

% split into training and validation and test set 4:1:1
im_list_kept = {};
for n_wnid = 1:length(im_cell_kept)
  im_list_kept = cat(1, im_list_kept, im_cell_kept{n_wnid});
end
rng(3); % fix rand seed for repeatibility
rand_array = rand(length(im_list_kept), 1);
is_train = (rand_array < 4 / 6);
is_val = (rand_array > 5 / 6);
is_test = ~is_train & ~is_val;
im_list_train = im_list_kept(is_train);
im_list_val = im_list_kept(is_val);
im_list_test = im_list_kept(is_test);
fprintf('train, val, test image numbers: %d %d %d\n', ...
  length(im_list_train), ...
  length(im_list_val), ...
  length(im_list_test));

% write out splits into files
fid_train = fopen('./data/splits/3k_list_train.txt', 'w');
for m = 1:length(im_list_train)
  fprintf(fid_train, '%s\n', im_list_train{m});
end
fclose(fid_train);

fid_val = fopen('./data/splits/3k_list_val.txt', 'w');
for m = 1:length(im_list_val)
  fprintf(fid_val, '%s\n', im_list_val{m});
end
fclose(fid_val);

fid_test = fopen('./data/splits/3k_list_test.txt', 'w');
for m = 1:length(im_list_test)
  fprintf(fid_test, '%s\n', im_list_test{m});
end
fclose(fid_test);