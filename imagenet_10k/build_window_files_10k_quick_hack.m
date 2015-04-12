% the quick hacky code to append 10K window files to an existing window file

% existing window file
window_file_name = '../window_file_mapped_10k_to_10k_train.txt';
% starting image index in window file
start_index = % change to your start index % ;

image_dir = './datasets/imagenet_10k/train';
image_list_file = './datasets/imagenet_10k/10k_list_train.txt';

%% -----------------------------------------------------------------------------

load external/mhex_graph/+imagenet/meta_extended.mat;
load external/mhex_graph/+imagenet/meta_1k.mat;
load external/mhex_graph/+imagenet/meta_200.mat;

image_ids = textread(image_list_file, '%s');

% keep those classes not in 200 or 1K and remove invalid images (filesize == 0)
% attach class labels
labels = zeros(length(image_ids), 1);
keep_im = false(length(image_ids), 1);
for n = 1:length(image_ids)
  tic_toc_print('preprocessing %d / %d\n', n, length(image_ids));
  wnid = image_ids{n}(1:9);
  labels(n) = wnid2label_extended(wnid);
  keep_im(n) = ~wnid2label_200.isKey(wnid) && ~wnid2label_1k.isKey(wnid);
end
fprintf('keeping %d out of %d\n', sum(keep_im), length(image_ids));
image_ids = image_ids(keep_im);
labels = labels(keep_im);
assert(length(image_ids) == length(labels));

IM_SIZE = 256;
channels = 3;
num_boxes = 1;
overlap = 1;
bboxes = [0, 0, IM_SIZE-1, IM_SIZE-1]; % whole image as bbox

% write window files
image_index = start_index;
% APPEND TO EXISTING WINDOW FILE
fid = fopen(window_file_name, 'a');
% start a new line before writing (not necessary since all window file
% end with '\n')
% fprintf(fid, '\n');
numimages = length(image_ids);
for n = 1:numimages
  tic_toc_print('writing %d / %d\n', n, numimages);
  %     # image_index
  %     img_path
  %     channels
  %     height
  %     width
  %     num_windows
  %     class_index overlap x1 y1 x2 y2
  fprintf(fid, '# %d\n', image_index);
  fprintf(fid, '%s/%s\n', image_dir, image_ids{n});
  fprintf(fid, '%d\n%d\n%d\n', channels, IM_SIZE, IM_SIZE);
  fprintf(fid, '%d\n', num_boxes);
  
  data = [labels(n) overlap bboxes];
  fprintf(fid, '%d %.3f %d %d %d %d\n', data');
  image_index = image_index + 1;
end
fclose(fid);
