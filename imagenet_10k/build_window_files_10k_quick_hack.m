% the quick hacky code to append 10K window files to an existing window file

% existing window file
window_file_name = '../window_file_mapped_10k_to_10k_hack.txt';
% starting image index in window file
start_index = 635067;

%% -----------------------------------------------------------------------------

load external/mhex_graph/+imagenet/meta_7k.mat;
load external/mhex_graph/+imagenet/meta_1k.mat;
load external/mhex_graph/+imagenet/meta_200.mat;

image_dir = './datasets/imagenet_10k/train';
iminfo_all = dir([image_dir '/*.jpg']);

% keep those classes not in 200 or 1K and remove invalid images (filesize == 0)
% attach class labels
image_names = {iminfo_all.name};
labels = zeros(length(iminfo_all), 1);
keep_im = false(length(iminfo_all), 1);
for n = 1:length(iminfo_all)
  tic_toc_print('preprocessing %d / %d\n', n, length(iminfo_all));
  wnid = iminfo_all(n).name(1:9);
  labels(n) = wnid2label_7k(wnid);
  keep_im(n) = (iminfo_all(n).bytes > 0) ...
    && ~wnid2label_200.isKey(wnid) ...
    && ~wnid2label_1k.isKey(wnid);
end
fprintf('keeping %d out of %d\n', sum(keep_im), length(iminfo_all));
iminfo_all = iminfo_all(keep_im);
image_names = image_names(keep_im);
labels = labels(keep_im);
assert(length(image_names) == length(labels));

image_names = {iminfo_all.name};
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
numimages = length(image_names);
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
  fprintf(fid, '%s/%s\n', image_dir, image_names{n});
  fprintf(fid, '%d\n%d\n%d\n', channels, IM_SIZE, IM_SIZE);
  fprintf(fid, '%d\n', num_boxes);
  
  data = [labels(n) overlap bboxes];
  fprintf(fid, '%d %.3f %d %d %d %d\n', data');
  image_index = image_index + 1;
end
fclose(fid);
