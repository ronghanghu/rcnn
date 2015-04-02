% build a image list for 10k training
% For those classes that are in ImageNET 200 or ImageNET 1K, use bounding
% box. For other classes, use the whole image

% find 10k classes that are not in 200 or 1K, and create an image list of
% them
load external/mhex_graph/+imagenet/meta_7k.mat;
load external/mhex_graph/+imagenet/meta_1k.mat;
load external/mhex_graph/+imagenet/meta_200.mat;

class_num_10k = length(synsets_7k);
is_in_200_or_1k = true(class_num_10k, 1);
for v = 1:class_num_10k
  WNID = synsets_7k(v).WNID;
  try
    wnid2label_200(WNID);
    wnid2label_1k(WNID);
    is_in_200_or_1k(v) = true;
  catch err_msg
    if strcmp(err_msg.identifier, 'MATLAB:Containers:Map:NoKey')
      is_in_200_or_1k(v) = false;
    else
      error(err_msg);
    end
  end
end

keep_classes = ~is_in_200_or_1k;