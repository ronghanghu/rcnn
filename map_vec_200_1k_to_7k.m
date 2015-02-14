load ../mhex_graph/+imagenet/meta_200.mat;
load ../mhex_graph/+imagenet/meta_1k.mat;
load ../mhex_graph/+imagenet/meta_7k.mat;

map_vec_200_to_7k = zeros(200, 1);
map_vec_1k_to_7k = zeros(1000, 1);

for v = 1:200
  WNID = synsets_200(v).WNID;
  try
    label = wnid2label_7k(WNID);
  catch
    label = -1;
  end
  map_vec_200_to_7k(v) = label;
end

for v = 1:1000
  WNID = synsets_1k(v).WNID;
  try
    label = wnid2label_7k(WNID);
  catch
    label = -1;
  end
  map_vec_1k_to_7k(v) = label;
end

map_vec_200_to_7k = [0; map_vec_200_to_7k];
map_vec_1k_to_7k = [0; map_vec_1k_to_7k];

save map_vec_200_1k_to_7k.mat map_vec_200_to_7k map_vec_1k_to_7k