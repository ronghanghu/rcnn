function [res_test, res_train] = rcnn_exp_test_cccp_rcnn()
% Runs an experiment that trains an R-CNN model and tests it.

% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
% ------------------------------------------------

cache_name = 'cccp3_rcnn';

imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');
imdb_test = imdb_from_voc(VOCdevkit, 'test', '2007');

% res_train = cccp_rcnn_test(cache_name, imdb_train);
res_test = cccp_rcnn_test(cache_name, imdb_test);
