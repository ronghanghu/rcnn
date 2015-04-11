function [images, boxes] = selective_search_boxes_imdb(imdb)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

fast_mode = true;

% % ------------------------------------------------------------------------
% % distributed computing configurations
% addpath(genpath('rootdir/simple-cluster-lib'));
% dconf                 = simple_cluster_lib_config();
% dconf.cd              = pwd();
% dconf.local           = false;
% dconf.dist_nodes      = 40;
% dconf.hours           = 12;
% dconf.cput            = 60*60*24;
% dconf.cleanup         = false;
% %dconf.resume = true;
% %dconf.work_dir_suffix = [testset '_' year];
% % ------------------------------------------------------------------------

im_width = 500;
boxes = op_selective_search_boxes(1, length(imdb.image_ids), imdb, im_width);
% mimic selective search output variable names

images = imdb.image_ids;
