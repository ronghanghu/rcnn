function rcnn_model = rcnn_model_from_mhex(imdb, net_file)

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParamValue('layer',           7,      @isscalar);
ip.addParamValue('crop_mode',       'warp', @isstr);
ip.addParamValue('crop_padding',    16,     @isscalar);
ip.parse(imdb);
opts = ip.Results;

opts.net_def_file = 'model-defs/step_2_det_deploy.prototxt';
opts.net_file = net_file;

conf = rcnn_config('sub_dir', imdb.name);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% ------------------------------------------------------------------------
% Create a new rcnn model
rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file);
rcnn_model = rcnn_load_model(rcnn_model, conf.use_gpu);
rcnn_model.detectors.crop_mode = opts.crop_mode;
rcnn_model.detectors.crop_padding = opts.crop_padding;
rcnn_model.classes = imdb.classes;
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Keep the default norm for no-scaling
opts.feat_norm_mean = 20;
rcnn_model.training_opts = opts;
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% create classifiers by net surgery on fc8 and mhex_mat1 layer
assert(strcmp(rcnn_model.cnn.layers(8).layer_names, 'fc8_finetune_bg'));
fc8_W_bg = rcnn_model.cnn.layers(8).weights{1};
fc8_B_bg = rcnn_model.cnn.layers(8).weights{2}';

assert(strcmp(rcnn_model.cnn.layers(9).layer_names, 'fc8_finetune_200_leaf'));
fc8_W_leaf = rcnn_model.cnn.layers(9).weights{1};
fc8_B_leaf = rcnn_model.cnn.layers(9).weights{2}';

assert(strcmp(rcnn_model.cnn.layers(10).layer_names, 'fc8_finetune_271_internal'));
fc8_W_internal = rcnn_model.cnn.layers(10).weights{1};
fc8_B_internal = rcnn_model.cnn.layers(10).weights{2}';

assert(strcmp(rcnn_model.cnn.layers(11).layer_names, 'mhex_mat1'));
mhex_mat1 = rcnn_model.cnn.layers(11).weights{1};

% subtract the background scores from every class score
% the background class is the first class
W = bsxfun(@minus, [fc8_W_leaf, fc8_W_internal] * mhex_mat1, fc8_W_bg);
B = bsxfun(@minus, [fc8_B_leaf, fc8_B_internal] * mhex_mat1, fc8_B_bg);

rcnn_model.detectors.W = W;
rcnn_model.detectors.B = B;

end

