function feat = spp_features(im, boxes, rcnn_model)

% PARAMETER OF THE NETWORK
% TODO: move these parameters into rcnn_model
% NOTE: if you change any of these parameters, you must also change the
% corresponding network prototext file
conv5_stride = 16;
max_proposal_num = 2500;
% 5 Scale
fixed_sizes = [640, 768, 917, 1152, 1600]';
conv5_sizes = [ 38,  46,  56,   70,   98]'; % Zeiler & Fergus net
% 1 Scale
% fixed_sizes = [917]';
% conv5_sizes = [ 56]'; % Zeiler & Fergus net

feat = spp_features_forward(im, boxes, rcnn_model, fixed_sizes, ...
    conv5_sizes, conv5_stride, max_proposal_num);

end
