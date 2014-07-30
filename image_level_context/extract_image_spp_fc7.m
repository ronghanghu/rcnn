function feat = extract_image_spp_fc7(im, rcnn_model)

[h, w, ~] = size(im);

whole_image = [1, 1, w, h]; % [x1 y1 x2 y2], 1-indexed
feat = spp_features_1_scale_(im, whole_image, rcnn_model);

end
