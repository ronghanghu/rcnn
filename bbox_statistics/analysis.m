close all;
clc;

load trainval_boxes.mat;

pos = all_overlaps > 0.5;
ft = all_overlaps > 0.1;

all_w = all_boxes(:, 3) - all_boxes(:, 1) + 1;
all_h = all_boxes(:, 4) - all_boxes(:, 2) + 1;
all_l = min(all_w, all_h);
all_r = log(all_w ./ all_h);
std_all_r = std(all_r);

pos_w = all_w(pos, :);
pos_h = all_h(pos, :);
pos_l = min(pos_w, pos_h);
pos_r = log(pos_w ./ pos_h);
std_pos_r = std(pos_r);

ft_w = all_w(ft, :);
ft_h = all_h(ft, :);
ft_l = min(ft_w, ft_h);
ft_r = log(ft_w ./ ft_h);
std_ft_r = std(ft_r);

figure;
subplot(1, 3, 1); hist(pos_r, 100); title('pos boxes'); xlabel(num2str(std_pos_r));
subplot(1, 3, 2); hist(ft_r, 100); title('ft boxes'); xlabel(num2str(std_ft_r));
subplot(1, 3, 3); hist(all_r, 100); title('all boxes'); xlabel(num2str(std_all_r));

figure;
subplot(1, 3, 1); hist(all_l, 100); title('pos boxes'); xlabel(num2str(std(all_l)));
subplot(1, 3, 2); hist(pos_l, 100); title('ft boxes'); xlabel(num2str(std(pos_l)));
subplot(1, 3, 3); hist(ft_l, 100); title('all boxes'); xlabel(num2str(std(ft_l)));