function [imdb, roidb] = load_imdb_roidb(imdb_name)

fprintf('loading imdb and roidb %s...', imdb_name);

imdb = load(['imdb/cache/imdb_' imdb_name '.mat']);
imdb = imdb.imdb;
roidb = load(['imdb/cache/roidb_' imdb_name '.mat']);
roidb = roidb.roidb;

fprintf('done\n');

end

