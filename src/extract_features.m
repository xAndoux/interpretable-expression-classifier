function [features, names] = extract_features(image, config)
%EXTRACT_FEATURES Bottom-up spatial descriptors from an aligned grayscale face.
% Input must already be preprocessed. No semantic detector is used. Features
% retain coarse location: block means/stds, directional gradients, gradient
% magnitude, reflection error, and horizontal/vertical intensity profiles.
if nargin < 2 || isempty(config)
    config = default_config();
end
validateattributes(image, {'numeric'}, {'2d', 'finite', 'nonempty'});
if ~isequal(size(image), config.imageSize)
    error('expression:InvalidSize', 'Image size must equal config.imageSize.');
end
image = double(image);
[rows, columns] = size(image);
rowEdges = round(linspace(1, rows + 1, config.blockGrid(1) + 1));
columnEdges = round(linspace(1, columns + 1, config.blockGrid(2) + 1));
blockCount = prod(config.blockGrid);
blockMean = zeros(1, blockCount);
blockStd = zeros(1, blockCount);
blockGx = zeros(1, blockCount);
blockGy = zeros(1, blockCount);
blockMagnitude = zeros(1, blockCount);
blockSymmetry = zeros(1, blockCount);
gx = zeros(size(image));
gy = zeros(size(image));
gx(:, 2:end-1) = (image(:, 3:end) - image(:, 1:end-2)) / 2;
gx(:, 1) = image(:, 2) - image(:, 1);
gx(:, end) = image(:, end) - image(:, end-1);
gy(2:end-1, :) = (image(3:end, :) - image(1:end-2, :)) / 2;
gy(1, :) = image(2, :) - image(1, :);
gy(end, :) = image(end, :) - image(end-1, :);
magnitude = sqrt(gx .^ 2 + gy .^ 2);
symmetry = abs(image - image(:, end:-1:1));
index = 0;
for r = 1:config.blockGrid(1)
    for c = 1:config.blockGrid(2)
        index = index + 1;
        ri = rowEdges(r):rowEdges(r+1)-1;
        ci = columnEdges(c):columnEdges(c+1)-1;
        blockMean(index) = mean2(image(ri, ci));
        blockStd(index) = std2(image(ri, ci));
        blockGx(index) = mean2(gx(ri, ci));
        blockGy(index) = mean2(gy(ri, ci));
        blockMagnitude(index) = mean2(magnitude(ri, ci));
        blockSymmetry(index) = mean2(symmetry(ri, ci));
    end
end

profileCount = 16;
rowMean = profile_statistic(image, profileCount, 1, false);
rowStd = profile_statistic(image, profileCount, 1, true);
columnMean = profile_statistic(image, profileCount, 2, false);
features = [blockMean, blockStd, blockGx, blockGy, blockMagnitude, ...
            blockSymmetry, rowMean, rowStd, columnMean];
if nargout > 1
    kinds = {'block_mean', 'block_std', 'gradient_x', 'gradient_y', ...
             'gradient_magnitude', 'reflection_error'};
    names = cell(1, numel(features));
    offset = 0;
    for k = 1:numel(kinds)
        for index = 1:blockCount
            names{offset + index} = sprintf('%s_block_%02d', kinds{k}, index);
        end
        offset = offset + blockCount;
    end
    names(offset + (1:profileCount)) = numbered_names('row_mean', profileCount);
    offset = offset + profileCount;
    names(offset + (1:profileCount)) = numbered_names('row_std', profileCount);
    offset = offset + profileCount;
    names(offset + (1:profileCount)) = numbered_names('column_mean', profileCount);
end
end

function result = mean2(x)
result = mean(x(:));
end

function result = std2(x)
x = x(:);
result = sqrt(mean((x - mean(x)) .^ 2));
end

function result = profile_statistic(image, count, dimension, useStd)
edges = round(linspace(1, size(image, dimension) + 1, count + 1));
result = zeros(1, count);
for k = 1:count
    if dimension == 1
        values = image(edges(k):edges(k+1)-1, :);
    else
        values = image(:, edges(k):edges(k+1)-1);
    end
    if useStd
        result(k) = mean(std(values, 0, 2));
    else
        result(k) = mean(values(:));
    end
end
end

function names = numbered_names(prefix, count)
names = cell(1, count);
for k = 1:count
    names{k} = sprintf('%s_%02d', prefix, k);
end
end
