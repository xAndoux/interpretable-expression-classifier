function image = preprocess_image(source, config)
%PREPROCESS_IMAGE Convert an aligned crop to robustly normalized grayscale.
% SOURCE is an image path or numeric/integer image array. No face detection
% or alignment is performed. Double inputs must already be in [0, 1].
if nargin < 2 || isempty(config)
    config = default_config();
end
if ischar(source) || (isscalar(source) && isstring(source))
    source = imread(char(source));
end
if ~isnumeric(source) && ~islogical(source)
    error('expression:InvalidImage', 'Image input must be numeric or logical.');
end
if ndims(source) == 3
    if size(source, 3) < 3
        source = source(:, :, 1);
    else
        % Fixed Rec. 601 luma weights; avoid rgb2gray toolbox dependency.
        source = 0.299 * double(source(:, :, 1)) + ...
                 0.587 * double(source(:, :, 2)) + ...
                 0.114 * double(source(:, :, 3));
    end
elseif ndims(source) ~= 2
    error('expression:InvalidImage', 'Expected a 2-D grayscale or 3-D color image.');
end
if isempty(source) || any(~isfinite(double(source(:))))
    error('expression:InvalidImage', 'Image is empty or contains non-finite values.');
end
if isinteger(source) || islogical(source)
    if islogical(source)
        image = double(source);
    else
        image = double(source) / double(intmax(class(source)));
    end
else
    image = double(source);
    if any(image(:) < 0) || any(image(:) > 1)
        error('expression:InvalidRange', 'Floating-point image values must be in [0, 1].');
    end
end
image = resize_bilinear(image, config.imageSize);
ordered = sort(image(:));
medianValue = percentile_sorted(ordered, 0.5);
scale = percentile_sorted(ordered, 0.75) - percentile_sorted(ordered, 0.25);
if scale < 1e-6
    scale = sqrt(mean((image(:) - mean(image(:))) .^ 2));
end
if ~isfinite(scale) || scale < 1e-8
    error('expression:ConstantImage', 'A constant image has no usable spatial contrast.');
end
image = (image - medianValue) / scale;
image = max(min(image, config.intensityClip), -config.intensityClip);
image = image / config.intensityClip;
end

function value = percentile_sorted(ordered, fraction)
position = 1 + (numel(ordered) - 1) * fraction;
lower = floor(position);
upper = ceil(position);
weight = position - lower;
value = ordered(lower) * (1 - weight) + ordered(upper) * weight;
end
