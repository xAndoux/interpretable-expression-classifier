function dataset = make_synthetic_dataset(config)
%MAKE_SYNTHETIC_DATASET Generate abstract patterns to exercise code paths ONLY.
% These are not faces and contain no emotion ground truth. Any metrics are
% synthetic-pattern execution measurements, never facial-expression accuracy.
names = class_names();
perClass = 15;
rows = config.imageSize(1);
columns = config.imageSize(2);
[x, y] = meshgrid(linspace(-1, 1, columns), linspace(-1, 1, rows));
n = numel(names) * perClass;
images = zeros(rows, columns, n);
labels = cell(1, n);
paths = cell(1, n);
groups = repmat({''}, 1, n);
for c = 1:numel(names)
    for k = 1:perClass
        index = (c - 1) * perClass + k;
        phase = 2 * pi * (k - 1) / perClass;
        shift = 0.12 * sin(phase);
        switch c
            case 1
                image = sin(8 * (x + 0.4 * y) + phase);
            case 2
                image = cos(8 * (x - 0.4 * y) + phase);
            case 3
                image = sin(7 * x + phase) .* cos(5 * y);
            case 4
                image = cos(9 * sqrt((x-shift).^2 + y.^2) + phase);
            case 5
                image = sin(5 * y + phase) + 0.4 * cos(3 * x);
            case 6
                image = (abs(x-shift) < 0.35) + (abs(y) < 0.2);
            case 7
                image = x .* y + 0.6 * cos(4 * x + phase);
        end
        image = image + 0.06 * sin(17 * x + phase) .* cos(13 * y - phase);
        image = 0.5 + 0.22 * image;
        images(:, :, index) = image;
        labels{index} = names{c};
        paths{index} = sprintf('synthetic/%s_pattern_%02d', names{c}, k);
    end
end
processed = zeros(size(images));
for k = 1:n
    processed(:, :, k) = preprocess_image(images(:, :, k), config);
end
[firstFeatures, featureNames] = extract_features(processed(:, :, 1), config);
features = zeros(n, numel(firstFeatures));
features(1, :) = firstFeatures;
keys = cell(1, n);
for k = 1:n
    features(k, :) = extract_features(processed(:, :, k), config);
    keys{k} = content_fingerprint(processed(:, :, k));
end
dataset.paths = paths;
dataset.labels = labels;
dataset.groups = groups;
dataset.features = features;
dataset.featureNames = featureNames;
dataset.contentKeys = keys;
dataset.identityProvided = false;
dataset.root = '';
dataset.config = config;
dataset.syntheticOnly = true;
end
