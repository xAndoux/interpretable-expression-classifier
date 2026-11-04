function model = train_candidate(features, labels, candidate)
%TRAIN_CANDIDATE Fit centroid or ridge baseline from normalized training data.
% Ridge W = (Z'Z + lambda I) \ (Z'Y), with unregularized intercept.
% Uses only base MATLAB matrix algebra; no machine-learning toolboxes.
normalizer = fit_normalizer(features);
z = apply_normalizer(features, normalizer);
names = class_names();
y = zeros(size(z, 1), numel(names));
for c = 1:numel(names)
    y(:, c) = strcmp(labels(:), names{c});
end
model.type = candidate.type;
model.normalizer = normalizer;
model.featureNames = candidate.featureNames;
model.classNames = names;
switch candidate.type
    case 'nearest_centroid'
        model.centroids = zeros(numel(names), size(z, 2));
        for c = 1:numel(names)
            if ~any(y(:, c))
                error('expression:MissingTrainingClass', 'Training split has no examples for %s.', names{c});
            end
            model.centroids(c, :) = mean(z(y(:, c) == 1, :), 1);
        end
        model.description = 'Nearest class centroid in train-normalized feature space.';
    case 'ridge'
        if ~isscalar(candidate.lambda) || ~isfinite(candidate.lambda) || candidate.lambda <= 0
            error('expression:InvalidModel', 'Ridge lambda must be a positive finite scalar.');
        end
        design = [z, ones(size(z, 1), 1)];
        penalty = eye(size(design, 2));
        penalty(end, end) = 0;
        model.weights = (design' * design + candidate.lambda * penalty) \ (design' * y);
        model.lambda = candidate.lambda;
        model.description = 'One-vs-rest ridge regression with unpenalized intercept.';
    otherwise
        error('expression:InvalidModel', 'Unsupported candidate model.');
end
end
