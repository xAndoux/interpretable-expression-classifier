function [labels, scores] = predict_model(model, features)
%PREDICT_MODEL Predict visible-expression dataset labels, not inner emotions.
normalized = apply_normalizer(features, model.normalizer);
names = class_names();
switch model.type
    case 'nearest_centroid'
        distances = zeros(size(normalized, 1), numel(names));
        for c = 1:numel(names)
            delta = normalized - model.centroids(c, :);
            distances(:, c) = sum(delta .^ 2, 2);
        end
        scores = -distances;
    case 'ridge'
        scores = [normalized, ones(size(normalized, 1), 1)] * model.weights;
    otherwise
        error('expression:InvalidModel', 'Unsupported model type.');
end
[~, indices] = max(scores, [], 2);
labels = names(indices);
end
