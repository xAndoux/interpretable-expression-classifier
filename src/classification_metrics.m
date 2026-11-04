function metrics = classification_metrics(trueLabels, predictedLabels)
%CLASSIFICATION_METRICS Confusion matrix and per-class/macro measurements.
% Rows are true classes; columns are predicted classes. Undefined precision
% or recall is reported as zero (sklearn-style zero_division=0).
names = class_names();
if numel(trueLabels) ~= numel(predictedLabels) || isempty(trueLabels)
    error('expression:InvalidLabels', 'Label vectors must be nonempty and equal in length.');
end
metrics.classNames = names;
metrics.confusion = zeros(numel(names));
for k = 1:numel(trueLabels)
    actual = find(strcmp(names, trueLabels{k}));
    predicted = find(strcmp(names, predictedLabels{k}));
    if isempty(actual) || isempty(predicted)
        error('expression:InvalidLabels', 'Unknown class label.');
    end
    metrics.confusion(actual, predicted) = metrics.confusion(actual, predicted) + 1;
end
metrics.precision = zeros(1, numel(names));
metrics.recall = zeros(1, numel(names));
metrics.f1 = zeros(1, numel(names));
metrics.support = sum(metrics.confusion, 2)';
for c = 1:numel(names)
    tp = metrics.confusion(c, c);
    predictedCount = sum(metrics.confusion(:, c));
    if predictedCount > 0
        metrics.precision(c) = tp / predictedCount;
    end
    if metrics.support(c) > 0
        metrics.recall(c) = tp / metrics.support(c);
    end
    if metrics.precision(c) + metrics.recall(c) > 0
        metrics.f1(c) = 2 * metrics.precision(c) * metrics.recall(c) / ...
            (metrics.precision(c) + metrics.recall(c));
    end
end
metrics.macroF1 = mean(metrics.f1);
metrics.accuracy = sum(diag(metrics.confusion)) / numel(trueLabels);
metrics.sampleCount = numel(trueLabels);
end
