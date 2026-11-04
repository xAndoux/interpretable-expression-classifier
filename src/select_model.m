function [bestModel, selection] = select_model(trainFeatures, trainLabels, validationFeatures, validationLabels, config)
%SELECT_MODEL Use validation macro F1 only; never inspect the held-out test set.
candidates = cell(1, 1 + numel(config.ridgeLambdas));
candidates{1} = struct('type', 'nearest_centroid', 'lambda', NaN, ...
    'featureNames', []);
for k = 1:numel(config.ridgeLambdas)
    candidates{k+1} = struct('type', 'ridge', 'lambda', config.ridgeLambdas(k), ...
        'featureNames', []);
end
results = repmat(struct('index', 0, 'type', '', 'lambda', NaN, 'validationMacroF1', NaN, ...
    'validationAccuracy', NaN), numel(candidates), 1);
bestScore = -Inf;
bestIndex = 0;
for k = 1:numel(candidates)
    model = train_candidate(trainFeatures, trainLabels, candidates{k});
    predicted = predict_model(model, validationFeatures);
    metrics = classification_metrics(validationLabels, predicted);
    results(k).index = k;
    results(k).type = candidates{k}.type;
    results(k).lambda = candidates{k}.lambda;
    results(k).validationMacroF1 = metrics.macroF1;
    results(k).validationAccuracy = metrics.accuracy;
    % Candidate order deliberately resolves exact ties: centroid then smaller lambda.
    if metrics.macroF1 > bestScore
        bestScore = metrics.macroF1;
        bestIndex = k;
        bestModel = model;
    end
end
if bestIndex == 0
    error('expression:InvalidModel', 'No candidate could be fitted.');
end
selection.candidates = results;
selection.selectedCandidateIndex = bestIndex;
selection.rule = ['Maximum validation macro F1; ties resolved by candidate order ' ...
    '(nearest centroid, then increasing ridge lambda). Test labels are not used.'];
end
