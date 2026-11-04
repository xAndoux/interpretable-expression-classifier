function results = run_experiment(dataset, config)
%RUN_EXPERIMENT Fit, select on validation, and evaluate the untouched test split.
if ~isstruct(dataset) || ~isfield(dataset, 'features') || ~isfield(dataset, 'labels')
    error('expression:InvalidDataset', 'Expected a structure returned by LOAD_DATASET or MAKE_SYNTHETIC_DATASET.');
end
split = stratified_split(dataset.labels, dataset.groups, config);
train = strcmp(split, 'train');
validation = strcmp(split, 'validation');
test = strcmp(split, 'test');
if ~all(train | validation | test) || any(train & validation) || any(train & test) || any(validation & test)
    error('expression:SplitFailure', 'Partitions must be exhaustive and disjoint.');
end
[model, selection] = select_model(dataset.features(train, :), dataset.labels(train), ...
    dataset.features(validation, :), dataset.labels(validation), config);
trainPredicted = predict_model(model, dataset.features(train, :));
validationPredicted = predict_model(model, dataset.features(validation, :));
testPredicted = predict_model(model, dataset.features(test, :));
results.model = model;
results.selection = selection;
results.split = split;
results.trainIndices = find(train);
results.validationIndices = find(validation);
results.testIndices = find(test);
results.paths = dataset.paths;
results.labels = dataset.labels;
results.groups = dataset.groups;
results.trainMetrics = classification_metrics(dataset.labels(train), trainPredicted);
results.validationMetrics = classification_metrics(dataset.labels(validation), validationPredicted);
results.testMetrics = classification_metrics(dataset.labels(test), testPredicted);
results.testPredicted = testPredicted;
results.dataset = dataset;
results.config = config;
results.evaluationNote = ['This is an execution/evaluation of this new implementation on the supplied ' ...
    'dataset only. It is not a reproduction of the historical project or its reported accuracy.'];
end
