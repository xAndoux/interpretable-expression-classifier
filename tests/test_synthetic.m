function test_synthetic()
config = default_config();
dataset = make_synthetic_dataset(config);
assert(dataset.syntheticOnly);
assert(size(dataset.features, 1) == 105);
assert(numel(dataset.contentKeys) == numel(unique(dataset.contentKeys)));
first = run_experiment(dataset, config);
second = run_experiment(dataset, config);
assert(isequal(first.split, second.split));
assert_close(first.testMetrics.confusion, second.testMetrics.confusion);
assert(first.model.normalizer.trainingRowCount == sum(strcmp(first.split, 'train')));
% Directly recompute from train rows to prove no full-dataset normalization.
expected = fit_normalizer(dataset.features(first.trainIndices, :));
assert_close(first.model.normalizer.mean, expected.mean);
assert_close(first.model.normalizer.scale, expected.scale);
assert(first.testMetrics.sampleCount == 21);
assert(first.validationMetrics.sampleCount == 21);
end
