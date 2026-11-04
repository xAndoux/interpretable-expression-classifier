function test_training()
config = default_config();
names = class_names();
features = [];
labels = {};
for c = 1:7
    features = [features; repmat([c, -c, 0], 4, 1) + (0:3)' * [0.01, 0.01, 0]]; %#ok<AGROW>
    labels = [labels, repmat(names(c), 1, 4)]; %#ok<AGROW>
end
candidate = struct('type', 'nearest_centroid', 'lambda', NaN, 'featureNames', {{}});
centroid = train_candidate(features, labels, candidate);
predicted = predict_model(centroid, features + 0.001);
assert(all(strcmp(predicted(:)', labels)));

% Ridge needs a nondegenerate fixture. Global z-scoring makes a single
% ordered scalar feature perfectly symmetric across seven classes: argmax
% regression scores cannot recover all class identities from that one axis.
z = [(-1.75:0.5:1.75)', [1; -1; 0.5; -0.5; -1; 1; -0.5; 0.5]];
ridgeLabels = names([ones(1,4), 7*ones(1,4)]);
ridge = train_candidate(z, ridgeLabels, struct('type', 'ridge', 'lambda', 0.1, 'featureNames', {{}}));
ridgePredicted = predict_model(ridge, z);
assert(all(strcmp(ridgePredicted(:)', ridgeLabels)));
assert(all(isfinite(ridge.weights(:))));
% Independently verify ordinary matrix-algebra solution (no toolbox).
penalty = eye(3); penalty(end, end) = 0;
y = zeros(8, 7); y(1:4, 1) = 1; y(5:8, 7) = 1;
normalized = apply_normalizer(z, ridge.normalizer);
design = [normalized, ones(8,1)];
expected = (design' * design + 0.1 * penalty) \ (design' * y);
assert_close(ridge.weights, expected);

[model, selection] = select_model(features, labels, features, labels, config);
assert(isfield(model, 'normalizer'));
assert(numel(selection.candidates) == 1 + numel(config.ridgeLambdas));
assert(selection.selectedCandidateIndex >= 1);
end
