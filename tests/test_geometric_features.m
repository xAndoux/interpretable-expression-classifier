function test_geometric_features()
%TEST_GEOMETRIC_FEATURES Verify the interpretable geometric layer measures the
% facial actions it claims, using synthetic 68-point landmark layouts that
% follow facial-action geometry (not real faces). These assertions check the
% MEASUREMENTS, not facial-expression accuracy.

config = default_config();
dataset = make_synthetic_landmark_dataset(config);
assert(dataset.syntheticOnly);
assert(size(dataset.landmarks, 2) == numel(dataset.labels));
assert(all(cellfun(@(lm) isequal(size(lm), [68, 2]), dataset.landmarks)));
assert(numel(dataset.featureNames) == size(dataset.features, 2));
assert(all(isfinite(dataset.features(:))));

% Sanity: the geometric feature extractor is deterministic and aligned.
[firstFeatures, firstNames] = extract_geometric_features(dataset.landmarks{1});
assert(isequal(firstNames, dataset.featureNames));
assert_close(firstFeatures, dataset.features(1, :), 1e-9);

% Helper to grab one synthetic face per class by its label.
function feats = feat_for(ds, label)
    idx = find(strcmp(ds.labels, label), 1);
    feats = ds.features(idx, :);
end

% Pick the human-readable proxies: happy vs sad lip curvature.
happy = feat_for(dataset, 'happy');
sad = feat_for(dataset, 'sad');
neutral = feat_for(dataset, 'neutral');
surprise = feat_for(dataset, 'surprise');
angry = feat_for(dataset, 'angry');

% lip_curvature is column 12; smile_proxy is column 13.
assert(happy(12) > sad(12), 'Happy should have a higher lip curvature than sad.');
assert(happy(13) > sad(13), 'Happy should have a higher smile proxy than sad.');
assert(sad(12) < neutral(12), 'Sad should pull mouth corners down vs neutral.');

% surprise: higher brow height (cols 1-2), wider eyes (6-7), more mouth open (11).
assert(surprise(1) > neutral(1) + 0.02, 'Surprise should raise the brows.');
assert(surprise(7) > neutral(7) + 0.02, 'Surprise should widen the eyes.');
assert(surprise(11) > neutral(11) + 0.02, 'Surprise should open the mouth.');

% angry: lower brow height, narrower eyes than neutral.
assert(angry(1) < neutral(1), 'Angry should lower the brows.');
assert(angry(7) < neutral(7), 'Angry should narrow the eyes.');
assert(angry(5) < neutral(5), 'Angry should pull brows together (smaller inter-brow).');

% The proxies are finite interpretable scalars; nothing NaN.
assert(all(isfinite([happy(13), surprise(14), angry(15)])));
end