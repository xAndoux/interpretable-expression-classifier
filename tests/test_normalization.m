function test_normalization()
train = [1, 10; 3, 10; 5, 10];
normalizer = fit_normalizer(train);
assert_close(normalizer.mean, [3, 10]);
assert_close(normalizer.scale(2), 1);
assert(normalizer.constantFeatureCount == 1);
assert_close(apply_normalizer(train, normalizer), [-sqrt(3/2), 0; 0, 0; sqrt(3/2), 0]);
% Validation/test outliers must not change training statistics or get clipped.
test = apply_normalizer([101, 1000], normalizer);
assert_close(test, [98*sqrt(3/8), 990]);
assert_close(fit_normalizer(train).mean, [3, 10]);
end
