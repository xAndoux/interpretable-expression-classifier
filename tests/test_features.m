function test_features()
config = default_config();
[x, y] = meshgrid(linspace(-1, 1, 64));
image = sin(3 * x) + 0.2 * cos(4 * y);
[features, names] = extract_features(image, config);
assert(numel(features) == 264);
assert(numel(names) == numel(features));
assert(all(isfinite(features)));
assert_close(extract_features(image, config), features);
% Coarse spatial location is represented, rather than a global histogram.
moved = circshift(image, [12, 0]);
assert(norm(features - extract_features(moved, config)) > 1e-3);
% First block mean matches the actual rounded 6x6 grid.
expected = mean(mean(image(1:11, 1:11)));
assert_close(features(1), expected);
end
