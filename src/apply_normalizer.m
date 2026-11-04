function normalized = apply_normalizer(features, normalizer)
%APPLY_NORMALIZER Apply previously learned training statistics without fitting.
if size(features, 2) ~= numel(normalizer.mean)
    error('expression:InvalidFeatures', 'Feature width differs from fitted normalizer.');
end
normalized = (features - normalizer.mean) ./ normalizer.scale;
end
