function normalizer = fit_normalizer(features)
%FIT_NORMALIZER Learn z-score location/scale from training rows ONLY.
if isempty(features) || any(~isfinite(features(:)))
    error('expression:InvalidFeatures', 'Training features must be nonempty and finite.');
end
normalizer.mean = mean(features, 1);
normalizer.scale = sqrt(mean((features - normalizer.mean) .^ 2, 1));
normalizer.constantFeatureCount = sum(normalizer.scale < 1e-12);
normalizer.scale(normalizer.scale < 1e-12) = 1;
normalizer.trainingRowCount = size(features, 1);
end
