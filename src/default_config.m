function config = default_config()
%DEFAULT_CONFIG Fixed preprocessing, feature, split, and candidate settings.
config.imageSize = [64, 64];
config.blockGrid = [6, 6];
config.intensityClip = 5;
config.validationFraction = 0.2;
config.testFraction = 0.2;
config.seed = 2024;
config.ridgeLambdas = [0.01, 0.1, 1, 10, 100];
config.minimumImagesPerClass = 5;
end
