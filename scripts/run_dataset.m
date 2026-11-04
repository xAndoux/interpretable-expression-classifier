function results = run_dataset(datasetRoot, identityFile)
%RUN_DATASET Evaluate a user-supplied seven-class aligned face-crop dataset.
% Usage: run_dataset('/path/to/dataset')
%        run_dataset('/path/to/dataset', '/path/to/identity_map.csv')
% No dataset is downloaded or bundled. Read README.md before interpreting.
if nargin < 1 || isempty(datasetRoot)
    error('expression:MissingDataset', 'Supply the dataset root directory.');
end
if nargin < 2
    identityFile = '';
end
here = fileparts(mfilename('fullpath'));
project = fileparts(here);
addpath(fullfile(project, 'src'));
config = default_config();
dataset = load_dataset(datasetRoot, config, identityFile);
results = run_experiment(dataset, config);
output = fullfile(project, 'results');
if exist(output, 'dir') ~= 7
    mkdir(output);
end
if isempty(identityFile)
    description = sprintf('User-supplied aligned crops at %s; identity metadata not provided.', datasetRoot);
else
    description = sprintf('User-supplied aligned crops at %s; identity map %s.', datasetRoot, identityFile);
end
write_report(results, fullfile(output, 'dataset-report.md'), description);
save_results(results, fullfile(output, 'dataset-results.mat'));
fprintf('Held-out test accuracy: %.6f; macro F1: %.6f\n', ...
    results.testMetrics.accuracy, results.testMetrics.macroF1);
fprintf('Report: %s\n', fullfile(output, 'dataset-report.md'));
end
