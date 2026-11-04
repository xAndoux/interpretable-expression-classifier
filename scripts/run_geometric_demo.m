function results = run_geometric_demo()
%RUN_GEOMETRIC_DEMO Prove the interpretable geometric (landmark) pipeline runs.
% Uses synthetic 68-point landmark layouts that follow facial-action geometry.
% These are NOT face images and carry NO real ground truth, so the reported
% numbers are execution measurements, never facial-expression accuracy.
here = fileparts(mfilename('fullpath'));
project = fileparts(here);
addpath(fullfile(project, 'src'));
output = fullfile(project, 'results');
if exist(output, 'dir') ~= 7
    mkdir(output);
end
config = default_config();
dataset = make_synthetic_landmark_dataset(config);
results = run_experiment(dataset, config);
description = ['Synthetic 68-point landmark geometry following facial actions ' ...
    '(brows raised/lowered, eyes widened/narrowed, mouth open, lips up/down). ' ...
    'This run proves the geometric-feature layer produces separable interpretable ' ...
    'measurements and that the classifier executes. It is NOT facial-expression ' ...
    'recognition accuracy and uses no face images and no real ground truth.'];
write_report(results, fullfile(output, 'geometric-demo-report.md'), description);
save_results(results, fullfile(output, 'geometric-demo-results.mat'));
fprintf(['Geometric (landmark) demo run complete. Synthetic geometry only; NOT ' ...
    'facial-expression accuracy.\n']);
fprintf('Selected model: %s\n', results.model.type);
fprintf('Synthetic held-out test accuracy: %.6f; macro F1: %.6f\n', ...
    results.testMetrics.accuracy, results.testMetrics.macroF1);
fprintf('Feature names (%d): %s ... %s\n', numel(results.dataset.featureNames), ...
    results.dataset.featureNames{1}, results.dataset.featureNames{end});
fprintf('Report: %s\n', fullfile(output, 'geometric-demo-report.md'));
end