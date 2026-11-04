function results = run_synthetic_demo()
%RUN_SYNTHETIC_DEMO Prove execution using abstract patterns, NOT face accuracy.
here = fileparts(mfilename('fullpath'));
project = fileparts(here);
addpath(fullfile(project, 'src'));
output = fullfile(project, 'results');
if exist(output, 'dir') ~= 7
    mkdir(output);
end
config = default_config();
dataset = make_synthetic_dataset(config);
results = run_experiment(dataset, config);
description = ['Generated abstract geometric patterns ONLY. This synthetic smoke run ' ...
    'proves code execution, not facial-expression recognition accuracy. No face images are used.'];
write_report(results, fullfile(output, 'synthetic-report.md'), description);
save_results(results, fullfile(output, 'synthetic-results.mat'));
fprintf(['Synthetic PATTERN smoke run complete. These measurements are not face ' ...
    'accuracy or a reproduction of any historical benchmark.\n']);
fprintf('Selected model: %s\n', results.model.type);
fprintf('Synthetic held-out test accuracy: %.6f; macro F1: %.6f\n', ...
    results.testMetrics.accuracy, results.testMetrics.macroF1);
fprintf('Report: %s\n', fullfile(output, 'synthetic-report.md'));
end
