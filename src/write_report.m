function write_report(results, filename, datasetDescription)
%WRITE_REPORT Write measurements computed by this invocation, with explicit scope.
fid = fopen(filename, 'w');
if fid < 0
    error('expression:ReportFailure', 'Cannot create report: %s', filename);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Expression classifier evaluation report\n\n');
fprintf(fid, '**Scope:** this repository evaluates only the code it contains.\n\n');
fprintf(fid, '**Dataset:** %s\n\n', datasetDescription);
fprintf(fid, ['Facial-expression labels describe visible appearance categories, not inner emotional truth. ' ...
    'No historical accuracy is reproduced or claimed. No affiliation or endorsement is implied.\n\n']);
fprintf(fid, '- Model: %s\n', results.model.description);
if strcmp(results.model.type, 'ridge')
    fprintf(fid, '- Validation-selected lambda: %g\n', results.model.lambda);
end
fprintf(fid, '- Seed: %d\n', results.config.seed);
fprintf(fid, '- Image size: %d x %d\n', results.config.imageSize);
fprintf(fid, '- Block grid: %d x %d\n', results.config.blockGrid);
fprintf(fid, '- Training z-score features: %d rows; %d constant features\n', ...
    results.model.normalizer.trainingRowCount, results.model.normalizer.constantFeatureCount);
if results.dataset.identityProvided
    fprintf(fid, '- Split: deterministic, identity-group-disjoint within seven-class subsets.\n');
else
    fprintf(fid, ['- Split: deterministic image-stratified; identity metadata absent. ' ...
        'Do not interpret results as person-independent face performance.\n']);
end
fprintf(fid, '- Exact duplicate handling: normalized decoded-content comparison; reject before split.\n');
fprintf(fid, '- Selection rule: %s\n\n', results.selection.rule);
write_partition(fid, 'Train (diagnostic only)', results.trainMetrics);
write_partition(fid, 'Validation (model selection)', results.validationMetrics);
write_partition(fid, 'Held-out test (one-time evaluation)', results.testMetrics);
fprintf(fid, '## Candidate selection measurements\n\n');
fprintf(fid, '| Candidate | Lambda | Validation macro F1 | Validation accuracy |\n');
fprintf(fid, '|---|---:|---:|---:|\n');
for k = 1:numel(results.selection.candidates)
    item = results.selection.candidates(k);
    fprintf(fid, '| %s | %g | %.6f | %.6f |\n', item.type, item.lambda, ...
        item.validationMacroF1, item.validationAccuracy);
end
fprintf(fid, '\n## Limitations\n\n');
fprintf(fid, ['- A small or nonrepresentative sample cannot establish a general benchmark.\n' ...
    '- This basic split does not by itself prove subject independence.\n' ...
    '- Pre-aligned crops and valid labels are prerequisites; no detector or aligner is included.\n' ...
    '- Exact duplicate rejection does not detect every near-duplicate, augmentation, or identity.\n' ...
    '- Reported values apply only to this exact run and must not be generalized to other data.\n']);
end

function write_partition(fid, title, metrics)
fprintf(fid, '## %s\n\n', title);
fprintf(fid, 'Samples: %d; accuracy: %.6f; macro F1: %.6f\n\n', ...
    metrics.sampleCount, metrics.accuracy, metrics.macroF1);
fprintf(fid, '| Class | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n');
for c = 1:numel(metrics.classNames)
    fprintf(fid, '| %s | %.6f | %.6f | %.6f | %d |\n', metrics.classNames{c}, ...
        metrics.precision(c), metrics.recall(c), metrics.f1(c), metrics.support(c));
end
fprintf(fid, '\nConfusion matrix (rows = true, columns = predicted):\n\n');
fprintf(fid, '| True \\ Pred | %s |\n|---|%s|\n', strjoin(metrics.classNames, ' | '), ...
    repmat('---:|', 1, numel(metrics.classNames)));
for c = 1:numel(metrics.classNames)
    fprintf(fid, '| %s |', metrics.classNames{c});
    fprintf(fid, ' %d |', metrics.confusion(c, :));
    fprintf(fid, '\n');
end
fprintf(fid, '\n');
end
