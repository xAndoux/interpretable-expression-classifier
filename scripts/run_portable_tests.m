function run_portable_tests()
%RUN_PORTABLE_TESTS Portable, dependency-light MATLAB/Octave regression suite.
% Run from any directory. Adds only this project's src/tests to the path.
here = fileparts(mfilename('fullpath'));
project = fileparts(here);
addpath(fullfile(project, 'src'));
addpath(fullfile(project, 'tests'));
tests = {@test_class_names, @test_resize, @test_preprocessing, ...
    @test_features, @test_geometric_features, @test_split, @test_group_split, ...
    @test_normalization, @test_metrics, @test_training, ...
    @test_duplicate_rejection, @test_synthetic};
failures = 0;
started = tic;
for k = 1:numel(tests)
    name = func2str(tests{k});
    try
        tests{k}();
        fprintf('PASS %s\n', name);
    catch exception
        failures = failures + 1;
        fprintf(2, 'FAIL %s: %s\n', name, exception.message);
        if isfield(exception, 'stack')
            for j = 1:numel(exception.stack)
                fprintf(2, '  %s:%d\n', exception.stack(j).name, exception.stack(j).line);
            end
        end
    end
end
fprintf('Executed %d tests: %d passed, %d failed in %.3f seconds.\n', ...
    numel(tests), numel(tests)-failures, failures, toc(started));
if failures > 0
    error('expression:TestFailure', '%d test(s) failed.', failures);
end
end
