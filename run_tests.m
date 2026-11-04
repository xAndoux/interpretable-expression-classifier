% Convenience entry point: run_tests from the project root.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'scripts'));
run_portable_tests();
