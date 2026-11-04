% Portable Octave/MATLAB smoke runner; run from the project root.
% Synthetic patterns test execution only, not face recognition accuracy.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'scripts'));
run_synthetic_demo();
