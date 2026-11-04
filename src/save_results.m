function save_results(results, filename)
%SAVE_RESULTS Save measured experiment and selected model for reproducibility.
% No bundled trained model is supplied; this is generated only by a real run.
save(filename, 'results', '-v7');
end
