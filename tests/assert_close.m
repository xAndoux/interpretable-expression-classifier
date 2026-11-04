function assert_close(actual, expected, tolerance)
if nargin < 3
    tolerance = 1e-10;
end
if ~isequal(size(actual), size(expected)) || any(abs(actual(:) - expected(:)) > tolerance)
    error('expression:Assertion', 'Numeric values differ beyond tolerance %g.', tolerance);
end
end
