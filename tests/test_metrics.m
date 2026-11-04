function test_metrics()
trueLabels = {'happy', 'happy', 'sad', 'sad', 'neutral'};
predicted = {'happy', 'sad', 'sad', 'happy', 'neutral'};
m = classification_metrics(trueLabels, predicted);
expected = zeros(7);
expected(4, 4) = 1; % happy -> happy
expected(4, 5) = 1; % happy -> sad
expected(5, 4) = 1; % sad -> happy
expected(5, 5) = 1; % sad -> sad
expected(7, 7) = 1;
assert_close(m.confusion, expected);
assert_close(m.accuracy, 3/5);
assert_close(m.precision(4:5), [0.5, 0.5]);
assert_close(m.recall(4:5), [0.5, 0.5]);
assert_close(m.f1(4:5), [0.5, 0.5]);
assert_close(m.support, [0, 0, 0, 2, 2, 0, 1]);
assert_close(m.macroF1, (0.5 + 0.5 + 1) / 7);
% Empty true/predicted classes remain explicit zero, not NaN.
assert(all(isfinite([m.precision, m.recall, m.f1])));
end
