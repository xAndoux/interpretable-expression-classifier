function order = deterministic_order(labels, groups, seed)
%DETERMINISTIC_ORDER Stable pseudo-random ordering, independent of filenames.
% Constructed without Statistics Toolbox or reliance on implicit RNG state.
n = numel(labels);
if numel(groups) ~= n
    error('expression:InvalidMetadata', 'Labels and groups must have equal length.');
end
keys = zeros(1, n);
prime = 2147483647;
for k = 1:n
    group = groups(k);
    if iscell(group)
        group = group{1};
    end
    if isnumeric(group)
        groupText = num2str(group);
    else
        groupText = char(group);
    end
    text = [labels{k} '|' groupText];
    value = mod(double(seed), prime);
    for j = 1:numel(text)
        value = mod(value * 257 + double(text(j)) + 1, prime);
    end
    keys(k) = value;
end
[~, order] = sortrows([keys(:), (1:n)'], [1, 2]);
order = order(:)';
end
