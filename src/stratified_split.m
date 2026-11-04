function split = stratified_split(labels, groups, config)
%STRATIFIED_SPLIT Deterministic separated train/validation/test assignment.
% Empty groups: stratify images independently. Nonempty groups: enforce
% group-disjoint splits and approximate per-class proportions. Each class
% needs enough distinct groups to cover all three partitions. No test data
% participates in fitting or selection after this assignment.
if nargin < 2 || isempty(groups)
    groups = repmat({''}, size(labels));
end
if ~iscell(labels) || ~iscell(groups) || numel(labels) ~= numel(groups)
    error('expression:InvalidMetadata', 'Labels and groups must be equally sized cell arrays.');
end
if any(cellfun(@isempty, labels))
    error('expression:InvalidMetadata', 'Every sample needs a class label.');
end
n = numel(labels);
split = repmat({'train'}, 1, n);
if isempty([groups{:}])
    names = unique(labels);
    for c = 1:numel(names)
        indices = find(strcmp(labels, names{c}));
        if numel(indices) < config.minimumImagesPerClass
            error('expression:TooFewImages', ...
                'Class %s has %d images; at least %d are needed for separated splits.', ...
                names{c}, numel(indices), config.minimumImagesPerClass);
        end
        ordered = indices(deterministic_order(labels(indices), indices, config.seed + c));
        nTest = max(1, round(numel(indices) * config.testFraction));
        nValidation = max(1, round(numel(indices) * config.validationFraction));
        if nTest + nValidation >= numel(indices)
            error('expression:TooFewImages', 'A class must retain training samples.');
        end
        split(ordered(1:nTest)) = {'test'};
        split(ordered(nTest+1:nTest+nValidation)) = {'validation'};
    end
else
    split = grouped_split(labels, groups, config);
end
end

function split = grouped_split(labels, groups, config)
if any(cellfun(@isempty, groups))
    error('expression:InvalidMetadata', 'Supply either all identity groups or none, not a mixture.');
end
names = class_names();
if any(~ismember(labels, names))
    error('expression:InvalidMetadata', 'Grouped evaluation requires the seven canonical labels.');
end
uniqueGroups = unique(groups);
nGroups = numel(uniqueGroups);
groupMembership = false(nGroups, numel(names));
groupSizes = zeros(1, nGroups);
for g = 1:nGroups
    members = find(strcmp(groups, uniqueGroups{g}));
    groupSizes(g) = numel(members);
    groupMembership(g, :) = ismember(names, labels(members));
end
assignment = zeros(1, nGroups); % 0 unassigned, 1 train, 2 validation, 3 test
classCounts = zeros(3, numel(names));
classGroupCounts = sum(groupMembership, 1);
[~, classOrder] = sortrows([classGroupCounts(:), (1:numel(names))'], [1, 2]);
% Reserve distinct groups for every class/partition. This may legitimately
% fail on tiny datasets; silently breaking disjointness would be worse.
for c = classOrder(:)'
    available = find(assignment == 0 & groupMembership(:, c)');
    if numel(available) < 3
        error('expression:InsufficientIdentityCoverage', ...
            ['Class %s cannot cover all three partitions with disjoint identity groups. ' ...
             'Add independent subjects or change the documented experiment design.'], names{c});
    end
    labelsForGroups = repmat(names(c), 1, numel(available));
    keys = deterministic_order(labelsForGroups, uniqueGroups(available), config.seed + c);
    chosen = available(keys(1:3));
    for destination = 1:3
        assignment(chosen(destination)) = destination;
        classCounts(destination, :) = classCounts(destination, :) + ...
            groupSizes(chosen(destination)) * groupMembership(chosen(destination), :);
    end
end
remaining = find(assignment == 0);
keys = deterministic_order(repmat({'group'}, size(remaining)), uniqueGroups(remaining), config.seed);
for j = remaining(keys)
    relevant = find(groupMembership(j, :));
    costs = zeros(1, 3);
    for destination = 1:3
        target = max(1, round(sum(groupSizes) * ...
            [1-config.validationFraction-config.testFraction, ...
             config.validationFraction, config.testFraction]));
        relative = classCounts(destination, relevant) ./ target(destination);
        costs(destination) = mean(relative);
    end
    [~, destination] = min(costs);
    assignment(j) = destination;
    classCounts(destination, :) = classCounts(destination, :) + ...
        groupSizes(j) * groupMembership(j, :);
end
split = repmat({'train'}, 1, numel(labels));
for g = 1:nGroups
    if assignment(g) == 2
        split(strcmp(groups, uniqueGroups{g})) = {'validation'};
    elseif assignment(g) == 3
        split(strcmp(groups, uniqueGroups{g})) = {'test'};
    end
end
end
