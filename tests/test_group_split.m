function test_group_split()
config = default_config();
names = class_names();
labels = {};
groups = {};
for c = 1:7
    for g = 1:6
        labels{end+1} = names{c}; %#ok<AGROW>
        groups{end+1} = sprintf('person-%02d-%02d', c, g); %#ok<AGROW>
    end
end
split = stratified_split(labels, groups, config);
assert(isequal(split, stratified_split(labels, groups, config)));
for g = 1:numel(groups)
    same = strcmp(groups, groups{g});
    assert(all(strcmp(split(same), split{g})), 'Identity group crossed partitions.');
end
for c = 1:7
    inClass = strcmp(labels, names{c});
    assert(any(inClass & strcmp(split, 'train')));
    assert(any(inClass & strcmp(split, 'validation')));
    assert(any(inClass & strcmp(split, 'test')));
end
% Failure is safer than silently degrading identity separation on tiny data.
threw = false;
try
    stratified_split(labels(1:2), groups(1:2), config);
catch exception
    threw = strcmp(exception.identifier, 'expression:InsufficientIdentityCoverage');
end
assert(threw);
end
