function test_split()
config = default_config();
names = class_names();
labels = {};
for c = 1:7
    labels = [labels, repmat(names(c), 1, 15)]; %#ok<AGROW>
end
groups = repmat({''}, size(labels));
a = stratified_split(labels, groups, config);
b = stratified_split(labels, groups, config);
assert(isequal(a, b));
for c = 1:7
    inClass = strcmp(labels, names{c});
    assert(sum(inClass & strcmp(a, 'train')) == 9);
    assert(sum(inClass & strcmp(a, 'validation')) == 3);
    assert(sum(inClass & strcmp(a, 'test')) == 3);
end
assert(isempty(intersect(find(strcmp(a, 'train')), find(strcmp(a, 'test')))));
end
