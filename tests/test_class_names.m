function test_class_names()
names = class_names();
assert(isequal(names, {'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'}));
assert(numel(unique(names)) == 7);
end
