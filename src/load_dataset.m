function dataset = load_dataset(datasetRoot, config, identityFile)
%LOAD_DATASET Decode and validate seven-class aligned face-crop directories.
% Expected: ROOT/angry, disgust, fear, happy, sad, surprise, neutral.
% Optional identity CSV: relative_path,identity (commas in paths unsupported).
% All exact duplicates of normalized decoded content are rejected before
% splitting. Raw file hashes alone would miss recompressed copies; normalized
% comparison catches decoded duplicates but not all near-duplicates.
if nargin < 2 || isempty(config)
    config = default_config();
end
if nargin < 3
    identityFile = '';
end
if exist(datasetRoot, 'dir') ~= 7
    error('expression:MissingDataset', 'Dataset directory not found: %s', datasetRoot);
end
names = class_names();
for c = 1:numel(names)
    if exist(fullfile(datasetRoot, names{c}), 'dir') ~= 7
        error('expression:MissingClass', 'Missing required class directory: %s', names{c});
    end
end
entries = dir(datasetRoot);
for k = 1:numel(entries)
    if entries(k).isdir && ~ismember(entries(k).name, [{'.', '..'}, names])
        error('expression:UnexpectedDirectory', 'Unexpected dataset subdirectory: %s', entries(k).name);
    end
end
identityMap = read_identity_map(datasetRoot, identityFile);
paths = {};
labels = {};
for c = 1:numel(names)
    entries = dir(fullfile(datasetRoot, names{c}));
    for k = 1:numel(entries)
        if entries(k).isdir
            continue;
        end
        [~, ~, extension] = fileparts(entries(k).name);
        if ~ismember(lower(extension), {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm'})
            error('expression:UnexpectedFile', 'Unsupported file in class folder: %s', entries(k).name);
        end
        paths{end+1} = [names{c} '/' entries(k).name]; %#ok<AGROW>
        labels{end+1} = names{c}; %#ok<AGROW>
    end
end
if isempty(paths)
    error('expression:EmptyDataset', 'No images found.');
end
if numel(unique(paths)) ~= numel(paths)
    error('expression:DuplicatePaths', 'Image paths must be unique.');
end
n = numel(paths);
processed = zeros([config.imageSize, n]);
keys = cell(1, n);
seen = containers.Map('KeyType', 'char', 'ValueType', 'any');
for k = 1:n
    absolutePath = fullfile(datasetRoot, paths{k});
    try
        info = imfinfo(absolutePath);
        if isempty(info) || info(1).Width < 2 || info(1).Height < 2
            error('expression:InvalidImage', 'Image dimensions are invalid.');
        end
        processed(:, :, k) = preprocess_image(absolutePath, config);
    catch exception
        error('expression:InvalidImage', 'Cannot decode/validate %s: %s', paths{k}, exception.message);
    end
    key = content_fingerprint(processed(:, :, k));
    keys{k} = key;
    if isKey(seen, key)
        candidates = seen(key);
        for previous = candidates
            if isequal(processed(:, :, previous), processed(:, :, k))
                error('expression:DuplicateImage', ...
                    'Exact normalized decoded duplicate rejected: %s and %s', paths{previous}, paths{k});
            end
        end
        seen(key) = [candidates, k];
    else
        seen(key) = k;
    end
end
% Feature allocation needs one descriptor lookup, kept separate for clarity.
[~, featureNames] = extract_features(processed(:, :, 1), config);
features = zeros(n, numel(featureNames));
for k = 1:n
    features(k, :) = extract_features(processed(:, :, k), config);
end
groups = repmat({''}, 1, n);
identityProvided = ~isempty(identityFile);
if identityProvided
    for k = 1:n
        if ~isKey(identityMap, paths{k})
            error('expression:MissingIdentity', 'No identity mapping for %s.', paths{k});
        end
        groups{k} = identityMap(paths{k});
    end
end
dataset.paths = paths;
dataset.labels = labels;
dataset.groups = groups;
dataset.features = features;
dataset.featureNames = featureNames;
dataset.contentKeys = keys;
dataset.identityProvided = identityProvided;
dataset.root = datasetRoot;
dataset.config = config;
end

function mapping = read_identity_map(root, filename)
mapping = containers.Map('KeyType', 'char', 'ValueType', 'char');
if isempty(filename)
    return;
end
if exist(filename, 'file') ~= 2
    filename = fullfile(root, filename);
end
fid = fopen(filename, 'r');
if fid < 0
    error('expression:MissingIdentity', 'Cannot open identity map: %s', filename);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
header = fgetl(fid);
if ~ischar(header) || ~strcmp(strtrim(header), 'relative_path,identity')
    error('expression:InvalidIdentity', 'Identity CSV header must be relative_path,identity.');
end
line = fgetl(fid);
while ischar(line)
    if ~isempty(strtrim(line))
        parts = regexp(line, ',', 'split');
        if numel(parts) ~= 2 || isempty(strtrim(parts{1})) || isempty(strtrim(parts{2}))
            error('expression:InvalidIdentity', 'Each identity row must contain exactly two nonempty fields.');
        end
        path = strrep(strtrim(parts{1}), '\', '/');
        if isKey(mapping, path)
            error('expression:InvalidIdentity', 'Duplicate identity-map path: %s', path);
        end
        mapping(path) = strtrim(parts{2});
    end
    line = fgetl(fid);
end
end
