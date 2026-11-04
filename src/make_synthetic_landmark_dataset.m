function dataset = make_synthetic_landmark_dataset(config)
%MAKE_SYNTHETIC_LANDMARK_DATASET Parameterized 68-point face geometry per class.
%
% This generates abstract 68-point landmark layouts in normalized face space
% whose GEOMETRY follows the facial actions associated with each visible
% expression (brows raised/lowered, eyes widened/narrowed, mouth open/closed,
% lip corners up/down). The reference interocular distance is fixed per face,
% so the ratio features in EXTRACT_GEOMETRIC_FEATURES are meaningful.
%
% HONESTY WARNING: these are NOT face images and carry NO real ground truth.
% This demo exists solely to prove that the geometric-feature layer produces
% separable, interpretable measurements and that the classifier pipeline runs.
% Any reported accuracy is an execution measurement on synthetic geometry,
% never facial-expression recognition accuracy.

if nargin < 1 || isempty(config)
    config = default_config();
end
names = class_names();
perClass = 12;
n = numel(names) * perClass;
landmarksCell = cell(1, n);
labels = cell(1, n);
paths = cell(1, n);
groups = repmat({''}, 1, n);

for c = 1:numel(names)
    params = emotion_parameters(names{c});
    for k = 1:perClass
        index = (c - 1) * perClass + k;
        jitter = 0.012 * (k - 1);
        lm = base_face();
        lm = apply_expression(lm, params, jitter);
        landmarksCell{index} = lm;
        labels{index} = names{c};
        paths{index} = sprintf('landmarks/%s_synth_%02d', names{c}, k);
    end
end

first = extract_geometric_features(landmarksCell{1});
features = zeros(n, numel(first));
featureNames = cell(1, numel(first));
for k = 1:n
    [features(k, :), featureNames] = extract_geometric_features(landmarksCell{k});
end

dataset.landmarks = landmarksCell;
dataset.paths = paths;
dataset.labels = labels;
dataset.groups = groups;
dataset.features = features;
dataset.featureNames = featureNames;
dataset.contentKeys = cell(1, n);
dataset.identityProvided = false;
dataset.root = '';
dataset.config = config;
dataset.syntheticOnly = true;
end

function params = emotion_parameters(label)
%EMOTION_PARAMETERS Deterministic geometry deltas per visible expression.
% browY>0 raises brows (downward image y). browInnerX>0 pulls inner brows
% toward midline. eyeOpen scales eye height (1=neutral). mouthOpen adds lip
% separation. mouthCornerY>0 raises lip corners (smile).
switch label
    case 'happy'
        params = struct('browY', 0.010, 'browInnerX', 0.000, 'eyeOpen', 0.90, ...
                        'mouthOpen', 0.30, 'mouthCornerY', 0.050, 'mouthWidth', 1.06);
    case 'sad'
        params = struct('browY', -0.010, 'browInnerX', 0.015, 'eyeOpen', 0.75, ...
                        'mouthOpen', 0.16, 'mouthCornerY', -0.050, 'mouthWidth', 0.90);
    case 'surprise'
        params = struct('browY', 0.060, 'browInnerX', -0.010, 'eyeOpen', 1.80, ...
                        'mouthOpen', 0.55, 'mouthCornerY', 0.005, 'mouthWidth', 0.85);
    case 'fear'
        params = struct('browY', 0.050, 'browInnerX', 0.030, 'eyeOpen', 1.50, ...
                        'mouthOpen', 0.30, 'mouthCornerY', -0.015, 'mouthWidth', 0.95);
    case 'angry'
        params = struct('browY', -0.050, 'browInnerX', 0.045, 'eyeOpen', 0.55, ...
                        'mouthOpen', 0.20, 'mouthCornerY', -0.030, 'mouthWidth', 0.88);
    case 'disgust'
        params = struct('browY', -0.030, 'browInnerX', 0.060, 'eyeOpen', 0.70, ...
                        'mouthOpen', 0.14, 'mouthCornerY', 0.020, 'mouthWidth', 0.80);
    otherwise % neutral
        params = struct('browY', 0.000, 'browInnerX', 0.000, 'eyeOpen', 1.00, ...
                        'mouthOpen', 0.18, 'mouthCornerY', 0.000, 'mouthWidth', 1.00);
end
end

function lm = base_face()
%BASE_FACE Neutral 68-point face template. Y grows downward.
% 1-based IBUG-68 mapping (Dlib/MediaPipe convention):
%   jaw 1:17 | left brow 18:22 | right brow 23:27 | nose 28:36
%   left eye 37:42 | right eye 43:48 | mouth outer 49:60 | inner 61:68
lm = zeros(68, 2);
% Jaw (1:17)
for i = 1:17
    t = (i - 1) / 16;
    lm(i, :) = [0.08 + 0.84 * t, 0.72 + 0.10 * sin(-2 * pi * t)];
end
% Left brow (18:22), outer->inner. Left brow sits LEFT of the face centre
% (x near 0.28..0.40), inner tip (22) nearest the midline.
for j = 0:4
    lm(18 + j, :) = [0.27 + 0.034 * j, 0.340];
end
% Right brow (23:27), inner->outer. Right brow sits RIGHT of centre
% (x near 0.60..0.73), inner tip (23) nearest the midline.
for j = 0:4
    lm(23 + j, :) = [0.60 + 0.034 * j, 0.340];
end
% Nose (28:36)
lm(28+0, :) = [0.5, 0.40]; lm(28+1, :) = [0.5, 0.47];
lm(28+2, :) = [0.5, 0.55]; lm(28+3, :) = [0.46, 0.565];
lm(28+4, :) = [0.5, 0.55]; lm(28+5, :) = [0.54, 0.565];
lm(28+6, :) = [0.5, 0.50]; lm(28+7, :) = [0.5, 0.575]; lm(28+8, :) = [0.5, 0.60];
% Left eye (37:42): outer, top-outer, top-inner, inner, bottom-inner, bottom-outer
leftEyeX = 0.30; earGapX = 0.085; eyeH = 0.018;
lm(37, :) = [leftEyeX - earGapX, 0.445];
lm(38, :) = [leftEyeX - earGapX*0.4, 0.445 - eyeH];
lm(39, :) = [leftEyeX + earGapX*0.4, 0.445 - eyeH];
lm(40, :) = [leftEyeX + earGapX, 0.445];
lm(41, :) = [leftEyeX + earGapX*0.4, 0.445 + eyeH];
lm(42, :) = [leftEyeX - earGapX*0.4, 0.445 + eyeH];
% Right eye (43:48)
rightEyeX = 0.60;
lm(43, :) = [rightEyeX - earGapX, 0.445];
lm(44, :) = [rightEyeX - earGapX*0.4, 0.445 - eyeH];
lm(45, :) = [rightEyeX + earGapX*0.4, 0.445 - eyeH];
lm(46, :) = [rightEyeX + earGapX, 0.445];
lm(47, :) = [rightEyeX + earGapX*0.4, 0.445 + eyeH];
lm(48, :) = [rightEyeX - earGapX*0.4, 0.445 + eyeH];
% Mouth outer (49:60): left corner, upper lip to right corner, lower lip back
mouthCX = 0.5; mouthCY = 0.645; mouthHalfW = 0.13; lipHalfH = 0.022;
for j = 0:11
    t = j / 11;
    lm(49 + j, :) = [mouthCX - mouthHalfW + 2*mouthHalfW*t, ...
                     mouthCY + lipHalfH * sin(2 * pi * t)];
end
% Mouth inner (61:68): inner aperture ring (narrower)
innerHalfW = 0.09;
for j = 0:7
    t = j / 7;
    lm(61 + j, :) = [mouthCX - innerHalfW + 2*innerHalfW*t, mouthCY + 0.012 * sin(2 * pi * t)];
end
end

function lm = apply_expression(lm, p, jitter)
%APPLY_EXPRESSION Apply geometry deltas to landmark groups (see 1-based map).
% 1. Brows: raise/lower (y) and pull inner tips toward midline.
% 2. Eyes: scale vertical opening about each eye centre.
% 3. Mouth: lip separation, corner height, width.
% Brow vertical shift: image y grows DOWN, so a positive browY (raise)
% must DECREASE the y coordinate; a negative browY lowers the brow.
lm(18:22, 2) = lm(18:22, 2) - p.browY + jitter;
lm(23:27, 2) = lm(23:27, 2) - p.browY + jitter;
% Inner tips pulled toward the midline: left inner (22) moves right (+x),
% right inner (23) moves left (-x) when browInnerX is positive.
lm(22, 1) = lm(22, 1) + p.browInnerX;
lm(23, 1) = lm(23, 1) - p.browInnerX;
% Eyes: scale vertical extent about centre.
centerL = mean(lm(37:42, :), 1);
for j = 37:42
    lm(j, :) = centerL + (lm(j, :) - centerL) .* [1, p.eyeOpen];
end
centerR = mean(lm(43:48, :), 1);
for j = 43:48
    lm(j, :) = centerR + (lm(j, :) - centerR) .* [1, p.eyeOpen];
end
% Mouth: lip separation about vertical centre.
cyM = (lm(52, 2) + lm(59, 2)) / 2;
openAmt = p.mouthOpen * 0.12;
lm([52, 53], 2) = lm([52, 53], 2) - openAmt;
lm([58, 59], 2) = lm([58, 59], 2) + openAmt;
% Mouth corners: raise for smile (downward y), lower for frown.
lm([49, 55], 2) = lm([49, 55], 2) - p.mouthCornerY;
% Mouth width: scale horizontal span about mouth centre.
mouthCX = (lm(49, 1) + lm(55, 1)) / 2;
lm(49:60, 1) = mouthCX + (lm(49:60, 1) - mouthCX) * p.mouthWidth;
% Clamp to sane space (safety for plotting / downstream checks).
lm = max(min(lm, 1.3), -0.3);
end