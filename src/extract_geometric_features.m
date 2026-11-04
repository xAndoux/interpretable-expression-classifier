function [features, names] = extract_geometric_features(landmarks)
%EXTRACT_GEOMETRIC_FEATURES Semantic facial-shape descriptors from 68 landmarks.
%
% LANMARKS is a 68x2 matrix of [x, y] coordinates in the standard IBUG-68
% face-landmark convention (the layout produced by Dlib and MediaPipe Face
% Mesh's 68-point variant). Y grows downward (image convention).
%
% This is the interpretable "geometric" layer: instead of opaque pixel
% statistics it measures named facial actions used in facial-expression
% research (brow raise/lower, eye widen/narrow, lip corner pull, mouth open).
% A downstream classifier then maps those measurements to expression labels.
%
% Feature names returned in NAMES always align with FEATURES (row vector).
%
% 1-based IBUG-68 mapping:
%   jaw 1:17 | left brow 18:22 | right brow 23:27 | nose 28:36
%   left eye 37:42 | right eye 43:48 | mouth outer 49:60 | inner 61:68

assert(size(landmarks, 1) == 68 && size(landmarks, 2) == 2, ...
    'expression:BadLandmarks', 'Expected a 68x2 landmark matrix.');

% Reference scale: interocular distance between inner eye corners
% (index 40 = left inner, index 43 = right inner). Every ratio feature is
% divided by this so results are comparable across image sizes.
interocular = norm(landmarks(43, :) - landmarks(40, :));
if interocular < eps
    error('expression:BadLandmarks', 'Interocular distance is zero; check landmark layout.');
end

eyeMidLeft = mean(landmarks(37:42, :), 1);   % left eye centroid
eyeMidRight = mean(landmarks(43:48, :), 1);  % right eye centroid
browMidLeft = mean(landmarks(18:22, :), 1);  % left brow centroid
browMidRight = mean(landmarks(23:27, :), 1); % right brow centroid

% --- 1. Brow height above the eyes (positive = raised, negative = lowered).
% y grows downward, so (eye_y - brow_y) > 0 means the brow sits higher.
browHeightL = (eyeMidLeft(2) - browMidLeft(2)) / interocular;
browHeightR = (eyeMidRight(2) - browMidRight(2)) / interocular;

% --- 2. Brow-to-eye vertical gap (small = brows pulled down onto the eyes).
% Same signed distance; kept as a separate named feature for interpretability.
browGapL = browHeightL;
browGapR = browHeightR;

% --- 3. Inter-brow separation (small = brows knit toward the midline).
interBrow = norm(landmarks(23, :) - landmarks(22, :)) / interocular;

% --- 4. Eye openness (classic EAR) + brow slope.
leftEAR = eye_aspect_ratio(landmarks(37:42, :), interocular);
rightEAR = eye_aspect_ratio(landmarks(43:48, :), interocular);
% Brow slope: positive when the inner end is higher than the outer end.
leftSlope = (landmarks(18, 2) - landmarks(22, 2)) / interocular;
rightSlope = (landmarks(27, 2) - landmarks(23, 2)) / interocular;

% --- 5. Mouth shape.
mouthLeft = landmarks(49, :); mouthRight = landmarks(55, :);
mouthWidth = norm(mouthRight - mouthLeft) / interocular;
mouthOpen = (mean(landmarks([58, 59], 2)) - mean(landmarks([52, 53], 2))) / interocular;
% Lip curvature: where the mouth's vertical centre sits relative to the
% line through the two corners. Positive = corners raised (smile), negative
% = corners lowered (frown).
cornerHeight = (mouthLeft(2) + mouthRight(2)) / 2;
lipCurve = (mean(landmarks([52, 53, 58, 59], 2)) - cornerHeight) / interocular;

% --- 6. Combined interpretable facial-action proxies.
smileProxy = lipCurve + 0.5 * max(0, mouthOpen);
surpriseProxy = (browHeightL + browHeightR) / 2 + 0.4 * (leftEAR + rightEAR) + 0.4 * mouthOpen;
angerProxy = -((browHeightL + browHeightR) / 2) + 0.3 * (2 - (leftEAR + rightEAR)) + 0.3 * (1 - interBrow);

features = [browHeightL, browHeightR, browGapL, browGapR, interBrow, ...
            leftEAR, rightEAR, leftSlope, rightSlope, mouthWidth, mouthOpen, ...
            lipCurve, smileProxy, surpriseProxy, angerProxy];
features = double(features(:)');
names = {'brow_height_left_rel', 'brow_height_right_rel', ...
         'brow_to_eye_left_rel', 'brow_to_eye_right_rel', 'inter_brow_distance_rel', ...
         'left_eye_openness', 'right_eye_openness', 'left_brow_slope', 'right_brow_slope', ...
         'mouth_width_rel', 'mouth_openness_rel', 'lip_curvature_rel', ...
         'smile_proxy', 'surprise_proxy', 'anger_proxy'};
end

function value = eye_aspect_ratio(eyePoints, interocular)
%EYE_ASPECT_RATIO EAR = (P2-P6 + P3-P5) / (2 * |P1-P4|).
% eyePoints is the 6-point eye ring in order (indices 37..42 or 43..48).
p1 = eyePoints(1, :); p2 = eyePoints(2, :); p3 = eyePoints(3, :);
p4 = eyePoints(4, :); p5 = eyePoints(5, :); p6 = eyePoints(6, :);
w = norm(p1 - p4);
if w < eps
    value = NaN;
    return;
end
value = (norm(p2 - p6) + norm(p3 - p5)) / (2 * w + eps);
end