function key = content_fingerprint(image)
%CONTENT_FINGERPRINT Fast content key; all equal keys are verified by ISEQUAL.
% This is not a security hash. Two modular rolling hashes avoid integer
% overflow and work identically in MATLAB and Octave. Collision verification
% in LOAD_DATASET is what makes exact-duplicate rejection authoritative.
validateattributes(image, {'numeric'}, {'2d', 'finite', 'nonempty'});
if isfloat(image)
    if any(image(:) < -1) || any(image(:) > 1)
        error('expression:InvalidRange', 'Floating-point image values must be in [-1, 1].');
    end
    % Preprocessing produces [-1,1]. Its float encoding is deliberately
    % distinct from the integer image encoding used by generic callers.
    payload = [uint8(1); typecast(uint16(size(image)), 'uint8')(:); ...
        typecast(double(image), 'uint8')(:)];
else
    payload = [uint8(0); typecast(uint16(size(image)), 'uint8')(:); ...
        typecast(image, 'uint8')(:)];
end
h1 = 0;
h2 = 0;
p1 = 2147483647;
p2 = 2147483629;
for k = 1:numel(payload)
    value = double(payload(k));
    h1 = mod(h1 * 257 + value + 1, p1);
    h2 = mod(h2 * 263 + value + mod(k, 251), p2);
end
key = sprintf('%010d-%010d-%010d', h1, h2, numel(payload));
end
