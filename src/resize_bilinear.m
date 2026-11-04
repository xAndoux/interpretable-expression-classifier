function output = resize_bilinear(input, outputSize)
%RESIZE_BILINEAR Toolbox-free bilinear resize with pixel-center coordinates.
validateattributes(outputSize, {'numeric'}, {'vector', 'numel', 2, 'integer', 'positive'});
if ndims(input) ~= 2
    error('expression:InvalidImage', 'Bilinear resizing expects a grayscale matrix.');
end
input = double(input);
[inputRows, inputColumns] = size(input);
outputRows = outputSize(1);
outputColumns = outputSize(2);
if inputRows == outputRows && inputColumns == outputColumns
    output = input;
    return;
end

% Equivalent geometric convention to common half-pixel image resizers.
rowCoordinates = ((1:outputRows) - 0.5) * inputRows / outputRows + 0.5;
columnCoordinates = ((1:outputColumns) - 0.5) * inputColumns / outputColumns + 0.5;
rowCoordinates = min(max(rowCoordinates, 1), inputRows);
columnCoordinates = min(max(columnCoordinates, 1), inputColumns);
r0 = floor(rowCoordinates);
c0 = floor(columnCoordinates);
r1 = min(r0 + 1, inputRows);
c1 = min(c0 + 1, inputColumns);
rw = rowCoordinates - r0;
cw = columnCoordinates - c0;

output = zeros(outputRows, outputColumns);
for r = 1:outputRows
    top = input(r0(r), c0) * (1 - rw(r)) + input(r1(r), c0) * rw(r);
    bottom = input(r0(r), c1) * (1 - rw(r)) + input(r1(r), c1) * rw(r);
    output(r, :) = top .* (1 - cw) + bottom .* cw;
end
end
