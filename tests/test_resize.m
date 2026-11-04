function test_resize()
image = reshape(1:16, 4, 4);
same = resize_bilinear(image, [4, 4]);
assert_close(same, image);
large = resize_bilinear(image, [8, 8]);
assert_close(large(1, 1), 1);
assert_close(large(end, end), 16);
assert(all(large(:) >= 1 & large(:) <= 16));
% Linear ramps remain linear away from clamped boundary interpolation.
ramp = repmat(0:3, 4, 1);
resized = resize_bilinear(ramp, [4, 8]);
assert_close(resized(2, 3:6), [0.75, 1.25, 1.75, 2.25]);
end
