function test_preprocessing()
config = default_config();
image = zeros(32, 32);
image(5:27, 5:27) = 0.8;
image(10:15, 10:15) = 0.2;
processed = preprocess_image(image, config);
assert(isequal(size(processed), config.imageSize));
assert(all(processed(:) >= -1 & processed(:) <= 1));
assert_close(median(processed(:)), 0, 1e-12);
gray = preprocess_image(repmat(image, [1, 1, 3]), config);
assert_close(gray, processed, 1e-12);
integer = preprocess_image(uint8(round(image * 255)), config);
assert_close(integer, processed, 1e-12);
threw = false;
try
    preprocess_image(zeros(8, 8), config);
catch exception
    threw = strcmp(exception.identifier, 'expression:ConstantImage');
end
assert(threw, 'Constant images must be rejected.');
assert(~strcmp(content_fingerprint([0, 0.25; 0.5, 0.75]), ...
    content_fingerprint([0, 0.5; 0.5, 0.75])));
end
