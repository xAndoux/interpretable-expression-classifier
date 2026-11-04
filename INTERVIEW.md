# Interview notes: interpretable expression-classifier reconstruction

## How should this project be introduced?

“After losing the earlier academic project code, I built a clean-room MATLAB reconstruction from a high-level description. It is a transparent seven-class visible-expression baseline with spatial features, strict data separation, and reproducible evaluation. It is not recovered original code, and I do not claim that it reproduces the historically mentioned ~81% accuracy.”

Use “facial-expression classifier,” not “emotion detector.” Dataset labels describe visible appearance categories; they do not establish a person's inner emotional state. No institutional affiliation or endorsement is implied.

## What was the engineering goal?

The goal was to make each stage inspectable without a deep-learning framework: fixed preprocessing, location-preserving statistical descriptors, and models that can be expressed with matrix operations. The implementation favors a defensible baseline and reproducible measurements over an unsupported performance claim.

## Walk through an image.

The caller supplies an aligned face crop. The loader decodes it, converts color to grayscale, resizes to 64×64 using a built-in bilinear interpolator, subtracts the median, and divides by the IQR. Near-constant IQR falls back to population standard deviation; a constant image is rejected. Values are clipped to reduce extreme-contrast influence.

A 6×6 grid supplies block means, population standard deviations, mean horizontal and vertical gradients, gradient magnitudes, and horizontal reflection errors. Row and column profiles retain coarse shape. The result is 264 named features. These are geometric/intensity proxies, not semantic facial landmarks or facial-action-unit detections.

## Why bottom-up spatial features?

Global pixel variance loses location. A dark region near the mouth and one near the eye can have the same global statistics but different spatial meaning. Blocks preserve coarse location and local contrast; gradients describe directional changes; reflection error provides a limited symmetry cue. These descriptors remain sensitive to alignment and are not causal explanations of an expression.

## How do the classifiers work?

Nearest centroid standardizes features using training means/scales, computes each class centroid, and selects the smallest squared Euclidean distance. Ridge uses one-vs-rest targets and solves

```text
W = (Z'Z + lambda I) \ (Z'Y)
```

with an unpenalized intercept. Lambda shrinks coefficients to reduce variance and improve conditioning. Candidates are centroid and ridge with lambdas `[0.01, 0.1, 1, 10, 100]`.

Interpretability should not be oversold: standardized ridge coefficients and centroid differences are inspectable, but correlated features do not yield unique causal importance.

## How is leakage controlled?

A fixed seed and stable hashing make the assignment reproducible without relying on global RNG state. Without identity metadata, each class is split 60/20/20 into train/validation/test. This is image-stratified, **not** person-independent. With identity metadata, a deterministic grouped splitter reserves separate identities for each class and partition, then balances the remaining groups. It fails if sufficient identities are unavailable rather than silently weakening separation.

All exact normalized decoded-content duplicates are rejected before splitting. Fingerprints narrow candidates, but exact matrix comparison makes the decision. Near-duplicates, augmentations, adjacent video frames, and same-person images are not all caught by this check.

Only training rows fit z-score statistics. Validation macro F1 alone selects the model. Test labels do not enter fitting or selection. After inspecting test results, changing the pipeline to improve them would make the test set part of tuning; a new untouched test set would be needed.

## Why report macro F1?

Accuracy can hide poor minority-class performance. The report includes the full confusion matrix, support, and per-class precision, recall, and F1, plus macro F1 as an unweighted mean across all seven canonical classes. Undefined per-class values are reported as zero, which matters when tiny partitions lack classes.

## What does the synthetic demo prove?

It generates abstract geometric patterns and verifies that preprocessing, splitting, training, selection, reporting, and serialization execute. It contains no face images and no emotional ground truth. Synthetic measurements are not face accuracy, a benchmark, or evidence that the original project was reproduced.

## What would you improve next with real data?

1. Establish documented consent/licensing and an alignment protocol.
2. Use a sufficiently large identity-disjoint dataset, with an untouched external test set.
3. Audit duplicate/near-duplicate sequences and dataset/class imbalance before feature experiments.
4. Compare against majority and simple global-feature baselines, with grouped cross-validation on development data.
5. Analyze errors by pose, occlusion, acquisition condition, and relevant subgroups without inferring sensitive attributes.
6. Consider robust landmark-based descriptors or regularized models only after the baseline is measured honestly.

## Known limitations

No face detector/aligner is included. Real-face performance is unmeasured until a user supplies data. Image-level splits can leak identities; even identity-disjoint splits can retain acquisition bias. Posed expressions and annotator labels may differ from spontaneous appearance. The simple baseline is not clinical or deployment-ready and should not inform consequential decisions about people.
