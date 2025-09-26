function [augmentedTrain, augmentedVal, augmentedTest] = load_and_preprocess()

% Define dataset path manually
datasetPath = 'E:\Projects\CorrosionDetectionProject\data';

% Create image datastores
imdsTrain = imageDatastore(fullfile(datasetPath, 'train'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsVal = imageDatastore(fullfile(datasetPath, 'val'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(fullfile(datasetPath, 'test'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Show label distribution
disp(countEachLabel(imdsTrain));

% Define input image size
inputSize = [224 224 3];

% Define augmentation options
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

% Apply augmentation to training images only
augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);

% No augmentation on validation or test sets
augmentedVal   = augmentedImageDatastore(inputSize, imdsVal);
augmentedTest  = augmentedImageDatastore(inputSize, imdsTest);



end












