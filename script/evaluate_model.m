%% Load the trained model
load('models/corrosionNet.mat', 'trainedNet');

%% Load the test data
datasetPath = 'E:\Projects\CorrosionDetectionProject\data';
imdsTest = imageDatastore(fullfile(datasetPath, 'test'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
inputSize = [224 224 3];
augmentedTest = augmentedImageDatastore(inputSize, imdsTest);

%% Predict on test data
predictedLabels = classify(trainedNet, augmentedTest);
trueLabels = imdsTest.Labels;

%% Show accuracy
accuracy = mean(predictedLabels == trueLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

%% Confusion Matrix
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix: Corrosion vs No Corrosion');

%% Visualize predictions on sample images
figure;
numImages = 15;
perm = randperm(numel(imdsTest.Files), numImages);

for i = 1:numImages
    subplot(5, 3, i)
    img = readimage(imdsTest, perm(i));
    resized = imresize(img, inputSize(1:2));
    label = classify(trainedNet, resized);
    imshow(img);
    title(string(label), 'FontSize', 10);
end
sgtitle('Sample Predictions');
