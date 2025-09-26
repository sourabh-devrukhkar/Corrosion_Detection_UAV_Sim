%% Grad-CAM Visualization with Blending and Export

% Load trained network
load('models/corrosionNet.mat', 'trainedNet');

% Path to test set
datasetPath = 'E:\Projects\CorrosionDetectionProject\data';
imdsTest = imageDatastore(fullfile(datasetPath, 'test'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Create output folder
outputDir = 'outputs/gradcam_images';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Define input size based on the network
inputSize = [224 224 3];

% Number of images to visualize
N = 10;
perm = randperm(numel(imdsTest.Files), N);

for i = 1:N
    % Read and resize image
    img = readimage(imdsTest, perm(i));
    resized = imresize(img, inputSize(1:2));

    % Get predicted label
    label = classify(trainedNet, resized);

    % Grad-CAM score map (default last conv layer)
    scoreMap = gradCAM(trainedNet, resized, label);

    % Convert scoreMap to RGB heatmap
    heatmapRGB = ind2rgb(uint8(scoreMap * 255), jet(256));
    heatmapRGB = imresize(heatmapRGB, inputSize(1:2));

    % Blend original image with Grad-CAM heatmap
    fusedImage = imfuse(resized, heatmapRGB, 'blend');

    % Show blended result
    figure;
    imshow(fusedImage);
    title(['Prediction: ' char(label)], 'FontSize', 14);

    % Save to output folder
    [~, name, ~] = fileparts(imdsTest.Files{perm(i)});
    saveName = fullfile(outputDir, ['gradcam_' name '_' char(label) '.jpg']);
    imwrite(fusedImage, saveName);
end
