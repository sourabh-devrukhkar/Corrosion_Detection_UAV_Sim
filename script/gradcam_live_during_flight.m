% gradcam_live_during_flight.m

% 1. Load trained ResNet model
modelPath = fullfile('E:', 'Projects', 'CorrosionDetectionProject', 'models', 'corrosionNet.mat');

if ~isfile(modelPath)
    error("Model not found at: %s", modelPath);
end
load(modelPath, 'trainedNet');

% 2. Set test image folder (change if needed)
testImageFolder = fullfile('E:', 'Projects', 'CorrosionDetectionProject', 'data', 'test');
imds = imageDatastore(testImageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 3. Set output folder for Grad-CAM results
outputFolder = fullfile('gradcam_outputs');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 4. Loop over test images
inputSize = trainedNet.Layers(1).InputSize;
for i = 1:min(20, numel(imds.Files))  % Limit to 20 samples for speed
    img = readimage(imds, i);
    resizedImg = imresize(img, inputSize(1:2));

    % Predict
    [label, scores] = classify(trainedNet, resizedImg);
    
    % Generate Grad-CAM heatmap
    scoreMap = gradCAM(trainedNet, resizedImg, label);

    % Overlay
    overlayImg = imfuse(resizedImg, scoreMap, ...
    'Method', 'blend', ...
    'Scaling', 'joint');

    
    % Display
    figure(1); clf;
    imshow(overlayImg);
    title(sprintf("Prediction: %s | Confidence: %.2f%%", ...
        string(label), max(scores) * 100));
    
    % Save output
    outputPath = fullfile(outputFolder, sprintf("gradcam_%02d.png", i));
    imwrite(overlayImg, outputPath);
    
    pause(0.5);  % pause for visibility
end

disp(" Grad-CAM overlay completed. Check gradcam_outputs/");
