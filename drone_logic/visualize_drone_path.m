%% Visualize Drone Path Based on Corrosion Detection

% Path to test images and trained model
projectRoot = 'E:\Projects\CorrosionDetectionProject';
imageFolder = fullfile(projectRoot, 'data', 'test');
modelPath   = fullfile(projectRoot, 'models', 'corrosionNet.mat');

% Load trained network
load(modelPath, 'trainedNet');

% Load test images
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

inputSize = [224 224 3];
numPoints = min(20, numel(imds.Files)); % Number of steps to simulate

% Define grid layout: 5x4 (rows x columns)
rows = 5;
cols = 4;
[xGrid, yGrid] = meshgrid(1:cols, 1:rows);
xPath = reshape(xGrid', [], 1);
yPath = reshape(yGrid', [], 1);

% Prepare results
labels = strings(numPoints,1);
actions = strings(numPoints,1);

for i = 1:numPoints
    img = readimage(imds, i);
    resized = imresize(img, inputSize(1:2));
    label = classify(trainedNet, resized);
    labels(i) = string(label);

    switch char(label)
        case 'no_corrosion'
            actions(i) = "Move";
        case 'mild'
            actions(i) = "Log";
        case 'moderate'
            actions(i) = "Hover";
        case 'severe'
            actions(i) = "Stop";
        otherwise
            actions(i) = "Unknown";
    end
end

% Map classes to colors
classColors = containers.Map(...
    {'no_corrosion','mild','moderate','severe'}, ...
    {[0.3 0.8 0.3], [0.9 0.8 0.1], [1 0.5 0], [0.9 0.1 0.1]});

figure('Name','Drone Path Simulation','Color','w');
hold on; grid on;
title('Drone Path Over Inspection Grid');
xlabel('Plant X Coordinate');
ylabel('Plant Y Coordinate');

for i = 1:numPoints
    class = labels(i);
    color = [0.5 0.5 0.5]; % default gray
    if isKey(classColors, class)
        color = classColors(class);
    end

    % Draw point
    plot(xPath(i), yPath(i), 'o', ...
        'MarkerSize', 12, ...
        'MarkerFaceColor', color, ...
        'MarkerEdgeColor', 'k');

    % Annotate with action
    text(xPath(i)+0.1, yPath(i), actions(i), 'FontSize', 8);
end

% Draw drone path
plot(xPath(1:numPoints), yPath(1:numPoints), '--k', 'LineWidth', 1.5);

legend({'Drone Path'}, 'Location','best');
axis equal;
xlim([0 cols+1]); ylim([0 rows+1]);
