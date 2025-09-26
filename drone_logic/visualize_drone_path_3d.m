%% 3D Visualization of Drone Path with Corrosion Detection

% === Configuration ===
projectRoot = 'E:\Projects\CorrosionDetectionProject';  % <-- adjust if needed
modelPath   = fullfile(projectRoot, 'models', 'corrosionNet.mat');
imageFolder = fullfile(projectRoot, 'data', 'test');
inputSize   = [224 224 3];

% === Load Model and Dataset ===
load(modelPath, 'trainedNet');
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

numPoints = min(20, numel(imds.Files));

% Simulate 3D coordinates (grid layout with altitude variations)
rows = 5;
cols = 4;
[xGrid, yGrid] = meshgrid(1:cols, 1:rows);
xPath = reshape(xGrid', [], 1);
yPath = reshape(yGrid', [], 1);

% Simulate altitude (Z-axis) — can vary based on severity
zPath = zeros(numPoints,1);
labels = strings(numPoints,1);
colors = zeros(numPoints,3);

% Define class-to-color mapping
classColors = containers.Map(...
    {'no_corrosion','mild','moderate','severe'}, ...
    {[0.3 0.8 0.3], [0.9 0.8 0.1], [1 0.5 0], [0.9 0.1 0.1]});
classZ = containers.Map(...
    {'no_corrosion', 'mild', 'moderate', 'severe'}, ...
    [5, 7, 10, 12]);  % altitude per severity

% Predict and assign values
for i = 1:numPoints
    img = readimage(imds, i);
    resized = imresize(img, inputSize(1:2));
    label = classify(trainedNet, resized);
    labels(i) = string(label);

    if isKey(classZ, label)
        zPath(i) = classZ(label);  % assign altitude
    else
        zPath(i) = 6;
    end

    if isKey(classColors, label)
        colors(i,:) = classColors(label);  % assign color
    else
        colors(i,:) = [0.5 0.5 0.5];
    end
end

% === 3D Plot ===
figure('Name','Drone Path 3D','Color','w');
hold on; grid on; axis equal;
title('3D Drone Path Based on Corrosion Severity','FontSize',14);
xlabel('X (meters)'); ylabel('Y (meters)'); zlabel('Altitude (meters)');

% Plot colored waypoints
for i = 1:numPoints
    plot3(xPath(i), yPath(i), zPath(i), 'o', ...
        'MarkerSize', 10, ...
        'MarkerFaceColor', colors(i,:), ...
        'MarkerEdgeColor', 'k');
    text(xPath(i)+0.1, yPath(i), zPath(i)+0.5, labels(i), 'FontSize', 8);
end

% Plot drone path line
plot3(xPath(1:numPoints), yPath(1:numPoints), zPath(1:numPoints), ...
    '--k', 'LineWidth', 1.5);

view(45, 25); % 3D viewing angle
legend({'Drone Path'}, 'Location','best');
