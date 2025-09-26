%%  3D Animated Drone Flight Over Corrosion Zones


% === Configuration ===
projectRoot = 'E:\Projects\CorrosionDetectionProject';  % <-- adjust path if needed
modelPath   = fullfile(projectRoot, 'models', 'corrosionNet.mat');
imageFolder = fullfile(projectRoot, 'data', 'test');
inputSize   = [224 224 3];

% === Load model and dataset ===
load(modelPath, 'trainedNet');
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

numPoints = min(20, numel(imds.Files));
rows = 5;
cols = 4;
[xGrid, yGrid] = meshgrid(1:cols, 1:rows);
xPath = reshape(xGrid', [], 1);
yPath = reshape(yGrid', [], 1);

% === Severity → Color + Altitude ===
classColors = containers.Map(...
    {'no_corrosion','mild','moderate','severe'}, ...
    {[0.3 0.8 0.3], [0.9 0.8 0.1], [1 0.5 0], [0.9 0.1 0.1]});
classZ = containers.Map(...
    {'no_corrosion', 'mild', 'moderate', 'severe'}, ...
    [5, 7, 10, 12]);  % Simulated altitude per class

% === Preallocate for results ===
zPath = zeros(numPoints,1);
labels = strings(numPoints,1);
colors = zeros(numPoints,3);

% === Predict Labels and Assign Altitude + Colors ===
for i = 1:numPoints
    img = readimage(imds, i);
    resized = imresize(img, inputSize(1:2));
    label = classify(trainedNet, resized);
    labels(i) = string(label);

    if isKey(classZ, label)
        zPath(i) = classZ(label);
    else
        zPath(i) = 6;
    end

    if isKey(classColors, label)
        colors(i,:) = classColors(label);
    else
        colors(i,:) = [0.5 0.5 0.5]; % Default gray
    end
end

% === Initialize 3D Plot ===
figure('Name','Animated Drone Flight (3D)','Color','w');
hold on; grid on; axis equal;
xlabel('X'); ylabel('Y'); zlabel('Altitude (Z)');
title('Drone Inspection Path in 3D','FontSize',14);

% === Plot Waypoints with Transparency ===
for i = 1:numPoints
    scatter3(xPath(i), yPath(i), zPath(i), ...
        120, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.25);
end

% === Initialize Drone Marker (moving point) ===
droneMarker = plot3(xPath(1), yPath(1), zPath(1), ...
    'ko', 'MarkerSize', 12, 'MarkerFaceColor', 'c');

% === Animate the Path ===
for i = 1:numPoints
    % Update drone position
    set(droneMarker, ...
        'XData', xPath(i), ...
        'YData', yPath(i), ...
        'ZData', zPath(i), ...
        'MarkerFaceColor', colors(i,:));

    % Draw line from previous point
    if i > 1
        plot3(xPath(i-1:i), yPath(i-1:i), zPath(i-1:i), ...
            '-k', 'LineWidth', 2);
    end

    % Annotate with class name
    labelStr = sprintf("Step %d: %s", i, labels(i));
    title(['Drone Inspecting → ' labelStr], 'FontSize', 14);

    pause(1); % Simulate real-time delay
end

% Final marker at the end
plot3(xPath(numPoints), yPath(numPoints), zPath(numPoints), ...
    'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'g');

view(40, 25);
disp("3D drone animation completed.");
