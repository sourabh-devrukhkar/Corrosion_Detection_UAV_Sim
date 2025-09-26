%% Simulate Drone Flight with Corrosion Detection Logic

% === Project Root Folder ===
projectRoot = 'E:\Projects\CorrosionDetectionProject';  % <-- Update if needed

% === Subfolder Paths ===
outputDir   = fullfile(projectRoot, 'drone_logic');
imageFolder = fullfile(projectRoot, 'data', 'test');
modelPath   = fullfile(projectRoot, 'models', 'corrosionNet.mat');
inputSize   = [224 224 3]; % ResNet input size

% === Ensure Output Folder Exists ===
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% === Load Trained Network ===
if ~isfile(modelPath)
    error("Trained model not found at: %s", modelPath);
end
load(modelPath, 'trainedNet');

% === Prepare Test Images ===
if ~isfolder(imageFolder)
    error("Test image folder not found at: %s", imageFolder);
end
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% === Open Log File ===
logPath = fullfile(outputDir, 'flight_log.txt');
[logFile, errmsg] = fopen(logPath, 'w');
if logFile == -1
    error("Failed to open log file: %s", errmsg);
end

% === Write Header to Log File ===
fprintf(logFile, "Time\t\tImage\t\t\tPrediction\t\tAction\n");
fprintf(logFile, "------------------------------------------------------------\n");

% === Simulate Drone Flight ===
numImages = min(20, numel(imds.Files)); % Process up to 20 images

for i = 1:numImages
    % Read and resize image
    img = readimage(imds, i);
    resized = imresize(img, inputSize(1:2));

    % Predict class using trained model
    label = classify(trainedNet, resized);

    % Decide drone action based on prediction
    switch char(label)
        case 'no_corrosion'
            action = 'Move Forward';
        case 'mild'
            action = 'Log & Continue';
        case 'moderate'
            action = 'Hover & Capture';
        case 'severe'
            action = 'Stop & Alert';
        otherwise
            action = 'Unknown';
    end

    % Extract image filename
    [~, imgName, ext] = fileparts(imds.Files{i});
    fileStr = [imgName ext];

    % Get current timestamp
    timestamp = string(datetime('now','Format','HH:mm:ss'));

    % Print to console
    fprintf('[%02d] Image: %-20s | Prediction: %-12s | Action: %s\n', ...
        i, fileStr, char(label), action);

    % Write to log file
    fprintf(logFile, "%s\t%-20s\t%-12s\t%s\n", ...
        timestamp, fileStr, char(label), action);

    % Pause to simulate drone delay
    pause(1);
end

% === Close Log File ===
fclose(logFile);

disp("Drone flight simulation complete. Check 'drone_logic/flight_log.txt'");
