% Load image datastores from preprocessing
[augmentedTrain, augmentedVal, augmentedTest] = load_and_preprocess();

%% Step 1: Load Pre-trained ResNet-18 and Modify Final Layers
net = resnet18;
lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;
numClasses = 4;

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fcNew')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')];

lgraph = replaceLayer(lgraph, 'fc1000', newLayers(1));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newLayers(3));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));


%% Step 2: Set Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augmentedVal, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'Plots', 'training-progress');


%% Step 3: Train the Network
trainedNet = trainNetwork(augmentedTrain, lgraph, options);

if ~exist('models', 'dir')
    mkdir('models');
end

modelPath = fullfile('E:', 'Projects', 'CorrosionDetectionProject', 'models', 'corrosionNet.mat');
save(modelPath, 'trainedNet');

