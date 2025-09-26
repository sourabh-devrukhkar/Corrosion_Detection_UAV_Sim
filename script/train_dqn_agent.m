% train_dqn_agent.m


% Load environment
env = CorrosionGridEnv3D;

% Get observation and action specifications
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Define the DQN neural network
statePath = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numel(actInfo.Elements), 'Name', 'output')
];

dnn = layerGraph(statePath);

% Q-value function
critic = rlVectorQValueFunction(dnn, obsInfo, actInfo, ...
    ObservationInputNames="state");

% DQN agent options
agentOpts = rlDQNAgentOptions( ...
    UseDoubleDQN=true, ...
    TargetSmoothFactor=1e-3, ...
    TargetUpdateFrequency=4, ...
    ExperienceBufferLength=1e5, ...
    DiscountFactor=0.99, ...
    MiniBatchSize=64);

% Set epsilon exploration properly
explorationOpts = rl.option.EpsilonGreedyExploration;
explorationOpts.Epsilon = 1.0;
explorationOpts.EpsilonMin = 0.05;
explorationOpts.EpsilonDecay = 0.995;
agentOpts.EpsilonGreedyExploration = explorationOpts;

% Create agent
agent = rlDQNAgent(critic, agentOpts);

% Training options
trainOpts = rlTrainingOptions( ...
    MaxEpisodes=1500, ...
    MaxStepsPerEpisode=50, ...
    Verbose=false, ...
    Plots="training-progress", ...
    StopTrainingCriteria="AverageReward", ...
    StopTrainingValue=90, ...
    ScoreAveragingWindowLength=30);

% Train agent
trainingStats = train(agent, env, trainOpts);

% Save the trained agent to local folder
if ~exist("models", "dir")
    mkdir("models");
end
modelPath = fullfile('E:', 'Projects', 'CorrosionDetectionProject', 'models', 'TrainedDQNDrone.mat');
save(modelPath, 'agent');

disp("Training complete.");

