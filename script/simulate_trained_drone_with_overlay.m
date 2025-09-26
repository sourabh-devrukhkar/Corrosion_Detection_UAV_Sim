%===  simulate_trained_drone_with_overlay.m (With Exploration & Heatmap) ===%

% Load environment and trained agent
env = CorrosionGridEnv3D;
modelPath = fullfile('E:', 'Projects', 'CorrosionDetectionProject', 'models', 'TrainedDQNDrone.mat');
load(modelPath, 'agent');

% Reset environment
obs = reset(env);
isDone = false;
stepCount = 0;
maxSteps = 100;
path = [];
epsilon_sim = 0.2;  %  20% exploration for variability

% Simulate drone path
while ~isDone && stepCount < maxSteps
    stepCount = stepCount + 1;

    % Add exploration
    if rand < epsilon_sim
        actionInfo = getActionInfo(env);              %  safe access
        action = randi(length(actionInfo.Elements));  % random action
    else
        actionCell = getAction(agent, {obs});  % Policy action
        action = actionCell{1};
    end

    % Step environment
    [obs, reward, isDone, ~] = step(env, action);
    path(end+1, :) = env.Position;
end

%===  Overlay Plot (Corrosion + Path + Heatmap) ===%
figure;
hold on;

% Plot corrosion points
[corX, corY, corZ] = ind2sub(size(env.CorrosionMap), find(env.CorrosionMap));
scatter3(corX, corY, corZ, 100, 'red', 'filled');

% Plot drone path
plot3(path(:,1), path(:,2), path(:,3), 'b-', 'LineWidth', 2);
scatter3(path(end,1), path(end,2), path(end,3), 150, 'green', 'filled');

% Plot visit heatmap
[vx, vy, vz] = ind2sub(size(env.VisitCount), find(env.VisitCount > 0));
intensity = env.VisitCount(env.VisitCount > 0);
scatter3(vx, vy, vz, 80, intensity, 'filled');

% Labels & legend
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Drone Path + Corrosion Map + Heatmap (with Exploration)');
legend({'Corrosion Points', 'Drone Path', 'Final Position', 'Visit Heatmap'}, 'Location', 'bestoutside');
colorbar; grid on; view(3);
hold off;
