% simulate_drone_rl_path_with_3d_overlay.m
clear; clc;

% Load environment and agent
env = CorrosionGridEnv3D;
load(fullfile('E:', 'Projects', 'CorrosionDetectionProject', 'models', 'TrainedDQNDrone.mat'), "agent");

% Run the environment
obs = reset(env);
path = env.Position;
isDone = false;
maxSteps = 100;
stepCount = 0;

% For exploration
epsilon = 0.2;

while ~isDone && stepCount < maxSteps
    stepCount = stepCount + 1;

    if rand < epsilon
        action = randi(length(getActionInfo(env).Elements));
    else
        actionCell = getAction(agent, obs);
        action = actionCell{1};
    end

    [obs, reward, isDone, ~] = step(env, action);
    path(end+1, :) = env.Position;
end

% === 3D Plot ===
map = env.CorrosionMap;
[rows, cols, heights] = size(map);

figure('Color','w');
hold on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('RL Drone Path with Corrosion Map (3D)');
grid on; axis equal; view(3);

% === Corrosion & No-Corrosion Voxels ===
% Store handles for legend
hCorrosion = gobjects(1,1);
hNoCorrosion = gobjects(1,1);

for i = 1:rows
    for j = 1:cols
        for k = 1:heights
            if map(i,j,k) == 1
                hCorrosion = scatter3(j, i, k, 200, ...
                    'filled', 'MarkerFaceColor', [1 0 0], ...
                    'MarkerEdgeColor', 'k', ...
                    'DisplayName', 'Corrosion');
            else
                hNoCorrosion = scatter3(j, i, k, 50, ...
                    'MarkerFaceColor', [0.8 0.8 0.8], ...
                    'MarkerEdgeColor', 'none', ...
                    'MarkerFaceAlpha', 0.1, ...
                    'DisplayName', 'No Corrosion');
            end
        end
    end
end

% === Drone Path ===
hPath = plot3(path(:,2), path(:,1), path(:,3), '-o', ...
    'Color', [0 0.4 1], ...
    'MarkerFaceColor', [0 0.4 1], ...
    'LineWidth', 2, ...
    'DisplayName', 'Drone Path');

% Final Position
hEnd = scatter3(path(end,2), path(end,1), path(end,3), ...
    250, 'g', 'filled', 'DisplayName', 'Final Position');

legend([hCorrosion, hNoCorrosion, hPath, hEnd], ...
    {'Corrosion', 'No Corrosion', 'Drone Path', 'Final Position'}, ...
    'Location', 'northeastoutside');
