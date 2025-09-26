classdef CorrosionGridEnv3D < rl.env.MATLABEnvironment
    % A custom 3D grid environment for drone corrosion inspection using RL

    properties
        GridSize = [5, 5, 3];
        MaxSteps = 100;
        Position
        CorrosionMap
        Visited
        NumSteps
        VisitCount     %  Tracks how many times each cell is visited
    end

    properties(Access = protected)
        IsDone = false;
    end

    methods
        function this = CorrosionGridEnv3D()
            ObservationInfo = rlNumericSpec([3 1]);
            ObservationInfo.Name = 'DronePosition';

            ActionInfo = rlFiniteSetSpec([1 2 3 4 5 6]);
            ActionInfo.Name = 'DroneMove';

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            this.reset();
        end

        function obs = reset(this)
            this.Position = [1, 1, 1];
            this.NumSteps = 0;
            this.Visited = zeros(this.GridSize);
            this.VisitCount = zeros(this.GridSize);  % Reset heatmap

            % Generate corrosion map
            corrosionDensity = 0.2;
            this.CorrosionMap = rand(this.GridSize) < corrosionDensity;
            this.CorrosionMap(1,1,1) = 0; % Start cell = no corrosion
            obs = this.Position';
            this.IsDone = false;
        end

        function [obs, reward, isDone, loggedSignals] = step(this, action)
            loggedSignals = [];
            this.NumSteps = this.NumSteps + 1;

            % Move drone
            pos = this.Position;
            switch action
                case 1, pos(1) = min(pos(1) + 1, this.GridSize(1)); % +X
                case 2, pos(1) = max(pos(1) - 1, 1);                % -X
                case 3, pos(2) = min(pos(2) + 1, this.GridSize(2)); % +Y
                case 4, pos(2) = max(pos(2) - 1, 1);                % -Y
                case 5, pos(3) = min(pos(3) + 1, this.GridSize(3)); % +Z
                case 6, pos(3) = max(pos(3) - 1, 1);                % -Z
            end
            this.Position = pos;

            % Update visit count
            this.VisitCount(pos(1), pos(2), pos(3)) = this.VisitCount(pos(1), pos(2), pos(3)) + 1;

            % Reward logic
            if this.CorrosionMap(pos(1), pos(2), pos(3))
                reward = 1;
            elseif this.Visited(pos(1), pos(2), pos(3)) > 0
                reward = -0.1;
            else
                reward = 0;
            end

            this.Visited(pos(1), pos(2), pos(3)) = 1;

            % Termination
            isDone = this.NumSteps >= this.MaxSteps;
            this.IsDone = isDone;
            obs = pos';
        end
    end
end