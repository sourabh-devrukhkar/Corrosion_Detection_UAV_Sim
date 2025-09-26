function newPos = updatePosition(pos, action)
% Updates position in 3D based on action
% pos: [x; y; z], action: scalar (1–6), returns newPos: [x; y; z]

% Ensure action is scalar
if iscell(action)
    action = action{1};
end
action = double(action);  % convert from any type to scalar

newPos = pos;

switch action
    case 1 % +X
        newPos(1) = newPos(1) + 1;
    case 2 % -X
        newPos(1) = newPos(1) - 1;
    case 3 % +Y
        newPos(2) = newPos(2) + 1;
    case 4 % -Y
        newPos(2) = newPos(2) - 1;
    case 5 % +Z
        newPos(3) = newPos(3) + 1;
    case 6 % -Z
        newPos(3) = newPos(3) - 1;
    otherwise
        warning('Invalid action passed to updatePosition: %s', num2str(action));
end
end
