classdef Integrator_3D < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','z','vx','vy','vz'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            if nargin < 5 || isempty(u)
                u = [0 0 0]';
            end
            F = obj.getStateMat(timestep,dt,state);
            G = obj.getInputMat(timestep,dt,state);
            nextState = F*state + G*u;
        end
        function stateMat = getStateMat(obj, timestep, dt, state, varargin)
            stateMat = [1 0 0 dt 0 0; 0 1 0 0 dt 0; 0 0 1 0 0 dt; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1];
        end
        function inputMat = getInputMat(obj, timestep, dt, state, varargin)
            inputMat = [dt 0 0; 0 dt 0; 0 0 dt; 1 0 0; 0 1 0; 0 0 1];
        end
    end
end 