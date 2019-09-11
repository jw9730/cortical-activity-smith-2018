function [V1_output] = function_integrate(INTEGRATE,GAMMA,M,I,TAU,DT)
M = gpuArray(M);
V1_output = 0.1*gpuArray.rand(size(I)); % Randomly initialized
for step = 1:INTEGRATE
    disp(step/INTEGRATE*100);
    V1_input = GAMMA*M*V1_output+I;
    delta = 1/TAU*(-V1_output+abs(V1_input));
    V1_output = V1_output+DT*delta;
end; V1_output = gather(V1_output);
end