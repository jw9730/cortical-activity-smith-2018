function [I] = function_input_drive(FSIZE,LAMBDA,ENG)
margin = 3;
temp_G = gather(gpuArray.randn(margin*FSIZE,margin*FSIZE));
max_scale = max(max(temp_G)); min_scale = min(min(temp_G));
temp_G = 255*(temp_G-min_scale)/(max_scale-min_scale);
f_low = margin*FSIZE/LAMBDA*0.1; f_high = margin*FSIZE/LAMBDA*1.9;
G = double(gaussianbpf(temp_G,f_low,f_high));
refpoint = margin*FSIZE/2-FSIZE/2; window = FSIZE-1;
G = G(refpoint:refpoint+window,refpoint:refpoint+window);
G = (G-mean(mean(G)))/std2(G);
I = 1+ENG*G;
end