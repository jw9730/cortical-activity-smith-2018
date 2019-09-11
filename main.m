clear all; close all; % close(myVideo);
parallel.gpu.rng(0, 'Philox4x32-10');
%% Paths
addpath('./init_model/');
addpath('./function_utils/');
addpath('./function_integrate/');
addpath('./function_analysis/');
addpath('./EXPORT/'); folderDir = "./EXPORT/"+datestr(now,'yyyy-mm-dd,HH-MM'); mkdir(folderDir);
%% Parameters
FSIZE = 100; % Cortical space
% Connectivity parameters
H = 0.725; % Heterogeneity
EPS_MEAN = H; EPS_STD = 0.13*H; % Eccentricity of Gaussian
SIG_MEAN = 1.8; SIG_STD = 0.1*SIG_MEAN*H; % Gaussian width along elongated axis
KAPPA = 2; % Spatial inhibition scale of MH in respect to excitation
LAMBDA = sqrt(4*pi^2*SIG_MEAN^2*(KAPPA^2-1)/4/log(KAPPA)); % Spatial pattern scale
% Response integration
TAU = 1; % Neuronal time constant
DT = TAU*0.15; % Integration time step
GAMMA = 1.02; % Input amplification
ENG = 0.01; % Input modulation
INTEGRATE = 500; N_EVENTS = 100;
save(folderDir+"/parameters.mat");
%% Model initialization
% Initial V1 locations
GRID = 1:FSIZE; [X,Y] = meshgrid(GRID,GRID);
V1_pos = [reshape(X,[1 FSIZE^2]); reshape(Y,[1 FSIZE^2])]; V1_N_pos = FSIZE^2;
% Heterogenous MH connectivity
M = init_MH(V1_N_pos,V1_pos,EPS_MEAN,EPS_STD,SIG_MEAN,SIG_STD,KAPPA,FSIZE);
figure;
subplot(221); imagesc(reshape(M(:,1),[FSIZE FSIZE])); axis xy image; colormap(jet); colorbar;
subplot(222); imagesc(reshape(M(:,2),[FSIZE FSIZE])); axis xy image; colormap(jet); colorbar;
subplot(223); imagesc(reshape(M(:,3),[FSIZE FSIZE])); axis xy image; colormap(jet); colorbar;
subplot(224); imagesc(reshape(M(:,4),[FSIZE FSIZE])); axis xy image; colormap(jet); colorbar;
%% Random activity
I = zeros(FSIZE*FSIZE,N_EVENTS);
for ii = 1:N_EVENTS
    disp(ii/N_EVENTS*100);
    % Constant input drive
    I_img = function_input_drive(FSIZE,LAMBDA,ENG);
    I(:,ii) = reshape(I_img,[FSIZE*FSIZE 1]);
end
% Numerical integration
O = function_integrate(INTEGRATE,GAMMA,M,I,TAU,DT);
figure;
subplot(121); imagesc(reshape(I(:,1),[FSIZE FSIZE])); axis xy image; colormap(gray); caxis([0.8 1.2]); colorbar;
subplot(122); imagesc(reshape(O(:,1),[FSIZE FSIZE])); axis xy image; colormap(gray); caxis([0 3.6]); colorbar;
%% Correlation pattern measurement
CORR_PATTERN = reshape(gather(corr(O','Type','Pearson')),[FSIZE FSIZE FSIZE FSIZE]);
myVideo = VideoWriter(char(folderDir+"/ActivityCorrPattern_.avi")); myVideo.FrameRate = 10; open(myVideo);
CORR = figure(777); xx = 50;
for yy = 1:100
    C_IMG = squeeze(CORR_PATTERN(yy,xx,:,:));
    CORR = figure(777); clf; imagesc(C_IMG); axis image xy;
    colormap(redblue); colorbar; hold on; plot(xx,yy,'ko','Linewidth',2); caxis([-0.75 0.75]);
    drawnow; writeVideo(myVideo,getframe(CORR));
end; close; close(myVideo); % Close movie frame

ROI_SIZE = 50; N = 100;
PX = []; PY = []; ANGLE = []; CORR_IMAGE = zeros(2*ROI_SIZE,2*ROI_SIZE,N);
CORR = figure(777);
myVideo = VideoWriter(char(folderDir+"/ROI_autocorr.avi")); myVideo.FrameRate = 10; open(myVideo);
for ii = 1:N
    xx = randi(FSIZE-ROI_SIZE)+ROI_SIZE/2; yy = randi(FSIZE-ROI_SIZE)+ROI_SIZE/2;
    x0 = xx-ROI_SIZE/2; y0 = yy-ROI_SIZE/2;
    C_IMG = squeeze(CORR_PATTERN(yy,xx,:,:));
    CORR = figure(777); clf; imagesc(C_IMG); axis image xy; colormap(redblue); colorbar;
    hold on; plot(xx,yy,'ko','Linewidth',2); caxis([-0.75 0.75]);
    rectangle('Position',[x0 y0 ROI_SIZE ROI_SIZE]); drawnow; writeVideo(myVideo,getframe(CORR));
    CORR_IMAGE(:,:,ii) = function_autocorrelation_analysis(1,x0,y0,ROI_SIZE,C_IMG,3,10,false);
    PEAK = FastPeakFind(CORR_IMAGE(:,:,ii)*100); PX = [PX; PEAK(1:2:end)]; PY = [PY; PEAK(2:2:end)];
end; close; close(myVideo); % Close movie frame% Close movie frame
MEAN_IMAGE = mean(CORR_IMAGE,3);
REMOVE = abs(PX-ROI_SIZE)<=5 | sqrt(((PX-ROI_SIZE).^2+(PY-ROI_SIZE).^2))>ROI_SIZE*0.9;
PX(REMOVE) = []; PY(REMOVE) = []; CENTER = ROI_SIZE*(1+1i); ANGLE = abs(angle(PX+1i*PY-CENTER)-pi/2);
ANGLE(ANGLE>pi) = 2*pi-ANGLE(ANGLE>pi); GMModel = fitgmdist(ANGLE*180/pi,2);
ANGLEHIST = figure; subplot(221); imagesc(MEAN_IMAGE); colormap(jet);
hold on; plot(PX,PY,'r+','markersize',2); axis image xy;
subplot(2,2,[3 4]); histogram(ANGLE*180/pi,0:10:180,'Normalization','pdf');
xticks([0 30 60 90 120 150 180]); xlim([0 180]); hold on; fplot(@(x)pdf(GMModel,x),[0 180]);
subplot(222); imagesc(MEAN_IMAGE);
PEAK_R = FastPeakFind(MEAN_IMAGE*100); PX_R = PEAK_R(1:2:end); PY_R = PEAK_R(2:2:end);
hold on; plot(PX_R,PY_R,'r+','markersize',2); axis image xy;
saveas(ANGLEHIST,folderDir+"/model_hist.fig");
%% Spontaneous fracture
FRACTURE = compute_fracture(CORR_PATTERN,FSIZE);
FMAP = figure; imagesc(FRACTURE); axis image xy;
c = colorbar; c.Label.String = "Fracture strength (1/pixel)"; colormap(flipud(gray)); caxis([0 0.5]);
saveas(FMAP,folderDir+"/fracture_map.fig");
