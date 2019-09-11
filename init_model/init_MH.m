function [M] = init_MH(V1_N_pos,V1_pos,EPS_MEAN,EPS_STD,SIG_MEAN,SIG_STD,KAPPA,FSIZE)
tic; RANDOM_PHI = pi*gpuArray.rand(1,V1_N_pos);
RANDOM_EPS = min(0.95,EPS_MEAN+EPS_STD*gpuArray.randn(1,V1_N_pos));
RANDOM_SIG = SIG_MEAN+SIG_STD*gpuArray.randn(1,V1_N_pos);
M = gpuArray.zeros(V1_N_pos,V1_N_pos); KAPPA2 = KAPPA^2;
for ii = 1:V1_N_pos
    disp(ii/V1_N_pos*100);
    phi = RANDOM_PHI(ii); rmat = [cos(phi) -sin(phi); sin(phi) cos(phi)];
    ecc = RANDOM_EPS(ii); sig = RANDOM_SIG(ii); sig2 = sig*sqrt(1-ecc^2);
    inv_covr = [sig^2 0; 0 sig2^2]^-1;
    temp_pos = V1_pos(:,ii); temp_vec = V1_pos-temp_pos; rot_vec = (rmat*temp_vec)';
    elong_Gaussian = sum(-1/2*rot_vec*inv_covr.*rot_vec,2);
    % Periodic boundary condition
    temp_vec_lxbound = temp_vec+[FSIZE;0]; temp_vec_rxbound = temp_vec+[-FSIZE;0];
    temp_vec_lybound = temp_vec+[0;FSIZE]; temp_vec_uybound = temp_vec+[0;-FSIZE];
    temp_vec_lxlybound = temp_vec+[FSIZE;FSIZE]; temp_vec_rxlybound = temp_vec+[-FSIZE;FSIZE];
    temp_vec_lxuybound = temp_vec+[FSIZE;-FSIZE]; temp_vec_rxuybound = temp_vec+[-FSIZE;-FSIZE];
    rot_vec_lxbound = (rmat*temp_vec_lxbound)'; rot_vec_rxbound = (rmat*temp_vec_rxbound)';
    rot_vec_lybound = (rmat*temp_vec_lybound)'; rot_vec_uybound = (rmat*temp_vec_uybound)';
    rot_vec_lxlybound = (rmat*temp_vec_lxlybound)'; rot_vec_rxlybound = (rmat*temp_vec_rxlybound)';
    rot_vec_lxuybound = (rmat*temp_vec_lxuybound)'; rot_vec_rxuybound = (rmat*temp_vec_rxuybound)';
    elong_Gaussian_lxbound = sum(-1/2*rot_vec_lxbound*inv_covr.*rot_vec_lxbound,2);
    elong_Gaussian_rxbound = sum(-1/2*rot_vec_rxbound*inv_covr.*rot_vec_rxbound,2);
    elong_Gaussian_lybound = sum(-1/2*rot_vec_lybound*inv_covr.*rot_vec_lybound,2);
    elong_Gaussian_uybound = sum(-1/2*rot_vec_uybound*inv_covr.*rot_vec_uybound,2);
    elong_Gaussian_lxlybound = sum(-1/2*rot_vec_lxlybound*inv_covr.*rot_vec_lxlybound,2);
    elong_Gaussian_rxlybound = sum(-1/2*rot_vec_rxlybound*inv_covr.*rot_vec_rxlybound,2);
    elong_Gaussian_lxuybound = sum(-1/2*rot_vec_lxuybound*inv_covr.*rot_vec_lxuybound,2);
    elong_Gaussian_rxuybound = sum(-1/2*rot_vec_rxuybound*inv_covr.*rot_vec_rxuybound,2);
    % MH output connectivity
    out_exct = exp(elong_Gaussian)+...
        exp(elong_Gaussian_lxbound)+exp(elong_Gaussian_rxbound)+...
        exp(elong_Gaussian_lybound)+exp(elong_Gaussian_uybound)+...
        exp(elong_Gaussian_lxlybound)+exp(elong_Gaussian_rxlybound)+...
        exp(elong_Gaussian_lxuybound)+exp(elong_Gaussian_rxuybound);
    out_inhb = -1/KAPPA2*(exp(elong_Gaussian/KAPPA2)+...
        exp(elong_Gaussian_lxbound/KAPPA2)+exp(elong_Gaussian_rxbound/KAPPA2)+...
        exp(elong_Gaussian_lybound/KAPPA2)+exp(elong_Gaussian_uybound/KAPPA2)+...
        exp(elong_Gaussian_lxlybound/KAPPA2)+exp(elong_Gaussian_rxlybound/KAPPA2)+...
        exp(elong_Gaussian_lxuybound/KAPPA2)+exp(elong_Gaussian_rxuybound/KAPPA2));
    M(:,ii) = 1/(2*pi*sig*sig2)*(out_exct+out_inhb);
end
% Normalize so that magnitude of maximal eigenvalue is 1
e = max(abs(eig(M))); M = M/e; M = gather(M); gpuArray(1); toc
end