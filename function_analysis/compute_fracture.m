function [FRACTURE] = compute_fracture(CORR_PATTERN,FSIZE)
tic; disp("Measuring fracture map");
CORR_PATTERN = gpuArray(CORR_PATTERN);
FRACTURE = gpuArray(zeros(FSIZE,FSIZE));
for yy = 1:FSIZE
    for xx = 1:FSIZE
        % Fracture strength
        if (yy == 1); dy = 1; else; dy = -1; end
        if (xx == 1); dx = 1; else; dx = -1; end
        C = reshape(CORR_PATTERN(yy,xx,:,:),[1 FSIZE*FSIZE]);
        Cdy = reshape(CORR_PATTERN(yy+dy,xx,:,:),[1 FSIZE*FSIZE]);
        Cdx = reshape(CORR_PATTERN(yy,xx+dx,:,:),[1 FSIZE*FSIZE]);
        Fdy = 1-(sum(C.*Cdy)-FSIZE*FSIZE*mean(C)*mean(Cdy))/((FSIZE*FSIZE-1)*std(C)*std(Cdy));
        Fdx = 1-(sum(C.*Cdx)-FSIZE*FSIZE*mean(C)*mean(Cdx))/((FSIZE*FSIZE-1)*std(C)*std(Cdx));
        FRACTURE(yy,xx) = sqrt(Fdx.^2+Fdy.^2);
    end
end
FRACTURE = gather(FRACTURE); toc
end