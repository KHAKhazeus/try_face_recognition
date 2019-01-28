clear all;
clc;
cd white;
totalCr = 0;
totalCb = 0;
allValidPointsCR = [];
allValidPointsCB = [];
for i = 1:4
    maskname = strcat(int2str(i), '.mat');
    mask = load(maskname, '-mat');
    maskBW = mask.BW;
    ycbcrmaskOri = rgb2ycbcr(maskBW);
    ycbcrmask = ycbcrmaskOri;
    figure,imshow(ycbcrmask);
    ycrimageOri = ycbcrmask(:,:,3);
%     imshow(ycrimageOri);
    sizeMat = size(ycrimageOri);
    ycrimageOri = imresize(ycrimageOri, 1000/sizeMat(2));
    ycrimageOri = imresize(ycrimageOri, 1000/sizeMat(1));
    figure;imshow(ycrimageOri);
    sizeMat = size(ycrimageOri);
    ycrimage = reshape(ycrimageOri, [1,sizeMat(1)*sizeMat(2)]);
    ycrimage(find(ycrimage==128))=[];
%     imshow(ycrimage)
    allValidPointsCR = horzcat(allValidPointsCR, ycrimage);
    
    
    ycbimageOri = ycbcrmask(:,:,2);
%     imshow(ycbimageOri);
    sizeMat = size(ycbimageOri);
    ycbimage = reshape(ycbimageOri, [1,sizeMat(1)*sizeMat(2)]);
    ycbimage(find(ycbimage==128))=[];
%     imshow(ycbimage)
    allValidPointsCB = horzcat(allValidPointsCB, ycbimage);
end
allValidPointsCR = double(allValidPointsCR);
allValidPointsCB = double(allValidPointsCB);
sizeCR = size(allValidPointsCR,2);
sizeCB = size(allValidPointsCB,2);
if(sizeCR > sizeCB)
    len = sizeCB;
else
    len = sizeCR;
end
allValidPointsCR = allValidPointsCR(:,1:len);
allValidPointsCB = allValidPointsCB(:,1:len);
fprintf('Cb: E: %f,D :%f\nCr: E: %f, D: %f', ...
    mean(allValidPointsCB), std(allValidPointsCB), ...
    mean(allValidPointsCR), std(allValidPointsCR));
cov(allValidPointsCB, allValidPointsCR)
