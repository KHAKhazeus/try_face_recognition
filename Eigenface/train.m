clear;
clc;
% 1.读取图片，拉成等长向量，拼接为像素值*(15*8)的向量
% 2.行中心化
% 3.特征向量
% 4.W^T每一行是一个特征向量
% 5.W^T的特征值进行排列

for people = 1:15
    for faceNumber = 1:8
        if(people < 10)
            path = strcat('./Yale2/training/subject0', int2str(people), '_', ...
                int2str(faceNumber) , '.bmp');
        else
            path = strcat('./Yale2/training/subject', int2str(people), '_', ...
                int2str(faceNumber) , '.bmp');
        end
        face = imread(path);
        [row, col] = size(face);
        faceFlattened = reshape(face, [1,row * col]);
        bigX(:,(people-1) * 8 + faceNumber) = faceFlattened';
    end
end

bigX = double(bigX);

meanVector = mean(bigX, 2);
for i = 1: 15*8
    bigX(:, i) = bigX(:, i) - meanVector;
end

meanFace = reshape(meanVector, [100,100]);
figure,imshow(uint8(meanFace)),title('meanFace');

siCovX = bigX' * bigX;
[siFVec, siFMat] = eig(siCovX);
FVec = bigX * siFVec;
FVecRow = FVec';
totalVecRow = 0;
matSize = size(siFMat);
for i = matSize(1): -1 : 1
    if siFMat(i,i) < 1
        break;
    end
    totalVecRow = totalVecRow + 1;
    compressedFeatureVec(totalVecRow, :) = FVecRow(i, :);
end

compressedSize = size(compressedFeatureVec);
figure
for j = 1: 15
    featureVec = uint8(compressedFeatureVec(j, :));
    featureFace = reshape(featureVec, [100, 100]);
    subplot(4,4,j),imshow(featureFace);
end
save('compressedFeatureVec.mat', 'compressedFeatureVec');
save('bigX.mat', 'bigX');
save('meanVector.mat', 'meanVector');
save('meanFace.mat','meanFace');
