clear;
clc;

meanFacemat = load('meanFace.mat');
meanFace = meanFacemat.meanFace;
figure, imshow(uint8(meanFace)), title('meanFace');
compressedFeatureVecmat = load('compressedFeatureVec.mat');
compressedFeatureVec = compressedFeatureVecmat.compressedFeatureVec;
bigXmat = load('bigX.mat');
meanVector = load('meanVector.mat');
meanVector = meanVector.meanVector;
bigX = bigXmat.bigX;
[row, rep] = size(bigX);

figure,title('featureFace')
for i = 1:3code
    for j = 1:3
        featureFaceVec = compressedFeatureVec((i - 1)*3 + j , :);
        featureFace = reshape(featureFaceVec, [100, 100]);
        subplot(3, 3, (i - 1)*3 + j);
        imshow(uint8(featureFace));
    end
end

% testImage = imread('./Yale2/training/subject15_5.bmp');
% testSize = size(testImage);
% testVec = reshape(testImage, [1, testSize(1) * testSize(2)]);
% testVec = double(testVec');
% testVec = testVec - meanVector;
% compressedX = compressedFeatureVec * bigX;
% compressedTest = compressedFeatureVec * testVec;
% minDistance = cos_distance(compressedX(:, 1), compressedTest);
% pos = 1;
% for j = 1: rep
%     newDistance = cos_distance(compressedX(:, j), compressedTest);
%     if newDistance < minDistance
%         minDistance = newDistance;
%         pos = j;
%     end
% end
% guess = floor((pos - 1) /8);
% fprintf("这人大概是对象%d", uint8(guess + 1));

total = 0;
correct = 0;

for i = 1: 15
    for j = 9: 11
        if i >=10
            path = strcat('./Yale2/test/subject', int2str(i),'_',...
                int2str(j), '.bmp');
        else
            path = strcat('./Yale2/test/subject0',int2str(i),'_',...
                int2str(j), '.bmp');
        end
        testImage = imread(path);
        testSize = size(testImage);
        testVec = reshape(testImage, [1, testSize(1) * testSize(2)]);
        testVec = double(testVec');
        testVec = testVec - meanVector;
        compressedX = compressedFeatureVec * bigX;
        compressedTest = compressedFeatureVec * testVec;
        minDistance = cos_distance(compressedX(:, 1), compressedTest);
        pos = 1;
        for k = 1: rep
            newDistance = cos_distance(compressedX(:, k), compressedTest);
            if newDistance < minDistance
                minDistance = newDistance;
                pos = k;
            end
        end
        guess = uint8(floor((pos - 1) /8) + 1);
%         fprintf("这人大概是对象%d", uint8(guess + 1));
        if(guess == i)
            correct = correct + 1;
        end
        total = total + 1;
    end
end

fprintf('准确率:%f', (correct/total) * 100);

function s = cos_distance(x, y)

all = vertcat(x', y');
s = pdist(all,'cosine');

end