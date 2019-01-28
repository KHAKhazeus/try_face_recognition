clear all;  
clc;
M = [113.834270 149.178237]'  ;
Sigma = [34.8830   -7.4244
   -7.4244   81.4047];
testImg = imread('testbig.jpeg');
figure,imshow(testImg),title('原始图像');
ycbcrTestImg = rgb2ycbcr(testImg);
oriImgSize = size(ycbcrTestImg);
if oriImgSize(2) > 1000 || oriImgSize(1) > 1000
    if(oriImgSize(2) > oriImgSize(1))
        rate = oriImgSize(2)/1000;
    else
        rate = oriImgSize(1)/1000;
    end
    testImg = imresize(testImg, 1/rate);
    ycbcrTestImg = rgb2ycbcr(testImg);
end
imgSize = size(ycbcrTestImg);
cbcr = zeros(2,1);
BinImg = zeros([imgSize(1), imgSize(2)]);
P = zeros([imgSize(1), imgSize(2)]);
for row = 1: imgSize(1)
    for col =1: imgSize(2)
        cbcr(2) = ycbcrTestImg(row,col,3); 
        cbcr(1) = ycbcrTestImg(row,col,2);
        P(row,col) = exp(-0.5*((cbcr-M)')*(inv(Sigma))*(cbcr-M));%计算似然度
        if P(row,col) > 0.15
             BinImg(row,col) = 1;%生成二值图像 
        end     
    end
end
figure,imshow(BinImg),title('二值化');

disk = strel('disk',2);
openBin=imopen(BinImg,disk);
% figure,imshow(BinImg),title('中间');
% [label, num] = bwlabel(BinImg, 8); % 区域标记
figure,imshow(openBin),title('中间');
[label, num] = bwlabel(openBin, 8); % 区域标记
stats = regionprops(label, 'BoundingBox'); % 得到包围矩形框
box = cat(1, stats.BoundingBox);
box = double(box);
[rowSize, olSize] = size(box);
sizeMat = box(:,3).*box(:,4);
mx = max(box(:,3).*box(:,4))/10;
figure,imshow(testImg),title('结果');
hold on;
for k = 1:rowSize
    p = box(k, 3)*box(k, 4);
    (double(box(k, 4))/double(box(k, 3)))
    if p>mx && 0.5 < (double(box(k, 4))/double(box(k, 3))) ...
        && (double(box(k, 4))/double(box(k, 3)))< 3
        rectangle('Position', box(k, :),'EdgeColor', 'r', 'LineWidth', 3);
    end
end
hold off;
% disk = strel('disk',2);
% openBin=imopen(BinImg,disk);
% figure,imshow(openBin),title('open')

% [label, number] = bwlabel(openBin,4);
% stats = regionprops(label,'basic');
% centroids = cat(1, stats.Centroid);
% figure,imshow(testImg),title('centroids')  
% hold on
% rects = cat(1,  stats.BoundingBox);
% plot(centroids(:,1), centroids(:,2), 'b*'),
% for i = 1:size(rects, 1)
%         rectangle('position', rects(i, :), 'EdgeColor', 'r');  
% end  
% hold off