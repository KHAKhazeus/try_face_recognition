image = imread('meow.jpeg');
% imshow(image);
gray = rgb2gray(image);
imshow(gray);
result = figure(1);
subplot(121);
imshow(image);
subplot(122);
imshow(gray);
upOneHundred = gray > 100;
[row, col] = find(upOneHundred);
rowMean = round(mean(row));
colMean = round(mean(col));
disp('最近点:')
fprintf('[%f,%f]\n', rowMean, colMean)