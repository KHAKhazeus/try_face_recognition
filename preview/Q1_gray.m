image = imread('meow.jpeg');
% imshow(image);
gray = rgb2gray(image);
imshow(gray);
result = figure(1);
subplot(121);
imshow(image);
subplot(122);
imshow(gray);
disp("灰度100以上: ");
number = sum(sum(gray > 100));
disp(number)