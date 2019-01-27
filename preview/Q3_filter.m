image = imread('meow.jpeg');
image = im2double(image);
% imshow(image);
gray = rgb2gray(image);
imshow(gray);
result = figure(1);
subplot(231);
imshow(image);
subplot(232);
imshow(gray);
sizemat = size(image);
horzero = zeros(1,sizemat(2)+2,3);
verzero = zeros(sizemat(1),1,3);
imageext = horzcat(verzero,image, verzero);
imageext = vertcat(horzero,imageext, horzero);
% subplot(133);
% imshow(imageext);
filterOne = [-1,-1,-1;-1,9,-1;-1,-1,-1];
filterSecond = [1/16,2/16,1/16;2/16,2/16,2/16;1/16,2/16,1/16];
filterPrewitt = [-1,0,1;-2,0,2;-1,0,1];
filterLaplacian = [1,1,1;1,-8,1;1,1,1];
changedImageOne = image;
changedImageSecond = image;
changedImageThird = image;
changedImageFourth = image;
for row = 2:sizemat(1) - 1
    for col = 2:sizemat(2) - 1
        for rgb = 1: 3
            subimage = image(row-1:row+1, col-1:col+1, rgb);
            sumOfAdjOne = sum(sum(subimage .* filterOne));
            changedImageOne(row, col, rgb) = sumOfAdjOne;
            sumOfAdjSecond = sum(sum(subimage .* filterSecond));
            changedImageSecond(row, col, rgb) = sumOfAdjSecond;
            sumOfAdj = sum(sum(subimage .* filterPrewitt));
            changedImageThird(row, col, rgb) = sumOfAdj;
            sumOfAdj = sum(sum(subimage .* filterLaplacian));
            changedImageFourth(row,col,rgb) = sumOfAdj;
        end
    end
end
subplot(233);
imshow(changedImageOne);
subplot(234);
imshow(changedImageSecond);
subplot(235);
imshow(changedImageThird);
subplot(236);
imshow(changedImageFourth);
