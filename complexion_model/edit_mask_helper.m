for i = 4:4
    filename = int2str(i);
    file = strcat(filename, '.jpeg');
    oriImage = imread(file);
    imageSegmenter(oriImage);
    reply = input('finished?', 's');
    maskpath = strcat(int2str(i),'.mat');
    save(maskpath, 'BW');
    imageSegmenter close
end

% for i = 1:7
%     filename = int2str(i);
%     file = strcat(filename, '.mat');
%     mask = load(file);
%     figure,imshow(mask.BW);
% end
    
    