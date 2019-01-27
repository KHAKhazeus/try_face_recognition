for i = 1:3
    filename = int2str(i);
    file = strcat(filename, '.jpeg');
    oriImage = imread(file);
    imageSegmenter(oriImage);
    reply = input('finished?', 's');
    maskpath = strcat(int2str(i),'.mat');
    save(maskpath, 'BW');
    imageSegmenter close
end
    