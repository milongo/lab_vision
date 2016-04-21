function TextonsCNN

% Note: RUN THIS FUNCTION ONLY: Has to be inside 'practical-cnn-2015a'
% directory provided by Andrea Vedaldi

setup;
if ~exist('textonsdbs.mat')
    setupdata;
end

% train_net;

if exist('data/textons-experiment/textonscnn.mat');
    net = load('data/textons-experiment/textonscnn.mat');
else
    s = dir('data/textons-experiment');
    for i=3:numel(s)
        ll = find(s(i).name== '.');
        num(i-2) = str2double(s(i).name(ll-1));
    end
    str = strcat('net-epoch-',int2str(max(num)),'.mat');
    net = load(strcat('data/textons-experiment/',str));
end

ex = exist('imdb', 'var');
if ex~=1
    imdb = load('textonsdbs.mat');
end


test_data = imdb.images.data(:,:,imdb.images.set==3);
% imageMean = mean(imdb.images.data(:)); %0.3947
test_data = test_data - 0.3947;
res = test_net(net,test_data);

truth = imdb.images.label(imdb.images.set==3);

CM = confusionmat(truth,res);
figure;imshow(CM,[],'InitialMagnification',1600);
title('Confusion matrix'); xlabel('Predictions'); ylabel('Groundtruth');

function setupdata

imdb = load('textonsdb.mat');
xx = imdb.images.data;
xx = im2single(xx);
imdb.images.data = xx;

imdb.images.set(18751:end)=3;

for i=1:18750
    if mod(i,750)==0
        for j=(i-225):(i);
            imdb.images.set(j) = 2;
        end
    end   
end

images = imdb.images;
meta = imdb.meta;
save('textonsdbs.mat','images','meta');
