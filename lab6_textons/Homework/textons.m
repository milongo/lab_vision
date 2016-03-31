addpath('lib','test','train'); 
train_dir = dir('train');
train_dir = train_dir(3:end-1);
test_dir = dir('test');
test_dir = test_dir(3:end-1);

[fb] = fbCreate;
k = 50;
sz = size(imread(train_dir(1).name));

% Preallocate arrays
c_MapTrain = cell((numel(train_dir)),3); % Array for storing train maps and histograms
c_MapTest = cell((numel(test_dir)),4);   % Array for storing test maps and histograms
c_FilteredTrain = cell((numel(train_dir)),1); % Array for storing entire train filtered dataset
im = zeros(sz);

% Apply filter bank to all images in dataset

for i=1:numel(train_dir)
    disp('Applying filter banks');
    disp(i);
    tic
    im = imread(train_dir(i).name);
    im = im2double(im);
    c_FilteredTrain{i,1} = fbRun(fb,im);
    toc
end

% save('filtered_train.mat','c_FilteredTrain','-v7.3');

% Initialize random set of integers for subsampling of train set 
inds = randi(sz(1)*sz(2),10000,1);

% Preallocation for next loop
partialkmeans = zeros(10000,32);
c_Totkmeans = cell(750,1);
totkmeans = zeros(10000*750,32);

% This loop takes all filtered image responses for all images in train set,
% and selects 10000 pixels from each image for posterior kmeans clustering

for i=1:numel(train_dir)
    disp('Subsampling');
    disp(i);
    x = c_FilteredTrain{i,1};
    for j=1:numel(x);
        if j<17
            xx = x{j,1}(:);
        else
            xx = x{j-16,2}(:);
        end
        subsampled = xx(inds);
        partialkmeans(:,j) = subsampled;
    end
    c_Totkmeans{i,1} = partialkmeans;
end

% Bring all together
AllKmeans = cell2mat(c_Totkmeans);
[map, textons] = kmeans(AllKmeans,k,'Replicates',2,'MaxIter',500);

% Train map assignment
for i=1:numel(train_dir)
    disp(i);
    disp('Train map assignment');
    im2 = imread(train_dir(i).name);
    im2 = im2double(im2);
    tmap = assignTextons(fbRun(fb,im2),textons');
    c_MapTrain{i,1} = tmap;
    c_MapTrain{i,2} = histc(tmap(:),1:k)/numel(tmap);
    c_MapTrain{i,3} = strtok(train_dir(i).name,'_');
end

% Test map assignment
for i=1:numel(test_dir)
    disp(i);
    disp('Test map assignment');
    im2 = imread(test_dir(i).name);
    im2 = im2double(im2);
    tmap = assignTextons(fbRun(fb,im2),textons');
    c_MapTest{i,1} = tmap;
    c_MapTest{i,2} = histc(tmap(:),1:k)/numel(tmap);
    c_MapTest{i,3} = strtok(test_dir(i).name,'_');
end

save('test.mat','c_MapTest');
save('train.mat','c_MapTrain');

