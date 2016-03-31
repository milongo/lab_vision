load test.mat
load train.mat

c_HistsTrain = cell(750,1);
c_HistsTest = cell(250,1);
c_LabelsTrain = cell(750,1);
c_KnnPredict = cell(250,1);
c_groundTruth = cell(250,1);

% KNN Classification
for i=1:numel(test_dir)
    dists = zeros(numel(test_dir),1);
    for j=1:numel(train_dir)
        dists(j) = chiSqrDist...
            (c_MapTrain{j,2},c_MapTest{i,2});
    end
    [Y,I] = min(dists);
    c_MapTest{i,4} = c_MapTrain{I,3};
    c_HistsTest{i,1} = c_MapTest{i,2}'; % For later Tree classification
    c_KnnPredict{i,1} = c_MapTest{i,4}; % For confusion matrix
    c_groundTruth{i,1} = c_MapTest{i,3}; % For confusion matrix
end

for i=1:numel(train_dir)
    c_HistsTrain{i,1} = c_MapTrain{i,2}';
    c_LabelsTrain{i,1} = c_MapTrain{i,3};
end

% Tree classification
features_TreeTraining = cell2mat(c_HistsTrain);
classify_TreeTest = cell2mat(c_HistsTest);
nTrees = 100;
B = TreeBagger(nTrees,features_TreeTraining,c_LabelsTrain,'Method','classification');
c_TreePredict = predict(B,classify_TreeTest);

m_groundTruth = cell2mat(c_groundTruth);
m_KnnPredict = cell2mat(c_KnnPredict);
m_TreePredict = cell2mat(c_TreePredict);

m_groundTruth = str2num(m_groundTruth(:,2:3));
m_KnnPredict = str2num(m_KnnPredict(:,2:3));
m_TreePredict = str2num(m_TreePredict(:,2:3));

C1 = confusionmat(c_groundTruth,c_KnnPredict);
C2 = confusionmat(c_groundTruth,c_TreePredict);


save('KnnConfusion.mat','C1');
save('TreeConfusion.mat','C2');

