function train_net(varargin)

setup;

net = initializeTextonsCNN() ;
trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 5 ;
trainOpts.continue = true ;
trainOpts.useGpu = false ;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'data/textons-experiment' ;
trainOpts = vl_argparse(trainOpts, varargin);

% Take the average image out
imdb = load('textonsdbs.mat') ;
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;

% Convert to a GPU array if needed
if trainOpts.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

% Call training function in MatConvNet
tic
[net,info] = cnn_train(net, imdb, @getBatchWithJitter, trainOpts) ;
trainingTime = toc
% Move the CNN back to the CPU if it was trained on the GPU
if trainOpts.useGpu
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
% net.imageMean = imageMean ;
save('data/textons-experiment/textonscnn.mat', '-struct', 'net') ;


% --------------------------------------------------------------------
function [im, labels] = getBatchWithJitter(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,batch) ;
labels = imdb.images.label(1,batch) ;
imageMean = mean(im(:));
n = numel(batch) ;

theta = randi([-359 360],1);

imr = imrotate(im,theta,'crop');

imr(imr==0) = imageMean;

im = 256 * reshape(im, 128, 128, 1, []) ;




