function res = test_net(net,test_data)

for i=1:size(test_data,3)
    tmp_im = test_data(:,:,i);
    tmp_im = im2single(tmp_im);
    ress = vl_simplenn(net,tmp_im);
    scores = squeeze(gather(ress(end).x)) ;
    [bestScore, best] = max(scores) ;
    resss(i) = best;
end

res = resss;