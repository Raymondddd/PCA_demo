clear;
load data/digit;

%transform all images into matrix
%transform trainging set
d = size(train{1},1) .^2;
trN = size(train,2);
imgs_tr = zeros(d, trN);
for i=1:trN
    %image matrix to vector, e.g. one column
    imgs_tr(:,i) = reshape( train{i}, d, 1 );
end
mn_tr = mean(imgs_tr,2);
%same transform for test set
teN = size(test,2);
imgs_te = zeros(d, teN);
for i=1:teN
    imgs_te(:,i) = reshape( test{i}, d, 1);
end
mn_te = mean(imgs_te,2);

%ICA---begin

