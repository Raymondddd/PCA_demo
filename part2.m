clear;
load digit;

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

%apply SVD-PCA on training set to find the principal components
[PC V] = pca2(imgs_tr);

%find appropriate new dimension M, 90% of variance
for i=1:size(V,1)
    PoV = floor( sum( V(1:i) ) /  sum(V) *100 );
    if( PoV >= 90)
        M = i;
        break;
    end
end

%project test set on M-dimensions
%Z_te = PC(:,1:M)' * ( imgs_te - repmat( mn_te, 1, teN) );
Z_te = PC(:,1:M)' * ( imgs_te - repmat( mn_tr, 1, teN) );

%reconstruct Z_te to original dimensions
%recon_te = PC(:,1:M) * Z_te + repmat( mn_te, 1, teN);
recon_te = PC(:,1:M) * Z_te + repmat( mn_tr, 1, teN);


%generate the reconstruction error by Euclidean Distance
recon_err = mean( sum( (imgs_te - recon_te).^2 ) )

%display original and reconstructed test set 
figure;

for j=1:teN
    subplot(2,10,j);
    display_digit( reshape( imgs_te(:,j), sqrt(d), sqrt(d) ), 'actual' );
end
for j=1:teN
    subplot(2,10,j+10);
    display_digit( reshape( recon_te(:,j), sqrt(d), sqrt(d) ), 'actual' );
end
suptitle('Original VS Reconstructed test images');