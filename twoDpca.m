clear;
load digit;

%training set size
tr_num = size(train,2);
%image size
[irow, icol] = size(train{1});

%get all 2-d images
imgs_tr = zeros(irow, icol, tr_num);
for i=1:tr_num
    imgs_tr(:, :, i) = train{i};
end
imgs_mn = mean(imgs_tr, 3);

%calculate the total scatter, the trace of 
Gt = zeros(icol,icol);
for i=1:tr_num
    temp = imgs_tr(:,:, i) - imgs_mn;
    Gt = Gt + ( temp' * temp );
end
Gt = Gt / tr_num;

%get eigenvectors and eigenvalues
[PC, V] = eig(Gt);
% extract diagonal of matrix as vector
V = diag(V);
% sort the variances in decreasing order
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
% find appropriate number of pricipal components.
for i=1:size(V,1)
    PoV = floor( sum( V(1:i) ) /  sum(V) *100 );
    if( PoV >= 90)
        M = i;
        break;
    end
end

%project test set on M-PC space
te_num = size(test,2);
imgs_te = zeros(irow, icol, te_num );
z_te = zeros(icol, M, te_num);
recon_te = zeros(irow, icol, te_num );
recon_err = 0;
for i=1:te_num
    imgs_te(:,:,i) = test{i};
    %project each image into M-PC space
    z_te(:,:,i) = imgs_te(:,:,i) * PC(:,1:M);
    %reconstruct feature image into original space
    recon_te(:,:,i) =  z_te(:,:,i) * PC(:,1:M)';
    recon_err = recon_err + sum( sum( ( imgs_te(:,:,i) - recon_te(:,:,i) ).^2 ) );
end
%average reconstruction error
recon_err = recon_err / te_num