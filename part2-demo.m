clear;
load digit;

%tranform all imgs to matrix: (28*28)dimentsion with 300 examples
d = size(train{1},1) * size(train{1},2);
n = size(train,2);
imgs_mat = zeros(d,n);
for i=1:n
    %load one image and transform to d*1 vector
    imgs_mat(:,i) = reshape( train{i}, d, 1); 
end

%transform test set
tn = size(test,2);
imgs_test = zeros(d,tn);
for i=1:tn
    %load image
    imgs_test(:,i) = reshape( test{i}, d, 1); 
end

%covariance PCA
[pc1 v1] = pca1(imgs_mat);
%output v is d eigenvalues

%SVD PCA
[pc2 v2] = pca2(imgs_test);
%output v is the minimize value between d and n

%90-95% of variance should chosen
sum_var = sum(v2);
d_comp = 0;
for i=1:size(v2,1)
    sum_M = sum( v2(1:i) );
    PoV = floor(sum_M/sum_var * 100);
    if(PoV > 90)
        d_comp = i;
        break;
    end
end

%projection on d_comp new space
mn = mean(imgs_mat,2);

z_mat1 = zeros( d_comp, n );
z_mat2 = zeros( d_comp, n );

%reconstruction
recon_mat1 = zeros(d,n);
recon_mat2 = zeros(d,n);

for i=1:n
    z_mat1(:,i) = pc1(:,1:d_comp)' * ( imgs_mat(:,i) - mn );
    recon_mat1(:,i) = mn + pc1(:,1:d_comp) * z_mat1(:,i);
    z_mat2(:,i) = pc2(:,1:d_comp)' * ( imgs_mat(:,i) - mn );
    recon_mat2(:,i) = mn + pc2(:,1:d_comp) * z_mat2(:,i);
end

mn_test = mean(imgs_test, 2);
z_test1 = zeros(d_comp, tn);
z_test2 = zeros(d_comp, tn);
recon_test1 = zeros(d,tn);
recon_test2 = zeros(d,tn);
for i=1:tn
    %z_test1(:,i) = pc1(:,1:d_comp)' * ( imgs_test(:,i) - mn );
    %recon_test1(:,i) = mn + pc1(:,1:d_comp) * z_test1(:,i);
    z_test2(:,i) = pc2(:,1:d_comp)' * ( imgs_test(:,i) - mn );
    recon_test2(:,i) = mn + pc2(:,1:d_comp) * z_test2(:,i);
end
e1 = sum( v1( (d_comp+1):size(v1,1), :) );
e2 = sum( v2( (d_comp+1):size(v2,1), :) );

Err = 0;

for i=1:size(recon_test2, 2)
    Err = Err + sum( ( imgs_test(:,i) - recon_test2(:,i) ) .^ 2 );
end

Err = Err/size(recon_test2,2)



%display test set with its reconstructed image
figure;
suptitle('Original VS reconstructed Image');
subplot(1,2,1);
display_digit( reshape( imgs_test(:,3), sqrt(d), sqrt(d) ) );
subplot(1,2,2);
display_digit( reshape( recon_test2(:,3), sqrt(d), sqrt(d) ) );
