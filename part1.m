function part1(pca)
    load data/iris;
    
    %for SVD PCA
    if(pca == 2)
        [pc, v] = pca2(X);
    elseif(pca == 1)    %for covariance PCA, default
        [pc, v] = pca1(X);
    end
    mn = mean(X,2);

    figure;

    %projection on pc1-pc2
    Z = pc(:,[1 2])' * ( X - repmat(mn,1,size(X,2)) );
    subplot(1,3,1);
    plot( Z(1,:), Z(2,:), 'g*' );
    xlabel('PC1');
    ylabel('PC2');
    title('PC1-PC2');

    %projection on pc1-pc3
    Z = pc(:,[1 3])' * ( X - repmat(mn,1,size(X,2)) );
    subplot(1,3,2);
    plot( Z(1,:), Z(2,:), 'r*' );
    xlabel('PC1');
    ylabel('PC3');
    title('PC1-PC3');

    %projection on pc2-pc3
    Z = pc(:,[2 3])' * ( X - repmat(mn,1,size(X,2)) );
    subplot(1,3,3);
    plot( Z(1,:), Z(2,:), 'k*' );
    xlabel('PC2');
    ylabel('PC3');
    title('PC2-PC3');

    if(pca == 2)
        suptitle('SVD-PCA');
    elseif(pca==1)
        suptitle('Covariance-PCA');
    end
    
end