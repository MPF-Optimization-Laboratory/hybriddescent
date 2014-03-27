function [x,cost] = HybridDescent(A,b,gamma,iter,batchsize,beta,delta,eps,nhat,eta)
%SOLVES LEAST SQUARES PROBLEM (1/2||Ax-b||^2) USING HYBRID GRADIENT
%DESCENT.  EACH DATA BLOCK IS A ROW OF A.

%gamma is the learning rate that is suitable for gradient descent

%batchsize is how many of the data blocks you want to handle at once.  This
%reduces the overall number of data blocks.

%TUNING PARAMETERS: beta//delta//eps//nhat//eta//stepSizeFCN//

[m,n] = size(A);

x = zeros(n,1);
cost = zeros(iter,1);

mu = 0;
lastUpdate = 0;
numblocks = floor(m/batchsize);
lastblock = numblocks*batchsize;

%Initialize learning rate
alpha = stepSize(mu,gamma,eta);

for i = 1:iter
    h = zeros(n,1);
    g = zeros(n,1);
    psi = x;
    
    randvect = randperm(m)';
    Arand = A(randvect,:);
    brand = b(randvect);
    
    k = 1;
    for j = 1:batchsize:lastblock
        %g is the sum needed for each h_i.  This rolling sum formulation
        %requires no additional storage.  The indexing is to jump between
        %sets of data blocks that fit the batchsize. k keeps track of what
        %the actual sum counter would be.
        g = g + xi(mu,k,numblocks)*Arand(j:j+batchsize-1,:)'*...
            (Arand(j:j+batchsize-1,:)*psi - brand(j:j+batchsize-1));
        h = mu*h + g;
        psi = x - alpha*h;
        k = k+1;
    end
    
    x_new = psi;
    
    %Decide whether to update mu
    if norm(x_new-x) <= eps
        mu = beta*mu + delta;
        alpha = stepSize(mu,gamma,eta);
        lastUpdate = 0;
    else
        if lastUpdate > nhat
            mu = beta*mu+delta;
            alpha = stepSize(mu,gamma,eta);
            lastUpdate = 0;
        else
            lastUpdate = lastUpdate + 1;
        end
    end
    
    x = x_new;
    cost(i) = (norm(A*x-b)^2)/2;
end