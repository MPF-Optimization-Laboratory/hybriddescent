function [x,cost] = IncrDescent(A,b,alpha,iter)
%SOLVES LEAST SQUARES PROBLEM (1/2||Ax-b||^2) USING INCREMENTAL GRADIENT
%DESCENT.  EACH DATA BLOCK IS A ROW OF A.

[m,n] = size(A);

x = zeros(n,1);
cost = zeros(iter,1);

for i = 1:iter
    psi = x;
    randvect = randperm(m)';
    Arand = A(randvect,:);
    brand = b(randvect);
    
    for j = 1:m
        psi = psi - alpha*Arand(j,:)'*(Arand(j,:)*psi - brand(j));
    end
    
    if mod(i,100) == 0
        alpha = alpha/2;
    end
    
    x = psi;
    cost(i) = (norm(A*x-b)^2)/2;
end