function [x,cost] = GradDescent(A,b,alpha,iter)
%SOLVES LEAST SQUARES PROBLEM (1/2||Ax-b||^2) USING GRADIENT DESCENT.

[~,n] = size(A);
x = zeros(n,1);
cost = zeros(iter,1);

for i = 1:iter
    x = x - alpha*A'*(A*x-b);
    cost(i,1) = (norm(A*x-b)^2)/2;
end