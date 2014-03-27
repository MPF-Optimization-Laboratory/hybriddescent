%CPSC 546 PROJECT, BASIC LEAST SQUARES  SOLVE ||Ax-B||^2 WITH 2 NORM
clear

A = csvread('titanic.csv',1,1,[1,1,891,5]);
n = length(A);
b = csvread('titanic.csv',1,0,[1,0,n,0]);
exact = A\b;

iter = 500;

alphaGD = 1.99/norm(A'*A);
alphaID = .0005;

%TUNING PARAMETERS FOR HYBRID METHOD
beta = 1.1;
delta = .01;
eps = 1e-5;
nhat = 100;
eta = 100;
batcheta= 17;

nobatch = 1;
batch = 50;

timeSpent = zeros(5,4);
err = zeros(5,4);

for i = 1:5
    tic;
    x = GradDescent(A,b,alphaGD,iter);
    timeSpent(i,1) = toc;
    tic;
    y = IncrDescent(A,b,alphaID,iter);
    timeSpent(i,2) = toc;
    tic;
    z = HybridDescent(A,b,alphaGD,iter,nobatch,beta,delta,eps,nhat,eta);
    timeSpent(i,3) = toc;
    tic;
    z1 = HybridDescent(A,b,alphaGD,iter,batch,beta,delta,eps,nhat,batcheta);
    timeSpent(i,4) = toc;
    
    err(i,1) = norm(exact-x,Inf);
    err(i,2) = norm(exact-y,Inf);
    err(i,3) = norm(exact-z,Inf);
    err(i,4) = norm(exact-z1,Inf); 
end
timeSpent = sum(timeSpent)/5;
err = sum(err)/5;