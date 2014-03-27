function alpha = stepSize(mu,gamma,eta)

%This function updates the step size based on the mu parameter from the
%hybrid descent method.  There are other choices of update functions, given
%below

phi = eta*(1-mu);
%phi = eta*(1-mu^2);
%phi = eta*(1-sqrt(mu));

if mu > 1
    alpha = gamma;
else
    alpha = (1+phi)*gamma;
end