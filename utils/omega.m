function [Omega]=omega(M,p)
[n1,n2,n3] =size(M);
m = floor(p*n1*n2*n3);
temp = rand(n1*n2*n3,1);
[B,I] = sort(temp);
I = I(1:m);
Omega = zeros(n1,n2,n3);
Omega(I) = 1;
end