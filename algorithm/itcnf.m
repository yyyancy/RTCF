function [L,loss,iter] = itcnf(L0,omega,X,Y,transform,lambda_m,lambda_n)

[n1,n2,n3] = size(L0);

R = L0.*omega;
m = nnz(omega);

d = [size(X,2),size(Y,2)]; 

M = zeros([d,n3]);
N = zeros(n1,n2,n3);
Z1 = zeros(n1,n2,n3);
Z2 = zeros(n1,n2,n3);
E = zeros(n1,n2,n3);
G = R;

DEBUG = 0;
tol = 1e-8;
iter = 0;
iter_max = 500;
L = zeros(n1,n2,n3);
rho = 1.1;
mu = 1e-4;
max_mu = 1e+10;

hasConverged = false;

while iter < iter_max
    iter = iter + 1;
    Lk = L;
    Ek = E;
    M = M_update;
    N = N_update;
    G = G_update;
    E = E_update;
    [Z1,dY1] = Z1_update;
    [Z2,dY2] = Z2_update;
    L = tprod(tprod(X,M,transform),tran(Y,transform),transform)+ N;
    chgL = max(abs(Lk(:)-L(:)));
    chgE = max(abs(Ek(:)-E(:)));
    chg = max([chgL chgE max(abs(dY1(:))) max(abs(dY2(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            err1 = norm(dY1(:));
            err2 = norm(dY2(:));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                   ', err1=' num2str(err1) ', err2=' num2str(err2) ...
                   ', chg=' num2str(chg) ]); 
        end
    end
    loss = norm(L(:)-L0(:))/norm(L0(:));
    if chg < tol
        break;
    end 
    mu = min(rho*mu,max_mu);
end


function new_M = M_update
    t_M = G - N + Z2./mu;
    [M_tmp,~] = prox_tnn(t_M,lambda_m./mu,transform);
    new_M = tprod(tprod(tran(X,transform),M_tmp,transform),Y,transform);
end


function new_N = N_update
    XMY = tprod(tprod(X,M,transform),tran(Y,transform),transform);
    t_N = G - XMY + Z2./mu;
    [N_tmp,~] = prox_tnn(t_N,lambda_n./mu,transform);
    new_N = N_tmp;
end


function new_G = G_update
    XMY = tprod(tprod(X,M,transform),tran(Y,transform),transform);
    new_G = ((R+E-Z1./mu) + (XMY+N-Z2./mu))./2;
end

function new_E = E_update
    E_tmp = G-R+Z1./mu;
    new_E = E_tmp - E_tmp.*omega;
end

function [new_Z2,dY2] = Z2_update
    XMY = tprod(tprod(X,M,transform),tran(Y,transform),transform);
    new_Z2 = Z2 + mu.*(G - XMY - N );
    dY2 = G - XMY - N ;
end

function [new_Z1,dY1] = Z1_update
    new_Z1 = Z1 + mu.*(G-R-E);
    dY1 = G-R-E;
end

end