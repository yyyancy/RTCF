function [L,S,loss_L,loss_S,iter] = tpcpnf(L0,omega,X,Y,transform,lambda_m,lambda_n,lambda_s,S_noisy)

[n1,n2,n3] = size(L0);

R = (L0+S_noisy).*omega;
S_noisy = S_noisy.*omega;
m = nnz(omega);

d = [size(X,2),size(Y,2)]; 

M = zeros([d,n3]);
N = zeros(n1,n2,n3);
S = zeros(n1,n2,n3);
Z1 = zeros(n1,n2,n3);
Z2 = zeros(n1,n2,n3);
E = zeros(n1,n2,n3);
G = R;
DEBUG = 0;

loss2 = 1e+7;
loss1 = 1e+7;
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
    Ek = E;
    Sk = S;
    Lk = L;
    M = M_update;
    N = N_update;
    S = S_update;
    G = G_update;
    E = E_update;
    [Z1,dY1] = Z1_update;
    [Z2,dY2] = Z2_update;
    L = tprod(tprod(X,M,transform),tran(Y,transform),transform)+ N;
    chgL = max(abs(Lk(:)-L(:)));
    chgE = max(abs(Ek(:)-E(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([chgL chgE chgS max(abs(dY1(:))) max(abs(dY2(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            err1 = norm(dY1(:));
            err2 = norm(dY2(:));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                   ', err1=' num2str(err1) ', err2=' num2str(err2) ...
                   ', chg=' num2str(chg) ]); 
        end
    end
    
    if chg < tol
        break;
    end 
    
    loss_L = norm(L(:)-L0(:))/norm(L0(:));
    loss_S = norm(S(:)-S_noisy(:))/norm(S_noisy(:));

    mu = min(rho*mu,max_mu);
end


function new_M = M_update
    t_M = G - N - S + Z2./mu;
    [M_tmp,~] = prox_tnn(t_M,lambda_m./mu,transform);
    new_M = tprod(tprod(tran(X,transform),M_tmp,transform),Y,transform);
end


function new_N = N_update
    XMY = tprod(tprod(X,M,transform),tran(Y,transform),transform);
    t_N = G - XMY - S + Z2./mu;
    [N_tmp,~] = prox_tnn(t_N,lambda_n./mu,transform);
    new_N = N_tmp;
end

function new_S = S_update
    XMY = tprod(tprod(X,M,transform),tran(Y,transform),transform);
    t_S = G - XMY - N + Z2./mu;
    S_tmp = prox_l1(t_S,lambda_s./mu);
    new_S = S_tmp;
end

function new_G = G_update
    XMY = tprod(tprod(X,M,transform),tran(Y,transform),transform);
    new_G = ((R+E-Z1./mu) + (XMY+N+S-Z2./mu))./2;
end

function new_E = E_update
    E_tmp = G-R+Z1./mu;
    new_E = E_tmp - E_tmp.*omega;
end

function [new_Z2,dY2] = Z2_update
    XMY = tprod(tprod(X,M,transform),tran(Y,transform),transform);
    new_Z2 = Z2 + mu.*(G - XMY - N - S);
    dY2 = G - XMY - N - S;
end

function [new_Z1,dY1] = Z1_update
    new_Z1 = Z1 + mu.*(G-R-E);
    dY1 = G-R-E;
end

end