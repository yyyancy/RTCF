%% demo on TRPCA simulation (Figure 5)
clear
close all
addpath("tensor_toolbox_2.6")
n = 50;
dim = 10;
n1 = n;
n2 = n;
n3 = n;
r = 0.1*n1; % tubal rank
P = randn(n1,r,n3)/n1;
Q = randn(r,n2,n3)/n2;

transform1.L = @fft; transform1.l = n3; transform1.inverseL = @ifft;
transform2.L = @dct; transform2.l = 1; transform2.inverseL = @idct;
transform3.L = RandOrthMat(n3);transform3.inverseL = inv(transform3.L);transform3.l = 1; 


L1 = tprod(P,Q,transform1);
L2 = tprod(P,Q,transform2);
L3 = tprod(P,Q,transform3);

p_obs = 1;
Omega = omega(L1,p_obs);

lambda_m = [0.01,0.1,1,10];
lambda_n = [0.01,0.1,1,10];
ps = [0.3,0.35,0.4];
pf = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];

lambda_s1 = 1/sqrt(transform1.l.*max(n1,n2));
lambda_s2 = 1/sqrt(transform2.l.*max(n1,n2));
lambda_s3 = 1/sqrt(transform3.l.*max(n1,n2));

fid = fopen("tensor_robust_pca",'a');

itermax = 10;
for iteration =1:itermax
fprintf(fid,'-------------------------iteration:%d-------------------------\n',iteration);
for e = 1:length(ps)
    S = S_noisy(L1,ps(e));
    fprintf(fid,'======================================\n');
    fprintf(fid,'p_s:%.10f\n',ps(e));
    for c = 1:length(pf)
        tmp_loss = [];
        i = 0;
        fprintf(fid,'--------------------------\n');
        fprintf(fid,'p_f:%.10f\n',pf(c));
        p_f = pf(c);
        
        [U_t1,~,V_t1] = tsvd(L1,transform1,'full');
        X_t1 = side_info(U_t1,transform1,dim,p_f);
        Y_t1 = side_info(V_t1,transform1,dim,p_f);
        
        [U_t2,~,V_t2] = tsvd(L2,transform2,'full');
        X_t2 = side_info(U_t2,transform2,dim,p_f);
        Y_t2 = side_info(V_t2,transform2,dim,p_f);
        
        [U_t3,~,V_t3] = tsvd(L3,transform3,'full');
        X_t3 = side_info(U_t3,transform3,dim,p_f);
        Y_t3 = side_info(V_t3,transform3,dim,p_f);

        for a = 1:length(lambda_m)	% parameter selection
           for b = 1:length(lambda_n)
                i = i+1;
		        [~,~,loss_fft,~,~] = tpcpnf(L1,Omega,X_t1,Y_t1,transform1,lambda_m(a),lambda_n(b),lambda_s1,S);
                [~,~,loss_dct,~,~] = tpcpnf(L2,Omega,X_t2,Y_t2,transform2,lambda_m(a),lambda_n(b),lambda_s2,S);
                [~,~,loss_rom,~,~] = tpcpnf(L3,Omega,X_t3,Y_t3,transform3,lambda_m(a),lambda_n(b),lambda_s3,S);
                tmp_loss_fft(i) = loss_fft;
                tmp_loss_dct(i) = loss_dct;
                tmp_loss_rom(i) = loss_rom;
           end
        end
        loss_ps_fft(c) = min(tmp_loss_fft);
        loss_ps_dct(c) = min(tmp_loss_dct);
        loss_ps_rom(c) = min(tmp_loss_rom);
        fprintf(fid,'loss_fft:%.10f\tloss_dct:%.10f\tloss_rom:%.10f\n',loss_ps_fft(c),loss_ps_dct(c),loss_ps_rom(c));
      
        tmp_loss_fft = ones(a*b,1);
        tmp_loss_dct = ones(a*b,1);
        tmp_loss_rom = ones(a*b,1);
    end
       t_loss_fft(:,e,iteration) = loss_ps_fft(:);
       t_loss_dct(:,e,iteration) = loss_ps_dct(:);
       t_loss_rom(:,e,iteration) = loss_ps_rom(:);
        
end
end

if itermax >=2
    for i=1:length(pf)
        for j=1:length(p_obs)
            final_loss_fft(i,j)=mean(t_loss_fft(i,j,:));
            final_loss_dct(i,j)=mean(t_loss_dct(i,j,:));
            final_loss_rom(i,j)=mean(t_loss_rom(i,j,:));
        end
    end
else
        final_loss_fft = t_loss_fft;
        final_loss_dct = t_loss_dct;
        final_loss_rom = t_loss_rom;
end

fprintf(fid,"---------------------------\n")
fprintf(fid,'loss_fft:%f\n',final_loss_fft);
fprintf(fid,'loss_dct:%f\n',final_loss_dct);
fprintf(fid,'loss_rom:%f\n',final_loss_rom);

fclose(fid);