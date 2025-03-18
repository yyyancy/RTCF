%% demo on Tensor Completion simulation (Figure 4)
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

lambda_m = [0.01,0.1,1,10];
lambda_n = [0.01,0.1,1,10];

p_f = [0.1,0.3,0.5];
p_obs = linspace(0,0.6,10);

fid = fopen("tensor_completion",'a');

itermax = 10;
for iter = 1:itermax
    fprintf(fid,'-------------------------iteration:%d-------------------------\n',iteration);
    for e = 1:length(p_f)
     pf = p_f(e);
     fprintf(fid,'======================================\n');
     fprintf('p_f:%.10f\n',p_f(e));
     fprintf(fid,'p_f:%.10f\n',p_f(e));

     [U_t1,~,V_t1] = tsvd(L1,transform1,'full');
     X_t1 = side_info(U_t1,transform1,dim,pf);
     Y_t1 = side_info(V_t1,transform1,dim,pf);
    
     [U_t2,~,V_t2] = tsvd(L2,transform2,'full');
     X_t2 = side_info(U_t2,transform2,dim,pf);
     Y_t2 = side_info(V_t2,transform2,dim,pf);
    
     [U_t3,~,V_t3] = tsvd(L3,transform3,'full');
     X_t3 = side_info(U_t3,transform3,dim,pf);
     Y_t3 = side_info(V_t3,transform3,dim,pf);
 
     
     for c = 1:length(p_obs)
        i = 0;
        fprintf(fid,'--------------------------\n');
        fprintf(fid,'p_obs:%.10f\n',p_obs(c));
        lambda_st1 = 1/sqrt(transform1.l.*max(n1,n2).*p_obs(c));
        lambda_st2 = 1/sqrt(transform2.l.*max(n1,n2).*p_obs(c));
        lambda_st3 = 1/sqrt(transform3.l.*max(n1,n2).*p_obs(c));
        Omega_t = omega(L1,p_obs(c));
      
        for a = 1:length(lambda_m)	% parameter selection
           for b = 1:length(lambda_n)
                i = i+1;
	            [~,loss_fft,~] = itcnf(L1,Omega_t,X_t1,Y_t1,transform1,lambda_m(a),lambda_n(b));
                [~,loss_dct,~] = itcnf(L2,Omega_t,X_t2,Y_t2,transform2,lambda_m(a),lambda_n(b));
                [~,loss_rom,~] = itcnf(L3,Omega_t,X_t3,Y_t3,transform3,lambda_m(a),lambda_n(b));
                tmp_loss_fft(i) = loss_fft;
                tmp_loss_rom(i) = loss_rom;
                tmp_loss_dct(i) = loss_dct;
           end
        end
        loss_ps_fft(c) = min(tmp_loss_fft);
        loss_ps_dct(c) = min(tmp_loss_dct);
        loss_ps_rom(c) = min(tmp_loss_rom);
        fprintf(fid,'loss_fft:%.10f\tloss_dct:%.10f\tloss_rom:%.10f\n',loss_ps_fft(c),loss_ps_dct(c),loss_ps_rom(c));
        loss_fft = ones(a*b,1);
        loss_dct = ones(a*b,1);
        loss_rom = ones(a*b,1);
        end
     t_loss_fft(:,e,iter) = loss_ps_fft(:);
     t_loss_dct(:,e,iter) = loss_ps_dct(:);
     t_loss_rom(:,e,iter) = loss_ps_rom(:);
     end
end

if itermax >=2
    for i=1:length(p_obs)
        for j=1:length(ps)
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