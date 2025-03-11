function [final_AUC,final_ACC] = real_tc(ten,idx,info,traIdx,valIdx,tstIdx,transform)
[n1,n2,n3] = size(ten);
Omega = zeros(n1,n2,n3);
Omega(idx(traIdx)) = 1;

traData= zeros(n1,n2,n3);
traData(idx(traIdx)) = ten(idx(traIdx));
valData= zeros(n1,n2,n3);
valData(idx(valIdx)) = ten(idx(valIdx));
tstData= zeros(n1,n2,n3);
tstData(idx(tstIdx)) = ten(idx(tstIdx));
lambda_m = [0.01,0.1,1,10];
lambda_n = [0.01,0.1,1,10];

%% Construct side information
[U1,~,V1] = tsvd(info,transform,'full');
X1 = side_info(U1,transform,25,0);
Y1 = side_info(V1,transform,25,0);

val_data = valData(idx(valIdx));
tst_data = tstData(idx(tstIdx));
i = 0;

%% Validation
for a = 1:length(lambda_m)	% parameter selection
   for b = 1:length(lambda_n)
        i = i+1;
		[Xhat_tmp,~,~] = itcnf(traData,Omega,X1,Y1,transform,lambda_m(a),lambda_n(b));
        prediction = Xhat_tmp(idx(valIdx));
        AUC(i) = calAUC(prediction,val_data, 1e6);
        Accuracy(i) = calAccuracy(prediction,val_data);
        prediction = Xhat_tmp(idx(tstIdx));
        final_AUC = calAUC(prediction,tst_data, 1e6);
        final_ACC = calAccuracy(prediction,tst_data);
        lambda_m_(i) = lambda_m(a);
        lambda_n_(i) = lambda_n(b);
   end
end

%% Test
[~,idx_opt] = max(AUC); 
lambda_m_opt = lambda_m_(idx_opt);
lambda_n_opt = lambda_n_(idx_opt);
[Xhat_tmp,~] = w_tc_real(traData,Omega,X1,Y1,transform,lambda_m_opt,lambda_n_opt);
prediction = Xhat_tmp(idx(tstIdx));
final_AUC = calAUC(prediction,tst_data, 1e6);
final_ACC = calAccuracy(prediction,tst_data);
end