%% demo on Link Prediction (IMDB)
clc;
clear all;
load("IMDB_sub.mat");
load("IMDB_info.mat")

ten = net;
[n1,n2,n3] = size(ten);

ten(ten==-1) = 0;
info(info==-1) = 0;
pos_idx = find(ten);
neg_idx = find(ten==0);
rand_idx = randperm(length(neg_idx));
negIdx = rand_idx(1:floor(length(pos_idx)));
ten(neg_idx(negIdx)) = -1;
for i = 1:length(negIdx)
    if info(neg_idx(negIdx(i))) == 0
        info(neg_idx(negIdx(i))) = -1;
    end
end
NZ = nnz(ten);
OBS = nnz(ten)/numel(ten);
transform1.L = @fft; transform1.l = n3; transform1.inverseL = @ifft;
transform2.L = @dct; transform2.l = 1; transform2.inverseL = @idct;
transform3.L = RandOrthMat(n3);transform3.inverseL = inv(transform3.L);transform3.l = 1; 

p_tra = 0.8;
p_val = 0.1;
itermax = 10;
idx = find(ten);
[~,~,val] = find(ten);
fid = fopen("link_prediction",'a');
fprintf(fid,"---------------------------Begin-------------------------------\n")
for i = 1:itermax
    fprintf(fid,"--------------- Iteration %d------------------\n",i);
    rand_idx = randperm(length(idx));
    traIdx = rand_idx(1:floor(length(val)*p_tra));
    valIdx = rand_idx(ceil(length(val)*p_tra):floor(length(val)*(p_tra+p_val)));
    tstIdx = rand_idx(ceil(length(val)*(p_tra+p_val)):end);


    fprintf(fid,"--------------start test fft---------------\n");
    [AUC11(i),accuracy11(i)] = real_tc(ten,idx,info,traIdx,valIdx,tstIdx,transform1);
    fprintf(fid,"AUC11:%.5f,Accuracy11:%.5f\n",AUC11(i),accuracy11(i));

    fprintf(fid,"--------------start test dct---------------\n");
    [AUC12(i),accuracy12(i)] = real_tc(ten,idx,info,traIdx,valIdx,tstIdx,transform2);
    fprintf(fid,"AUC12:%.5f,Accuracy12:%.5f\n",AUC12(i),accuracy12(i));
    
    fprintf(fid,"--------------start test rom---------------\n");
    [AUC13(i),accuracy13(i)] = real_tc(ten,idx,info,traIdx,valIdx,tstIdx,transform3);
    fprintf(fid,"AUC13:%.5f,Accuracy13:%.5f\n",AUC13(i),accuracy13(i));
end

fprintf(fid,"==========================\n");
final_AUC11 = mean(AUC11);
final_AUC12 = mean(AUC12);
final_AUC13 = mean(AUC13);

final_accuracy11 = mean(accuracy11);
final_accuracy12 = mean(accuracy12);
final_accuracy13 = mean(accuracy13);


fprintf(fid,"fft AUC:%.5f, fft accuracy:%.5f\n",final_AUC11,final_accuracy11)
fprintf(fid,"dct AUC:%.5f,dct accuracy:%.5f\n",final_AUC12,final_accuracy12)
fprintf(fid,"rom AUC:%.5f,rom accuracy:%.5f\n",final_AUC13,final_accuracy13)
fclose(fid);


