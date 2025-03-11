function [res] = calAccuracy(X, data_test)
    idx = find(data_test);
    res = sum(sign(X(idx))==sign(data_test(idx)))/length(data_test);
end