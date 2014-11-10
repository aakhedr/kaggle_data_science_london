function [errors] = validation_curve(Xtrain, Xval, yTrain, yVal)
    sigma_vec = [.1 .3 .5 .7 .9 1 2 2.5 3 3.5 4 5 6 7 8 9 10 11 12];
    m = length(sigma_vec);
    Ein = zeros(m, 1); Eval = zeros(m, 1);
    
    fprintf('\tsigma\t\tEin\t\tEval\n');
    for i = 1 : m
        svm_struct = svmtrain(Xtrain, yTrain, 'kernel_function', 'rbf', ...
            'rbf_sigma', sigma_vec(i));
        
        y_train_est = svmclassify(svm_struct, Xtrain);
        Ein(i) = length(y_train_est(y_train_est~=yTrain)) / length(yTrain);
        
        y_val_est = svmclassify(svm_struct, Xval);
        Eval(i) = length(y_val_est(y_val_est~=yVal))/ length(yVal);
        
        fprintf('\t%f\t%f\t%f\n', sigma_vec(i), Ein(i), Eval(i));
    end
    errors = [sigma_vec' Ein Eval];
end