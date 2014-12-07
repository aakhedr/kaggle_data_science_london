function [errors] = validation_curve(Xtrain, Xval, yTrain, yVal)
    sigma_vec = [3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9];
    C_vector = [1 2 3 4 5 6 7 8 9 10 11 12 13];
    m = length(sigma_vec);
    n =length(C_vector);

    Ein = zeros(m, 1); Eval = zeros(m, 1);
    
    for i = 1 : m
        fprintf('\trbf_sigma=%f\n', sigma_vec(i));
        fprintf('\tC\t\tEin\t\tEval\n');

        for j = 1 : n
            svm_struct = svmtrain(Xtrain, yTrain, 'kernel_function', 'rbf', ...
                'rbf_sigma', sigma_vec(i), 'boxconstraint', C_vector(j));
        
            y_train_est = svmclassify(svm_struct, Xtrain);
            Ein(i) = length(y_train_est(y_train_est~=yTrain)) / length(yTrain);
        
            y_val_est = svmclassify(svm_struct, Xval);
            Eval(i) = length(y_val_est(y_val_est~=yVal))/ length(yVal);
            
            fprintf('\t%f\t%f\t%f\n', C_vector(j), Ein(i), Eval(i));
        end
        
        fprintf('\n');
    end
    errors = [sigma_vec' C_vector' Ein Eval];
end