function [Ein, Eout] = learning_curve(Xtrain, Xval, yTrain, yVal)
    [m, ~] = size(Xtrain);
    Ein = zeros(m, 1);
    Eout = zeros(m, 1);
    
    % loop thru all examples
    for i = 4 : m    % y must contain exactly 2 groups for SMO to run
        X = Xtrain(1:i, :); y = yTrain(1:i, :);
        
        % train svm classifier with rbf kernel and sigma=4
        svm_struct = svmtrain(X, y, 'kernel_function', 'rbf', ...
            'rbf_sigma', 4);
        
        % calculate Ein for each set of examples
        y_train_est = svmclassify(svm_struct, X);
        Ein(i) = length(y_train_est(y_train_est~=y))/ length(y);
        
        % calculate Eout for each set of examples in the validation set
        y_val_est = svmclassify(svm_struct, Xval);
        Eout(i) = length(y_val_est(y_val_est~=yVal))/ length(yVal);
    end
    plot(1:m, Ein, 'b-');   hold on;
    plot(1:m, Eout, 'r-'); 
    
    legend('Ein', 'Eout');
    xlabel('Number of examples');
    ylabel('Error');        hold off;
end