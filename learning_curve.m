function [Ein, Eval] = learning_curve(Xtrain, Xval, yTrain, yVal)
    [m, ~] = size(Xtrain);
    Ein = zeros(m, 1);
    Eval = zeros(m, 1);
    
    % loop thru all examples
    for i = 4 : m    % y must contain exactly 2 groups for SMO to run
        X = Xtrain(1:i, :); y = yTrain(1:i, :);
        
        % train svm classifier with rbf kernel and sigma=5.5 boxconstraint=9
        svm_struct = svmtrain(X, y, 'kernel_function', 'rbf', ...
            'rbf_sigma', 5.5, 'boxconstraint', 9);
        
        % calculate Ein for each set of training examples
        y_train_est = svmclassify(svm_struct, X);
        Ein(i) = length(y_train_est(y_train_est~=y))/ length(y);
        
        % calculate Eout by each svm_struct
        y_val_est = svmclassify(svm_struct, Xval);
        Eval(i) = length(y_val_est(y_val_est~=yVal))/ length(yVal);
    end
    plot(1:m, Ein, 'b-');   hold on;
    plot(1:m, Eval, 'r-'); 
    
    legend('Ein', 'Eval');
    title('Learning Curve with rbf sigma=5.5 boxconstraint=9');
    xlabel('Number of examples');
    ylabel('Error');        hold off;
end