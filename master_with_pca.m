%% read data into MATLAB
[data, test, labels] = read_data();

%% segregate train and validation data
[Xtrain, Xval, yTrain, yVal] = segregate_data(data, labels);

%% normalize features
[Xtrain_norm, muTrain, sigmaTrain] = featureNormalize(Xtrain);
[Xval_norm, muVal, sigmaVal] = featureNormalize(Xval);

%% run pca
[Utrain, Strain, ~] = pca(Xtrain_norm);
[Uval, Sval, ~] = pca(Xval_norm);
K = 12;
Ztrain = projectData(Xtrain_norm, Utrain, K);
Zval = projectData(Xval_norm, Uval, K);

%% generate validation curve
errors = validation_curve(Ztrain, Zval, yTrain, yVal);

%% generate learning curve
% [Ein, Eval] = learning_curve(Ztrain, Zval, yTrain, yVal);

%% fit the test set using svm_struct
% svm_struct = svmtrain(data, labels, 'kernel_function', 'rbf', ...
%     'rbf_sigma', 4);
% yTest = svmclassify(svm_struct, test);

% %% submit
% Id = (1:9000)';
% submission = [Id yTest];
% csvwrite('submission7.csv', submission);

%%