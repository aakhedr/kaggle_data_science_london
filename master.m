%% read data into MATLAB
[data, test, labels] = read_data();

%% segregate train and validation data
[Xtrain, Xval, yTrain, yVal] = segregate_data(data, labels);

%% generate validation curve
errors = validation_curve(Xtrain, Xval, yTrain, yVal);

%% generate learning curve
% [Ein, Eval] = learning_curve(Xtrain, Xval, yTrain, yVal);

%% fit the test set using svm_struct
svm_struct = svmtrain(data, labels, 'kernel_function', 'rbf', ...
    'rbf_sigma', 8, 'boxconstraint', 10);
yTest = svmclassify(svm_struct, test);

%% submit
Id = (1:9000)';
submission = [Id yTest];
csvwrite('submission5.csv', submission);

%%