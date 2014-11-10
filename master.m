%% read data into MATLAB
[data, test, labels] = read_data('train.csv', 'test.csv', ...
    'trainLabels.csv');

%% segregate train and validation data
[Xtrain, Xval, yTrain, yVal] = segregate_data(data, labels);

%% train svm with rbf kernel 
svm_struct = svmtrain(Xtrain, yTrain, 'kernel_function', 'rbf', ...
    'rbf_sigma', 4);

%% classify train and validation set
y_est_train = svmclassify(svm_struct, Xtrain);
y_est_val = svmclassify(svm_struct, Xval);

%% calculate prediction accuracy on train and validation set
Ein = length(y_est_train(y_est_train~=yTrain))/ length(yTrain);
Eout = length(y_est_val(y_est_val~=yVal))/ length(yVal);

fprintf('Ein: %f\n', Ein);
fprintf('Eout: %f\n', Eout);
