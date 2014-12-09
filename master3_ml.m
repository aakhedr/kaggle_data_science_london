%% read data into MATLAB
[data, test, labels] = read_data();

%% segregate train and validation data
[Xtrain, Xval, yTrain, yVal] = segregate_data(data, labels);

%% generate validation curve
[C, sigma, errors] = params(Xtrain, yTrain, Xval, yVal);


%% fit the test set using C and sigma obtained
% model= svmTrain(data, labels, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
% predictions = svmPredict(model, test);

% %% submit
% Id = (1:9000)';
% submission = [Id predictions];
% csvwrite('submission12.csv', submission);

