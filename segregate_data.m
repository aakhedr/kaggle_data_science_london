function [Xtrain, Xval, yTrain, yVal] = segregate_data(data, labels)
    Xtrain = data(1:800, :);    Xval = data(801:end, :);
    yTrain = labels(1:800, :);       yVal = labels(801:end, :);
end