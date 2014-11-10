function [data, test, labels] = read_data()
    data = csvread('train.csv');
    test = csvread('test.csv');
    labels = csvread('trainLabels.csv');
end