function [data, test, y] = read_data(trainFile, testFile, labelsFile)
    data = csvread(trainFile);
    test = csvread(testFile);
    y = csvread(labelsFile);
end