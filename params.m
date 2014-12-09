function [C, sigma, errors] = params(X, y, Xval, yval)

	models = [3 10 13 20 23 30];
	errors = [];
	for i = 1:length(models)
		for j = 1:length(models)
			model = svmTrain(X, y, models(i), @(x1, x2) gaussianKernel(x1, x2, models(j)), 1e-3, 20);
			
			predictions = svmPredict(model, Xval);
			errors = [errors; mean(double(predictions ~= yval)) models(i) models(j)];
		end
	end
	[~, index] = min(errors(:, 1));
	C = errors(index, 2); sigma = errors(index, 3);
end
