function [U, S, V] = pca(X)	
	[m, ~] = size(X);
	Sigma = 1/m * X' * X;
	[U, S, V] = svd(Sigma);
end
