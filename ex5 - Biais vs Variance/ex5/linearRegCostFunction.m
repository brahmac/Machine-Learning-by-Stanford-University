function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

    h = X * theta;
    thetaWOBias = theta(2:end,:);
    J = 1/(2*m) * sum( (h-y).^2 ) + lambda/(2*m) * sum(thetaWOBias.^2);

    X_with_j_equal_0 = X(:,1);
    gradThetaZero = (1/m) * X_with_j_equal_0' * (h - y);
    X_with_j_greaterThan_0 = X(:,2:end);
    gradOtherThetas = (1/m) * ( X_with_j_greaterThan_0' * (h - y) ) + (lambda/m*thetaWOBias);
    grad= [gradThetaZero;gradOtherThetas];

% =========================================================================

end
