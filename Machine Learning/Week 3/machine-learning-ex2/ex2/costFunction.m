function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h_theta = sigmoid(theta'*X');
J = (1/m)*(sum((-y'*log(h_theta'))-((1.-y)'*log(1-h_theta'))));
grad = (1/m)*((h_theta-y')*X);

%sum1 = [0 0 0]
%for i = 36:40
%  sum1 += ((-y(i))*log(sigmoid(X(i,:))))-((1-y(i))*log(1-sigmoid(i,:)));
%end

%J=(1/m)*sum1;
%% grad =

%for i = 1:2
%  sum1 += ((-y(i))*log(sigmoid(X(i,:))))-((1-y(i))*log(1-sigmoid(i,:)))
%end



% =============================================================

end
