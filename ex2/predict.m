function p = predict(theta, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

prob = sigmoid(X * theta);
p = ones(m, 1);					% initialize all predictions to ones first
p( find(prob<0.5) ) = 0;	% set zeros to all probabilities < threshold 0.5


end
