function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Calculate second layer
a2 = sigmoid(Theta1 * X');

% Add ones to the a2 data matrix
a2 = [ones(1,m); a2];

% Calculate third layer
a3 = sigmoid(Theta2 * a2);

% h (5000 x 10): pour chacun des 5000 exemples,
%   on a la probabilité que ce soit un 1,2,...,9,0
h = a3';

% P (5000 x 1) est la probabilité d'accuracy maximum pour chaque exemple
% p (5000 x 1) est la l'indice de la probabilité d'accuracy maximum pour
%   chaque exemple
[P,p] = max(h,[], 2);

% Remplacer les 10 dans le vecteur p par des 0 pour donner les prédictions
%    de l'algorithme
p(p==10)=0;

% =========================================================================


end
