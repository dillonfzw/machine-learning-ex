function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
%%%%%%%%%%%%%%%%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%


% a1/X is "m x input_layer_size"
a1 = X;

% Theta1 is "hiden_lay_size x (input_layer_size + 1)"
% z2/a2 is "m x hidden_layer_size"
z2 = [ones(size(a1, 1), 1) a1] * Theta1';
a2 = sigmoid(z2);

% Theta2 is "num_labels x (hiden_layer_size + 1)"
% z3/a3 is "m x num_labels"
z3 = [ones(size(a2, 1), 1) a2] * Theta2';
a3 = sigmoid(z3);

% calculate cost J

% input y is "m x 1" which need to be converted to y2
% as "m x num_labels" matrix
% NOTE:
% - method 1 is the straight forward implementation of the formular in lecture 9
%   which needs a "for" loop for the calculation.
% - method 2 is the more complete vectorization implementation.
method = 2;
fprintf('Use cost method %d\n', method);

if method == 2
    y2 = zeros(m, num_labels);
end
J = 0;
for i=1:num_labels
    yi = (y == i);  % m x 1

    if method == 1
        ai = a3(:,i);   % m x 1
        J = J - sum(yi .* log(ai) + (1 - yi) .* log(1 - ai));

    else if method == 2
        y2(:, i) = yi;
    end
end
if method == 2
    J = -sum(sum(y2 .* log(a3) + (1 - y2) .* log(1 - a3)));
end
J = J / m;



%%%%%%%%%%%%%%%%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

Delta_3 = zeros(m, num_labels);
Delta_2 = zeros(m, hidden_layer_size);
for t=1:m
    xt = X(t,:)';                % input_layer_size x 1
    yt = y2(t,:)';               % num_labels x 1

    at1 = xt;

    zt2 = Theta1 * [1; at1];     % hidden_layer_size x (input_layer_size + 1) x 1
    at2 = sigmoid(zt2);

    zt3 = Theta2 * [1; at2];     % num_labels x (hidden_layer_size + 1) x 1
    at3 = sigmoid(zt3);

    delta_3t = at3 - yt;         % num_labels x 1
    delta_2t = Theta2(:,2:end)' * delta_3t .* sigmoidGradient(zt2); % hidden_layer_size x (num_labels) x 1

    Delta_3(t,:) = delta_3t';
    Delta_2(t,:) = delta_2t';
end
%Delta_3 = a3 - y2;


%Delta_2 = Theta2' * Delta_3 .* sigmoidGradient(z2);


%%%%%%%%%%%%%%%%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
J = J + (sum([Theta1(:,2:end)(:); ...
              Theta2(:,2:end)(:)].^2)) * lambda / (2 * m);

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
