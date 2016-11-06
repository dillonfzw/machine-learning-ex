function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


mode = 2;
if mode == 1

CVec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaVec = CVec;
J = zeros(length(CVec), length(sigmaVec));

for CIdx = 1:length(CVec)
    for sigmaIdx = 1:length(sigmaVec)
        C = CVec(CIdx);
        sigma = sigmaVec(sigmaIdx);

        fprintf('fzw: Trying C=%f, sigma=%f ...\n', C, sigma);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

        predicts = svmPredict(model, Xval);

        J(CIdx, sigmaIdx) = mean(double(predicts ~= yval));
    end
end

display(J);

[sigmaMinVec, CMinIdxVec] = min(J);
[CMinVec, sigmaMinIdxVec] = min(sigmaMinVec);
sigmaMinIdx = sigmaMinIdxVec;
CMinIdx = CMinIdxVec(sigmaMinIdx);

C = CVec(CMinIdx);
sigma = sigmaVec(sigmaMinIdx);

fprintf('fzw: best param is C=%f, sigma=%f, J=%f\n', C, sigma, CMinVec);
fprintf('fzw: expected C = 1, sigma=0.1, J=0.03\n');

else if mode == 2

C= 1.0;
sigma = 0.1;

end


% =========================================================================

end
