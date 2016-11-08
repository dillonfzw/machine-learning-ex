function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

m = length(yval);

stepsize = (max(pval) - min(pval)) / 1000;
fprintf('min(pval) = %f, max(pval) = %f\n', min(pval), max(pval));
for epsilon = min(pval):stepsize:max(pval)
%for epsilon = 8.99e-05
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    predict = pval < epsilon;
    real = yval == 1;  % normalize to {0, 1}

    allPredictPositive = sum(predict); % = tp + fp
    allRealTrue = sum(real); % = tp + tn;

    tp = sum(predict & real);
    %fp = sum(predict & ~real);
    %tn = sum(~predict & real);
    %fn = sum(~predict & ~real);
    

    if allPredictPositive == 0
        fprintf('tp + fp == 0 causes precision divided by zero error, while epsilon equals to %f\n', epsilon);
        continue;
    elseif allRealTrue == 0
        fprintf('tp + tn == 0 causes recall divided by zero error, while epsilon equals to %f\n', epsilon);
        continue;
    end

    prec = tp / allPredictPositive;
    rec  = tp / allRealTrue;

    F1 = 2 * prec * rec / (prec + rec);








    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
