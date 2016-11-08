function visualizeBoundary(X, y, model, varargin)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on

% Trouble with the contour plot (visualizeBoundary.m)
%
% Octave 3.8.x and higher
%
% If you have Octave 3.8.x, the ex6 script will not plot decision boundary, and
% prints 'Unknown hggroup property Color' with stack trace.
%
% One fix is to modify line 21 in visualizeBoundary.m with this code:
% >> 01: contour(X1, X2, vals, [1 1], 'linecolor', 'blue');
% >> 02:
% (Note: I tried this and although the error went away, I still don't see any contour line drawn; sokolov 3/22/2015)
%
% I had the same problem with the line not displaying until i changed the [0 0] to [1 1] - tmcarthur 7/1/2016
%
% OR
%
% If you change line 21 to following, it will show two lines and will work with >= 3.8.x .
% >> 01: contour(X1, X2, vals);
% >> 02:
% For more information see
%
% http://lists.gnu.org/archive/html/octave-bug-tracker/2014-01/msg00226.html
contour(X1, X2, vals, [1 1], 'linecolor', 'blue');

hold off;

end
