function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
  C_choice     = [0.01 0.03 0.1 0.3 1 3 10 30]';
  sigma_choice = [0.01 0.03 0.1 0.3 1 3 10 30]';

  prediction_errors = zeros(length(C_choice),length(sigma_choice));
  
  final_result = zeros(length(C_choice)*length(sigma_choice), 3);
  
  row = 1; 
  
  for i_index = 1:length(C_choice)
    for j_index = 1:length(sigma_choice)
      model= svmTrain(X, y, C_choice(i_index), @(x1, x2) gaussianKernel(x1, x2, sigma_choice(j_index)));
      predictions = svmPredict(model, Xval);
      
      prediction_errors(i_index, j_index) = mean(double(predictions ~= yval));
      
      final_result(row, :) = [prediction_errors(i_index, j_index), C_choice(i_index), sigma_choice(j_index)];
      row = row + 1;
      
    endfor
  endfor
  
% Sorting predictions_error in ascending order 
  
  final_result = sortrows(final_result, 1);

% C and sigma corresponding to min(prediction_error)
  
  C = final_result(1, 2);
  sigma = final_result(1, 3);
  
% =========================================================================

end
