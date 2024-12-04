%
function [classifiers] = trainclassifiers(xapp, yapp,num_classifiers, dim, epsilon, kernel, verbose, c)
classifiers = cell(num_classifiers, 1);
for i = 1:num_classifiers
kerneloption = rand(1, dim) * dim;
tic
[xsup,w,b,pos,pos1,pos2,pos3,ps,H,n3]=svmclassrand(xapp,yapp,c,epsilon,kernel,kerneloption,verbose);
classifiers{i} = struct('xsup', xsup, 'w', w, 'b', b, 'kerneloption', kerneloption,'n3',n3);
end
end


%%bagging
% function [classifiers] = trainclassifiers(xapp, yapp,num_classifiers, dim, epsilon, kernel, verbose, c)
% classifiers = cell(num_classifiers, 1);
% num_samples=fix(size(xapp,1)/2);
% num1=size(xapp,1);
% for i = 1:num_classifiers
%      indices = randi([1,num1],1,num_samples);
% 
%      x_subset = xapp(indices, :);
%      y_subset = yapp(indices);
% kerneloption = rand(1, dim) * dim;
% tic
% [xsup,w,b,pos,pos1,pos2,pos3,ps,H,n3]=svmclassrand(x_subset,y_subset,c,epsilon,kernel,kerneloption,verbose);
% classifiers{i} = struct('xsup', xsup, 'w', w, 'b', b, 'kerneloption', kerneloption,'n3',n3);
% end
% end