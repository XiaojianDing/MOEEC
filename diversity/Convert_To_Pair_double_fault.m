% function D = double_fault(Prediction,ytest)
% N=size(Prediction,1);
% n=size(Prediction,2);
% li=zeros(n,1);
% for i=1:n
%     li(i)=mean(Prediction(:,i)~=ytest);
% end
% L=length(unique(ytest));
% P=mean(li);
% D = 1 / (N * L * (L - 1)) * sum(li .^ 2) - 1 / L - P + 1;
% end
function DF = double_fault1(errorMatrix,ytest)
    % N 是数据集中的样本数量
    % L 是分类器的数量
    % errorMatrix 是一个N x L的矩阵，其中如果分类器j在样本i上分类错误，则errorMatrix(i,j) = 1，否则为0
N=size(errorMatrix,1);
L=size(errorMatrix,2);
for i=1:N
for j=1:L
    if errorMatrix(i,j)==ytest(i)
        errorMatrix(i,j)=1;
    else
        errorMatrix(i,j)=0;
    end
end
end
    doubleFaultCount = 0;
    for i = 1:N
        for j = 1:L
            for k = (j+1):L
                if errorMatrix(i,j) == 0 && errorMatrix(i,k) == 0
                    doubleFaultCount = doubleFaultCount + 1;
                end
            end
        end
    end
    
    DF = (2 * doubleFaultCount) / (N * L * (L - 1));
end
