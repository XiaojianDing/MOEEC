function P=Convert_To_Pair_Correlation(errorMatrix,ytest)
% This code used to compute correlation coefficient between more than two
% classifiers
% This code implemented by Eng. Alaa Tharwat Abd El Monaaim - Egypt- TA in
% El Shorouk Academy
% engalaatharwat@hotmail.com  +201006091638
% C (MxN) M represents number of objects, N represents number of
% classifiers, T is the labels of the objects (0 represents  false, 1 true)
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
S=0;
Counter=0;
for i=1:size(errorMatrix,2)
   for j=i:size(errorMatrix,2)
      if(i==j) 
          continue;
      end
    S=S+Correlation(errorMatrix(:,i),errorMatrix(:,j));      
    Counter=Counter+1;
   end
end

P=S/(size(errorMatrix,2));