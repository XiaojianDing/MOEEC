function D=Convert_To_Pair_Disagreement(C)
% This code used to compute disagreement coefficient between more than two classifiers
% This code implemented by Eng. Alaa Tharwat Abd El Monaaim - Egypt- TA in
% El Shorouk Academy
% engalaatharwat@hotmail.com  +201006091638
% C (MxN) M represents number of objects, N represents number of
% classifiers, T is the labels of the objects (0 represents  false, 1 true)

S=0;
Counter=0;
for i=1:size(C,2)
   for j=i:size(C,2)
      if(i==j) 
          continue;
      end
    S=S+Disagreement(C(:,i),C(:,j));      
    Counter=Counter+1;
   end
end

D=S/(size(C,2));