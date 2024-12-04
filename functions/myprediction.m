function selectedY = myprediction(classifiers, testData)
    xtest = testData(:,1:end-1);
    selectedY = ones(length(testData(:,end)), length(classifiers));
    % 对每个选出的分类器进行预测
for i = 1:length(classifiers)
    classifier = classifiers{i};
    xsup = classifier.xsup;
    w = classifier.w;
    b = classifier.b;
    kerneloption = classifier.kerneloption;
    kernel='gaussian';
    n3=classifier.n3;
    
    % 使用选出的分类器进行预测
    y1 = svmvalrand(xtest, xsup, w, b, kernel, kerneloption, n3);
    selectedY(:,i)=y1;
end
    end
    
