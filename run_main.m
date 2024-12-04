clc;
clear;
addpath(genpath('./'));
load('result_moeec.mat');
acc = zeros(1, 1);
recall = zeros(20, 1);
precision = zeros(20, 1);
F1 = zeros(20, 1);
best_fitnesss=cell(20,1);
datasets = {'banknote', 'heart', 'iono', 'liver', 'cancer', 'fourclass', 'pwlinear', 'appendicitis', 'indian', 'sonar'};
split_points = 200;
c = 10;
load iono;             
results = cell(length(datasets), 1);
% m_values=[2,5,10,20,30];
% k_values=[1,3,5,7,9];



for i=3
    m = 10;
    k = 1;
    iono = [iono(:, 2:end), iono(:, 1)];

    for runs = 1:20
        rand_sequence = randperm(size(iono, 1));
        temp_dataset = iono;
        iono = temp_dataset(rand_sequence, :);

        P = iono(1:split_points,1:end-1)';
        T = iono(1:split_points, end)';
        pos5 = find(T == 0);
        T(pos5) = -1;

        xapp = P';
        yapp = T';

        P1 = iono(split_points + 1:end, 1:end-1)';
        T1 = iono(split_points + 1:end, end)';
        pos6 = find(T1 == 0);
        T1(pos6) = -1;
 
        xtest = P1';
        ytest = T1';

        trainData = [xapp, yapp];
        dim = size(xapp, 2);
        num_classifiers = 100;
        kernel = 'gaussian';
        verbose = false;
        epsilon = 0.000001;
        w = 0.9; % Inertia weight
        c1 = 2; % Individual learning factor
        c2 = 2; % Social learning factor
        testData = [xtest, ytest];

        [classifiers] = trainclassifiers(xapp, yapp, num_classifiers, dim, epsilon, kernel, verbose, c);
        [final_idx, iteration, global_best_fitness, time,  alpha_values, beta_values, gamma_values] = AOWE(classifiers, xtest, ytest, m, w, c1, c2, k);
        selected = classifiers(final_idx);
        decision_matrix = myprediction(selected, testData);
        final_y = mode(sign(decision_matrix), 2);
        error = mean(final_y ~= ytest);
        acc(runs) = 1 - error;
        confMat = confusionmat(ytest, final_y);

        TP = confMat(2, 2);  % True positive
        FP = confMat(1, 2);  % False positive
        FN = confMat(2, 1);  % False negative

        recall(runs) = TP / (TP + FN);
        precision(runs) = TP / (TP + FP);
        F1(runs) = 2 * (precision(runs) * recall(runs)) / (precision(runs) + recall(runs));
    end

    % Save results for the current dataset
    result = struct();
    result.(datasets{i}).accuracy_mean = mean(acc);
    result.(datasets{i}).accuracy_std = std(acc);
    result.(datasets{i}).acc_values = acc;  % Save the acc array
    result.(datasets{i}).recall_values = recall;
    result.(datasets{i}).F1_values = F1;


    results{i} = result;
    disp(['c_number: ', num2str(m)]);
    fprintf('Dataset: %s\n', datasets{i});
    disp(['Mean Accuracy: ', num2str(mean(acc))]);
    disp(['Std Dev: ', num2str(std(acc))]);
end