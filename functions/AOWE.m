

function [final_idx, iteration, global_best_fitness, time, alpha_values, beta_values, gamma_values] = AOWE(classifier_list, xtest, ytest, m, w, c1, c2, k)
    tic; % start timer

    num_particles = 50; % number of particles
    max_iterations = 100; % maximum iterations
    num_classifiers = length(classifier_list);

    particles_position = zeros(num_particles, num_classifiers); % initialize particle positions
    particles_velocity = rand(num_particles, num_classifiers); % initialize particle velocities
    personal_best = particles_position; % initialize personal best positions
    global_best = []; % initialize global best position
    stagnation_threshold = 5; % threshold for stagnation
    consecutive_stagnation = 0; % counter for consecutive stagnation

    personal_best_fitness = inf(1, num_particles);
    global_best_fitness = inf; % initialize global best fitness

    all_particle_positions = zeros(max_iterations + 1, num_particles, num_classifiers);
    all_global_best_positions = zeros(max_iterations + 1, num_classifiers);
    global_best_fitness_curve = zeros(1, max_iterations + 1); % Initialize global best fitness curve

    % Initialize arrays to store alpha, beta, gamma, and error1 values over iterations
    alpha_values = zeros(1, max_iterations);
    beta_values = zeros(1, max_iterations);
    gamma_values = zeros(1, max_iterations);
    error1_values = zeros(1, max_iterations); % Initialize error1 values array

    for i = 1:num_particles
        random_selection = randperm(num_classifiers, m); % randomly select m positions
        particles_position(i, random_selection) = 1; % set m positions to 1
        
        % Compute fitness and diversity measures
        [fitness, alpha_value, beta_value, gamma_value, error1] = compute_fitness(classifier_list, particles_position(i, :), xtest, ytest, k);
            % Store alpha, beta, gamma, and error1 values for each iteration
        alpha_values(1) = alpha_value;
        beta_values(1) = beta_value;
        gamma_values(1) = gamma_value;
        error1_values(1) = error1; % Store the error1 value for each iteration
        personal_best_fitness(i) = fitness; % update personal best fitness
        
        % Update personal best and global best
        if fitness < global_best_fitness
            global_best = particles_position(i, :);
            global_best_fitness = fitness;
        end
    end

    all_particle_positions(1, :, :) = particles_position;
    all_global_best_positions(1, :) = global_best;
    global_best_fitness_curve(1) = global_best_fitness; % Record initial global best fitness

    previous_global_best = global_best;

    for iteration = 1:max_iterations
        for i = 1:num_particles
            % Update velocity
            particles_velocity(i, :) = w * particles_velocity(i, :) ...
                + c1 * rand() * (personal_best(i, :) - particles_position(i, :)) ...
                + c2 * rand() * (global_best - particles_position(i, :));
            
            % Limit velocity
            particles_velocity(i, :) = min(max(particles_velocity(i, :), -1), 1);

            particles_position(i, :) = round(particles_position(i, :) + particles_velocity(i, :));
            particles_position(i, :) = max(particles_position(i, :), 0);

            % Ensure exactly m ones in the position
            num_ones = sum(particles_position(i, :) ~= 0);
            if num_ones < m
                zero_indices = find(particles_position(i, :) == 0);
                rand_indices = randperm(length(zero_indices), m - num_ones);
                particles_position(i, zero_indices(rand_indices)) = 1;
            elseif num_ones > m
                one_indices = find(particles_position(i, :) ~= 0);
                rand_indices = randperm(length(one_indices), num_ones - m);
                particles_position(i, one_indices(rand_indices)) = 0;
            end

            % Compute fitness and record diversity measures
            [fitness, alpha_value, beta_value, gamma_value, error1] = compute_fitness(classifier_list, particles_position(i, :), xtest, ytest, k);
            
            % Update personal best
            if fitness < personal_best_fitness(i)
                personal_best(i, :) = particles_position(i, :);
                personal_best_fitness(i) = fitness;
            end

            % Update global best
            if fitness < global_best_fitness
                global_best = particles_position(i, :);
                global_best_fitness = fitness;
            end
        end

        all_particle_positions(iteration + 1, :, :) = particles_position;
        all_global_best_positions(iteration + 1, :) = global_best;
        global_best_fitness_curve(iteration + 1) = global_best_fitness; % Record global best fitness
        
        % Store alpha, beta, gamma, and error1 values for each iteration
        alpha_values(iteration+1) = alpha_value;
        beta_values(iteration+1) = beta_value;
        gamma_values(iteration+1) = gamma_value;
        error1_values(iteration+1) = error1; % Store the error1 value for each iteration

        % Check for stagnation
        if iteration > 1 && isequal(global_best, previous_global_best)
            consecutive_stagnation = consecutive_stagnation + 1;
        else
            consecutive_stagnation = 0;
        end

        previous_global_best = global_best;

        % Terminate if stagnation threshold reached
        if consecutive_stagnation >= stagnation_threshold
            final_idx = find(global_best);
            break;
        end
    end

    time = toc; % stop timer
end

function [fitness, alpha_value, beta_value, gamma_value, error1] = compute_fitness(classifier_list, particle_position, xtest, ytest, k)
    idx = find(particle_position);
    selected_classifiers = classifier_list(idx);
    trainData = [xtest, ytest];
    selected_predictions = myprediction(selected_classifiers, trainData);
    Y = sign(selected_predictions);

    % Compute error rate
    Y1 = mode(Y, 2);
    error1 = mean(Y1 ~= ytest);

    % Compute standard deviation of accuracy
    accuracy_list = zeros(size(selected_predictions, 2), 1);
    for i = 1:size(selected_predictions, 2)
        pred = Y(:, i);
        accuracy_list(i) = mean(pred == ytest);
    end
    s_acc = std(accuracy_list);

    % Compute diversity measure
    D = double_fault1(Y, ytest);

    % Compute weights based on diversity measures
    f1 = @(x) (2/pi) * atan(1./(k*x));
    beta_value = f1(D);
    gamma_value = f1(s_acc);
    alpha_value = f1(error1);
    sumw = alpha_value + beta_value + gamma_value;
    alpha_value = alpha_value / sumw;
    beta_value = beta_value / sumw;
    gamma_value = gamma_value / sumw;

   fitness = alpha_value * error1 + beta_value * D + gamma_value * s_acc;

end
