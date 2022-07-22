% for reproducability 
seed = randi(1e3);
rng(seed)


%% create some input output data
F = oneTankModel(1); % get a symbolic integrator of the system
nSamples = 3600; % number of data points

% create some random input signals vor testing, validation and training
inputSignal = ampPRBS(nSamples, [0.2, 0.4], 60);

% initial height
H0 = 0.5;

% preallocate some variables to store the target values
yTrain = NaN(size(inputSignal));
    yTrain(1) = H0;

% generate the target values by integrating the system
for j = 1:nSamples
    y_j = F('x0', yTrain(j), 'p', inputSignal(j));    
    yTrain(j+1) = full(y_j.xf); 
end

% plot the data
if 0
    figure(11)
        subplot(211)
            hold on; grid on;
            plot(yTrain)
            xlabel("t [s]"); ylabel("h [m]")
        subplot(212)
            title("Validation Data")
            hold on; grid on;
            plot(inputSignal)
            xlabel("t [s]"); ylabel("u [%]")
end

% get the correct matrices for training the NARX models
nDelay = 1; % input/target delay

% this function shifts and stacks the data to get the correct shape
[inputs, targets] = getInputOutputForNARX(inputSignal, yTrain(1:end-1), nDelay, nDelay);

% However, matlab wants the data to be cell arrays
% inputs = tonndata(inputs, 1, 0);
% targets = tonndata(targets, 1, 0);


%% set up the network and train it

% initialize network with 2 layers and 10 neurons each
net = fitnet([10, 10]);
   
% train the network
trainedNet = train(net, inputs, targets);

%% get the one-step predictions 
y_hat = trainedNet(inputs);

figure()
    grid on; hold on;
    plot(y_hat); plot(targets, '--')
    legend("y_{hat}", "data")
    xlabel("t [s]"); ylabel("h [m]")

%% One-Tank example, returns a symbolic integrator
function [F] = oneTankModel(ts)
    import casadi.*
    
    V = SX.sym('V');
    H = SX.sym('H');
    
    F_in = 180 / 1000 / 3600;
    A  = 0.04^2 * pi;
    
    c_13 = 3.4375e7; % [s²/m^5]
    c_23 = 0.9128e7; % [s²/m^5]

    
    dH_dt = F_in/A - 1/A * (V * sqrt(H))/sqrt(c_13 * V^2 + c_23);
    
    ode = struct('x', H, 'p', V, 'ode', dH_dt);
    opts.tf = ts;
    opts.t0 = 0;
    F = integrator('F', 'idas', ode, opts);
    
    
end