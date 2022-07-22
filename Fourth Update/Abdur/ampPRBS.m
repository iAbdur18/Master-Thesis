%% ampPRBS calculates an amplitude modulated pseudo random binary signal
% nSample : scalar integer, length of the signal
% ampRange: Nx2 matrix, each row contains the min and max of the i-th
%           signal, N is the number of independent input signals
% noChange: scalar integer or vector of length N, min number of constant 
%           entries without a change in the input signals

% Author: Jens Ehlhardt (2021)

% adopted from: 
% https://de.mathworks.com/matlabcentral/answers/373968-how-do-i-create-an-
% amplitude-modulated-pseudorandom-bit-sequence-using-the-system-
% identification-too

function finalSignal = ampPRBS(nSamples, ampRange, noChange)
%% some checks
assert(license('test','identification_toolbox'),...
    "Matlab's 'System Identification Toolbox' needs to be installed!")
assert(isnumeric(nSamples) && sum(size(nSamples)) == 2 && mod(nSamples,1) == 0 ...
    && nSamples>0, 'nSamples needs to be an Integer greater 0')
assert(isnumeric(noChange) && size(noChange, 2)== 1 && mean(mod(noChange,1)) == 0 ...
&& mean(noChange>0) == 1, 'noChange needs to be an Integer or a Nx1 vector greater 0')
assert(size(ampRange,2) == 2 && isnumeric(ampRange),...
    'Size of ampRange needs to be Nx2');
% fall überprüfen, wenn ampRange und noChange N>1
%% get some information
nInputs = size(ampRange,1);
ampMax = max(ampRange, [], 2);
ampMin = min(ampRange, [], 2);

if size(noChange, 1) == 1
   noChange = repmat(noChange, nInputs, 1); 
end

%% create the signal

% create output variable
finalSignal = NaN(nSamples, nInputs);

% alter the amplitude
for j = 1:nInputs % for all input channels
    % create prbs signal with noChange-constant intervals
    u = idinput([nSamples],'prbs',[0 1/noChange(j)],[-1 1]);
    % find the switching points
    d = diff(u(:)); 
    idx = find(d) + 1; 
    idx = [1;idx];
    
    for ii = 1:length(idx) - 1 % for all switching points
        amp = randn; % randn : normal distribution, rand : uniform distribution
        u(idx(ii):idx(ii+1)-1) = amp*u(idx(ii)); % change all amplitudes ...
        % between two switching points
    end % for ii
    finalSignal(:,j) = u;
end % for j

% normalize the signal between [0 1]
finalSignal = finalSignal./max(abs(finalSignal),[],1);
finalSignal = (finalSignal+1)/2;

% transform to the desired amplitudes
finalSignal = (finalSignal.*(ampMax'-ampMin')) + ampMin';
