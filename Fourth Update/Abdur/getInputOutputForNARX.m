function [inputs, outputs] = getInputOutputForNARX(rawIn, rawOut, xShift,...
    uShift)
% 
%     
%     rawIn = rawIn';
    %% order the intputs as follows:
   
    % then shifted inputs in the same order
    xShift = xShift + 1;
    uShift = uShift + 1;
    
    % TODO: checks for inputs einbauen
    nInputs = min(size(rawIn));
    nStates = min(size(rawOut));
    nSamples = max(size(rawIn));
    maxShift = max(xShift, uShift);
    inputs = NaN(nStates*xShift + nInputs*uShift , nSamples- (maxShift-1));
    
    % rawIn und rawOut müssen spaltenbasiert sein
    % add shifted states
    for i = 1:xShift
       for j = 1:nStates
          inputs((i-1)*nStates + j, :) =...
              rawOut(maxShift-(i-1):end-(i-1), j)'; 
       end % for_j
    end % for_i
    % add shifted inputs
    for i = 1:uShift
        for j = 1:nInputs
            inputs(nStates*xShift + (i-1)*nInputs + j,:) = ...
                rawIn(maxShift-(i-1):end-(i-1), j)'; 
        end % for_j
    end % for_i
    
    % cut the targets
    inputs = inputs(:, 1:end-1);
    outputs = rawOut(maxShift+1:end,:)';
end