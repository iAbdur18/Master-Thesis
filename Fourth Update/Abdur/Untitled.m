u = idinput(120,'prbs',[0 1],[-1 1]);
d = diff(u);
idx = find(u) + 1;
idx = [1;idx];
for ii = 1:length(idx) - 1
     amp = randn;
     u(idx(ii):idx(ii+1)-1) = amp*u(idx(ii));
end
u = u/max(u);
u = iddata([],u,1);
% Plot the data
figure
plot(u)