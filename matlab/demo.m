% This is a matlab script illustrating how to use CLFDM_lib to learn
% an arbitrary model from a set of demonstrations.

%Reference paper:
% S.M. Khansari-Zadeh and A. Billard (2014), "Learning Control Lyapunov Function to Ensure Stability
% of Dynamical System-based Robot Reaching Motions." Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.

%%
%The following example demos are provided
close all;
modelNames = {'w','Sshape'};
modelNumber = 2; %choose either 1 or 2 to select a different example

%% Putting CLFDM and regress_gauss_mix library in the MATLAB Path
if isempty(regexp(path,['CLFDM_lib' pathsep], 'once'))
    addpath([pwd, '/CLFDM_lib']);    % add SEDS dir to path
end
if isempty(regexp(path,['regress_gauss_mix_lib' pathsep], 'once'))
    addpath([pwd, '/regress_gauss_mix_lib']);    % add regress_gauss_mix dir to path
end

%% User Parameters and Setting
demoIndices = []; %to avoid a stupid matlab bug
load(['ExampleModels/' modelNames{modelNumber}],'Data','demoIndices')
% the variable 'Data' composed of 3 demosntrations. Each demonstrations is
% recorded from Tablet-PC at 50Hz. Datas are in millimeters.

%% Setting parameters of the Control Lyapunov Function
clear Vxf0
switch modelNumber
    case 1
        Vxf0.L = 2; %the number of asymmetric quadratic components L>=0.
    case 2
        Vxf0.L = 1; %the number of asymmetric quadratic components L>=0.
end
Vxf0.d = size(Data,1)/2;
Vxf0.w = 1e-4; %A positive scalar weight regulating the priority between the
               %two objectives of the opitmization. Please refer to the
               %page 7 of the paper for further information.

% A set of options that will be passed to the solver. Please type
% 'doc preprocess_demos' in the MATLAB command window to get detailed
% information about other possible options.
options.tol_mat_bias = 10^-1; % a very small positive scalar to avoid
                              % having a zero eigen value in matrices P^l [default: 10^-15]

options.display = 1;          % An option to control whether the algorithm
                              % displays the output of each iterations [default: true]

options.tol_stopping=10^-10;  % A small positive scalar defining the stoppping
                              % tolerance for the optimization solver [default: 10^-10]

options.max_iter = 500;       % Maximum number of iteration for the solver [default: i_max=1000]

options.optimizePriors = true;% This is an added feature that is not reported in the paper. In fact
                              % the new CLFDM model now allows to add a prior weight to each quadratic
                              % energy term. IF optimizePriors sets to false, unifrom weight is considered;
                              % otherwise, it will be optimized by the sovler.

options.upperBoundEigenValue = true; %This is also another added feature that is impelemnted recently.
                                     %When set to true, it forces the sum of eigenvalues of each P^l
                                     %matrix to be equal one.



%% Estimating an initial guess for the Lyapunov function
b_initRandom = false;
if b_initRandom
    lengthScale = sqrt(var(Data(1:Vxf0.d,:)'));
    lengthScaleMatrix = sqrtm(cov(Data(1:Vxf0.d,:)'));
    lengthScale = lengthScale(:);
    Vxf0.Priors = rand(Vxf0.L+1,1);
    for l=1:Vxf0.L+1
        tmpMat = randn(Vxf0.d,Vxf0.d);
        Vxf0.Mu(:,l) = randn(Vxf0.d,1).*lengthScale;
        Vxf0.P(:,:,l) = lengthScaleMatrix*(tmpMat*tmpMat')*lengthScaleMatrix;
    end
else
    Vxf0.Priors = ones(Vxf0.L+1,1);
    Vxf0.Priors = Vxf0.Priors/sum(Vxf0.Priors);
    Vxf0.Mu = zeros(Vxf0.d,Vxf0.L+1);
    for l=1:Vxf0.L+1
        Vxf0.P(:,:,l) = eye(Vxf0.d);
    end
end

% Solving the optimization
Vxf = optimize_lyapunov(Vxf0,Data,options);

%% Plotting the result
fig = figure;
sp = gca;
hold on
h(1) = plot(sp,Data(1,:),Data(2,:),'r.');
axis tight
ax=get(gca);
axis([ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
      ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);

h(3) = EnergyContour(Vxf,axis,[],[],sp, [], false);
h(2) = plot(0,0,'g*','markersize',15,'linewidth',3);

xlabel('x (mm)','fontsize',15);
ylabel('y (mm)','fontsize',15);
title('Energy Levels of the learned Lyapunov Functions')
legend(h,'demonstrations','target','energy levels','location','southwest')
%% Simulation

% A set of options that will be passed to the Simulator. Please type
% 'doc preprocess_demos' in the MATLAB command window to get detailed
% information about each option.
opt_sim.dt = 0.01;
opt_sim.i_max = 4000;
opt_sim.tol = 1;
d = size(Data,1)/2; %dimension of data
x0_all = Data(1:d,demoIndices(1:end-1)); %finding initial points of all demonstrations

load(['ExampleModels/' modelNames{modelNumber}],'Priors_EM','Mu_EM','Sigma_EM')

% rho0 and kappa0 impose minimum acceptable rate of decrease in the energy
% function during the motion. Refer to page 8 of the paper for more information
rho0 = 1;
kappa0 = 0.1;

in = 1:Vxf.d;
out = Vxf.d+1:2*Vxf.d;
fn_handle_regress_gauss_mix = @(x) regress_gauss_mix(Priors_EM, Mu_EM, Sigma_EM, x, in,out);
fn_handle = @(x) DS_stabilizer(x,fn_handle_regress_gauss_mix,Vxf,rho0,kappa0);

[x xd]=Simulation(x0_all,[],fn_handle,opt_sim); %running the simulator

for i=1:size(x,3)
    h(4) = plot(sp,x(1,:,i),x(2,:,i),'b','linewidth',1.5);
end
lg = legend(h,'demonstrations','target','energy levels','reproductions','location','southwest','orientation','horizontal');
set(lg,'position',[0.0673    0.9278    0.8768    0.0571])
