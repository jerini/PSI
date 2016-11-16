clear all;
close all;

% Learning rate
% Choose a good learning rate
%------------
gamma = 0;
%------------
% Number of training steps
% choose it
%------------
numEpochs = 5000;
%------------

% dataset X (p times 2 matrix)
% X(i,:) is the i-th Input-Vector
% the dataset is loaded
% the commented line loads dataset 2
%------------
load ue3_dp1.mat;
% load ue3_dp2.mat;
%------------

% number of datapoints
p = size(X,1);


%% plot data
figure(1);
clf;
plot(X(:,1),X(:,2),'.');

%% Compute correlation matrix
% use matrix multiplication
%------------------
C = rand(2,2);
%------------------

%% eigenvalues and eigenvectors
%% eigV(:,i)...i-th eigenvector of C
%% lambda(i,i)...i-th eigenvalue of C
[eigV, lambda] = eig(C);

% initialize weight vector
% random initialization
w = rand(2,1);

%% number of graphics updates
inter_epoch = 5;
for i=1:numEpochs
  % one learn step
  
  %% choose datapoint
  mu = ceil(50*rand(1,1));
  X_mu = X(mu,:); %% momentary data point
  
  %% compute output y
  y = X_mu*w;
  
  %% Apply learning rule
  % code Hebbs rule here
  %------------
  w = w;
  %------------
  
  % Plot new weight vector
  if(mod(i,inter_epoch)==0) 
    fprintf('Epoch %g\n',i);
    figure(1);
    clf;
    plot(X(:,1),X(:,2),'.');
    hold on;
    w_normed = w/norm(w);
    plot([0,w_normed(1)],[0,w_normed(2)],'r-');
  end
end

% Plot final weight vector
figure(1);
clf;
plot(X(:,1),X(:,2),'.');
hold on;
w_normed = w/norm(w);
plot([0,w_normed(1)],[0,w_normed(2)],'r-');

% Print eigenvalues and eigenvectors
for i=1:2
  fprintf('Eigenwert %g: %g\n',i,lambda(i,i));
  fprintf('Eigenvektor %g: (%g, %g)\n',i,eigV(1,i),eigV(2,i));
end