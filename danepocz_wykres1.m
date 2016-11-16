X = [1.0 -1.2];
T = [0.5 1.0];
w_range = -1:0.2:1;  b_range = -1:0.2:1;
ES = errsurf(X,T,w_range,b_range,'purelin');
plotes(w_range,b_range,ES);
maxlr = 0.40*maxlinlr(X,'bias');
net = newlin([-2 2],1,[0],maxlr);
net.trainParam.goal = .001;
%
p = rand(2,1);
a = rand(3,1);
lp.lr = 0.5;
dW = learnh([],p,[],[],a,[],[],[],[],[],lp,[])