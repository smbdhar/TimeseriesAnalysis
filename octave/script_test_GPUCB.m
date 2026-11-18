% Test script benchmarking /////////////////////////////////
% This script is for benchmark testing of GP-UCB code.
% Author: Sambuddha Dhar
clear; clc;

% Benchmarking with Rosenbrock function
objfun  = @(m) - ((1 - m(1))^2 + 100*(m(2) - m(1)^2)^2 );
m_min = [0.2; 0.2];   m_max = [2; 3];   %m0 = [-1.5; 2.5];

% Benchmarking with Sphere function
##objfun  = @(m) - ((m(1)-1)^2 + (m(2)-1)^2);
##m_min = [0.2; 0.2];   m_max = [2; 2];

objfun=objfun;
bounds=[m_min, m_max];
N=1000;
n=1000;
iter=1000;

[bestX,bestscore,Xr,y]=GPUCBlog(objfun,bounds,N,n,iter);
bestX
% ---- visualization -----------------------------------------------
figure(91); clf; hold on; box on; grid on;
plot(Xr(:,1),Xr(:,2),'bo');
plot(bestX(:,1),bestX(:,2),'ro');
xlim([bounds(1,1) bounds(1,2)]); ylim([bounds(2,1) bounds(2,2)])
title(sprintf('Iteration=%d',iter), 'fontsize',14)
xlabel('m1'); ylabel('m2');

