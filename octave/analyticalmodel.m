function [varargout]=analyticalmodel(tspan,m)
% ------------------------------------------------------------------ %
% Author: Sambuddha Dhar (2021-11-28)
%
% Related paper:
% Dhar, S., & Muto, J. (2023). Function model based on nonlinear transient
% rheology of rocks: An analysis of decadal GNSS time series after
% the 2011 Tohoku-oki earthquake. Geophysical Research Letters, 50,
% e2023GL103259. https://doi.org/10.1029/2023GL103259

% dcal=c1+a*VR(tau1,tau2,t)+b*AS(lambda,t);
% ------------------------------------------------------------------ %
[c1,a,b,tau1,tau2,lambda]=deal(m(1),m(2),m(3),m(4),m(5),m(6));
[strain]=viscousflow(tau1,tau2,tspan);
[slip]=fltcreep(lambda,tspan);
VR=a.*strain;  AS=b.*slip; dcal=c1+VR+AS;
varargout{1}=dcal;  %;
varargout{2}=VR; %
varargout{3}=AS; %
end

function [strain]=viscousflow(tau1,tau2,tspan)
n=3; m=3;
nn1=-1./(n-1); nn2=-1./(m-1);
strain=2-((1+(tspan./tau1)).^nn1) -(((1+(tspan./tau2)).^nn2));
end

function [slip]=fltcreep(lambda,tspan)
G=30e3;
Lflt=90e3;
V0=0.08;%0.0599; %
sn=300;
lambda=lambda.*sn;
slip=1-( 2.*lambda.*acoth(coth(1./(2.*lambda)).* exp((tspan.*2.*V0.*G)./(lambda.*Lflt))) );
end
