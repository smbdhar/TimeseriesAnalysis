function [residualsq, dcalpred, VRcalpred, AScalpred, scalar_const] = timeseriesregression(tau1, tau2, lambda, beta2, t_pred, tobsfit, dobsfit, tobspred, dobspred)
% Author: Sambuddha Dhar


  Ndobs     = size(dobsfit, 1);
  dcalpred  = zeros(Ndobs, length(t_pred));
  VRcalpred = zeros(Ndobs, length(t_pred));
  AScalpred = zeros(Ndobs, length(t_pred));
  residual  = zeros(Ndobs, 1);
  scalar_const = zeros(Ndobs, 3);

  for kk = 1:Ndobs
    tobsfitk  = tobsfit(kk,:);   dobsfitk  = dobsfit(kk,:);
    tobspredk = tobspred(kk,:);  dobspredk = dobspred(kk,:);

    [mb] = linearLeastSquareInversion(tau1, tau2, lambda, tobsfitk, dobsfitk);

    [dcalpredk, VRcalpredk, AScalpredk] = analyticalmodel(t_pred, [mb(:)', tau2,tau1,lambda]);

    % Error calculation ---------------------------------------------------------
    obsfilter = @(Dinp) [movmean(Dinp(1:7), 2), movmean(Dinp(8:12), 14), movmean(Dinp(13:end), 14)];
    dcalpredm = interp1(t_pred, dcalpredk, tobspredk);
    obs_residual = (obsfilter(dobspredk) - dcalpredm).^2;
    obs_residual(isnan(obs_residual)) = 0;
    residualk = sum(obs_residual, 2);
    % -------- ---------------------------------------------------------

    dcalpred(kk,:)  = dcalpredk;
    VRcalpred(kk,:) = VRcalpredk;
    AScalpred(kk,:) = AScalpredk;
    residualsq(kk,:)  = residualk;
    scalar_const(kk,:)= mb(:)';
  endfor

  ##  % Gaussian Error ---------------------------------------------------------
  ##    residualgp = -sum(residual, 1);
  ##    residualgp = 1e3 .* exp(0.5 .* residualgp ./ beta2);
  ##    residualgp = ((2*pi*beta2).^(-3/2)) .* residualgp;
  ##    posteriorPDF=residualgp;
  ##  posteriorPDF=((2*pi*beta2).^(-numel(dobspred)/2)) .* exp(-sum(residual, 1)./(2*beta2));
endfunction

function [mb] = linearLeastSquareInversion(tau1, tau2, lambda, tobs, dobs)
  tobs = tobs - tobs(1);   tobs = tobs(:);
  ##  tsi  = ts(1);            ts   = ts(:) - tsi;
  di   = dobs(1);          dobs = dobs - di;   dobs = dobs(:);
  C = eps .* eye(length(tobs));

  A  = [tobs.^0, viscousflow(tau1, tau2, tobs), fltcreep(lambda, tobs)];
  mb = ((A' * (C\A)) \ A') * (C\dobs);
endfunction

function strain = viscousflow(tau1, tau2, tspan)
  n = 3;  m = 3;
  nn1 = -1 ./ (n-1);  nn2 = -1 ./ (m-1);
  strain = 2 - ((1 + (tspan./tau1)).^nn1) - ((1 + (tspan./tau2)).^nn2);
  strain=strain(:);
endfunction


function slip = fltcreep(lambda, tspan)
  G    = 30e3;
  Lflt = 90e3;
  V0   = 0.08;
  sn   = 300;
  lambda = lambda .* sn;
  slip = 1 - (2 .* lambda .* acoth(coth(1./(2.*lambda)) .* exp((tspan.*2.*V0.*G) ./ (lambda.*Lflt))));
  slip=slip(:);
endfunction
