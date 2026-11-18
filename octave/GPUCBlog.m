function [bestX, bestscore, Xr, y] = GPUCBlog(f, bounds, N, n, iter)
% =================================================================== %
% Multivariate bayesian optimization GP-UCB method
% Author: Sambuddha Dhar
%
% Example: [bestX,bestscore,X,y] = GPUCBlog(f,bounds,N,n,iter)
% Input: f      = loss function; log space
%        bounds = parameter bounds [m1min m1max;
%                                   m2min m2max;...]
%        N      = initial sampling (nodal points)
%        n      = reconnaissance points number (not related to sampling)
%        iter   = number of iterations
% Output: bestX     = best-fit model parameters
%         bestscore = value at best-fit parameters
%         Xr        = the sampling points
%         y         = values at the sampling points
% See details: https://github.com/smbdhar/timeseriesForecast
%
% UCB method follows Srinivas et al. (2012 IEEE):
% https://doi.org/10.1109/TIT.2011.2182033
%
% Copyright (c) 2023 SAMBUDDHA DHAR
%
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files
% (the "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to permit
% persons to whom the Software is furnished to do so, subject to the
% following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
% NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
% OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
% USE OR OTHER DEALINGS IN THE SOFTWARE.
% =================================================================== %

  d = size(bounds, 1);

  randunf = @(a, b, N, d) a + (b - a) .* rand(N, d);
  bounds = log10(bounds);

  % initiate X and f(X) ------------------------------------------------
  X = randunf(bounds(:,1)', bounds(:,2)', N, d);
  y = zeros(size(X, 1), 1);
  for i = 1:size(X, 1)
    y(i) = f(10.^X(i,:));                 % <<Forward model>>
  endfor

  for i = 1:iter
    % initiation of test points
    Xtest = randunf(bounds(:,1)', bounds(:,2)', n, d);

    % mu and sigma of test points
    [mu_Xtest, sigma_Xtest] = return_musigma(X, y, Xtest);
    fxb = max(y);
    EI = UCB(mu_Xtest, sigma_Xtest, fxb, i);
    [~, bestEIid] = max(EI);              % max returns a single index
    bestXtest = Xtest(bestEIid, :);
    ybestXtest = f(10.^bestXtest);        % <<Forward model>>

    % update the search points
    X = [X; bestXtest];
    y = [y; ybestXtest];
  endfor

  Xr = 10.^X;
  bestscore = unique(y(y == max(y), :));
  bestX = 10.^(unique(X(y == max(y), :), 'rows'));
endfunction

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
function [mu_Xtest, sigma_Xtest] = return_musigma(X, fX, Xtest)
% X     = node points [sample x dimension]
% fX    = f(X) where f() is the loss function
% Xtest = test/query points [sample x dimension]
  nv = 5e-14;                             % noise variance
  N = size(X, 1);                         % sample number
  fX = fX + nv * rand(N, 1);
  K = kernelgen(X, X);

  % Cholesky with jitter fallback (Octave returns a partial factor if p > 0)
  jitter = nv;
  [R, p] = chol(K + jitter * eye(N));
  while p > 0
    jitter = jitter * 10;
    [R, p] = chol(K + jitter * eye(N));
  endwhile
  L = R';

  % mean at test points
  Lk = L \ kernelgen(X, Xtest);
  mu_Xtest = (Lk' * (L \ fX));            % column vector

  % variance at test points
  K_ = kernelgen(Xtest, Xtest);
  s2 = diag(K_)' - sum(Lk.^2, 1);
  sigma_Xtest = sqrt(max(s2, 0))';        % clamp rounding negatives
endfunction

function K = kernelgen(a, b)
% a, b = [sample x dimensions]
  kp = 10;                                % kernel parameter
  K = exp((-0.5 ./ kp) .* (sum(a.^2, 2) + sum(b.^2, 2)' - 2 .* (a * b')));
endfunction

function EI = UCB(mux, sigmax, fxb, iter)
% UCB score of test points
  t = floor(iter ./ 1) + 0;  D = 2;  del = 0.1;
  bt = sqrt(2 .* log(D .* t .* t .* pi .* pi ./ (6 .* del))) ./ 5;
  EI = mux + (bt .* sigmax);
endfunction
