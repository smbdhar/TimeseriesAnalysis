%% SCRIPT FOR FITTING AND PREDICTING GNSS TIME SERIES
% =========================================================================== %
% Author: Sambuddha Dhar (2026-10)
%
% This Octave version of source code is shared in Github: (https://github.com/smbdhar/)
% This code is a refined version of the original Matlab source code shared in
% Zenodo (https://doi.org/10.5281/zenodo.7645102) and
% Github (https://github.com/smbdhar/timeseriesForecast).
%
% Related paper:
% Dhar, S., & Muto, J. (2023). Function model based on nonlinear transient
% rheology of rocks: An analysis of decadal GNSS time series after
% the 2011 Tohoku-oki earthquake. Geophysical Research Letters, 50,
% e2023GL103259. https://doi.org/10.1029/2023GL103259
% =========================================================================== %

clear;clc;

gnssid='940028'; stname={'Miyako'}; % [Training station]
# gnssid='960549'; stname={'Yamoto'}; % [Training station]
# gnssid='940042'; stname={'Hitachi'}; % [Training station]
# gnssid='950193'; stname={'Minase'}; % [Training station]
# gnssid='950151'; stname={'Kanita'}; % [Training station]
# gnssid='950233'; stname={'Ryoutsu2'}; % [Training station]

# gnssid='950172'; stname={'Kesennuma'}; % [Evaluation station]
# gnssid='960552'; stname={'Tennou'}; % [Evaluation station]
# gnssid='950206'; stname={'Tadami'}; % [Evaluation station]
# gnssid='940041'; stname={'Iwaki'}; % [Evaluation station]
# gnssid='970803'; stname={'Yamanobe'}; % [Evaluation station]

gnss=importdata(['./data/rcor',gnssid,'new2.mat']);
gnsssitename={[gnssid, ' ',stname{1},' [NS]'],[gnssid, ' ',stname{1},' [EW]'],[gnssid, ' ',stname{1},' [UD]']}';
% ------------------------------------------------------------------- %

[tobs,dobs,dint]=assemble_gnssdata({gnss});
t_fit=linspace(0,3.9,1424); % Fitting period
t_pred=linspace(0,10,3651); % Prediction period
[tobsfit,dobsfit,tobspred,dobspred]=shuffle_gnssdata(tobs,dobs,t_fit,t_pred);

% Best results presented in Dhar & Muto (2023 GRL)
[tau2,tau1,lambda,beta2]=deal(0.0427527108314662,0.975363685964396,0.00300442746400926,0.968378142612612);
% Performing the predicion algorithm of Dhar & Muto (2023 GRL)
[residual, dcalpred, VRcalpred, AScalpred, scalar_const] = ...
timeseriesregression(tau1, tau2, lambda, beta2, t_pred, tobsfit, dobsfit, tobspred, dobspred);

#### Visualization of time series prediction model (NO SST included) /////////// #
##figure(91); clf; gold = [255 215 0] / 255;
##subplot(3,1,1);  hold on; box on;
##  plot(tobspred(1,:),dobspred(1,:),'k*'); % timeseries during prediction period
##  p2=plot(tobsfit(1,:),dobsfit(1,:),'b*'); % timeseries during fitting period
##  set(p2, 'Color', gold);
##  plot(t_pred,dcalpred(1,:),'r','linewidth',2); % modeled timeseries
##  xlabel('Year after earthquake','fontsize',14);
##  ylabel('North [m]','fontsize',14);
##  title(['Station ID: ',gnssid, '(Only Postseismic displacement)'],'fontsize',18);
##subplot(3,1,2);  hold on; box on;
##  plot(tobspred(2,:),dobspred(2,:),'k*');
##  p2=plot(tobsfit(2,:),dobsfit(2,:),'b*');
##  set(p2, 'Color', gold);
##  plot(t_pred,dcalpred(2,:),'r','linewidth',2);
##  xlabel('Year after earthquake','fontsize',14);
##  ylabel('East [m]','fontsize',14);
##subplot(3,1,3);  hold on; box on;
##  plot(tobspred(3,:),dobspred(3,:),'k*');
##  p2=plot(tobsfit(3,:),dobsfit(3,:),'b*');
##  set(p2, 'Color', gold);
##  plot(t_pred,dcalpred(3,:),'r','linewidth',2);
##  xlabel('Year after earthquake','fontsize',14);
##  ylabel('Up [m]','fontsize',14);

# //// Adding the steady-state linear trend to both obs and model //////////// #
# velocity for each component (rows 1-3 = NS, EW, UD) from the first two epochs
SST_vel = (dint(1:3,2) - dint(1:3,1)) ./ (tobs(1:3,2) - tobs(1:3,1));   # 3x1
SST_vel_NS = SST_vel(1);
SST_vel_EW = SST_vel(2);
SST_vel_UD = SST_vel(3);

# Octave broadcasts the 3x1 velocity down the rows of the 3xN time arrays
dSSTobspred   = tobspred(1:3,:) .* SST_vel;
dSSTobsfit    = tobsfit(1:3,:)  .* SST_vel;
dSSTmodelpred = SST_vel * t_pred(:).';    # 3xN (t_pred as a row)

dSSTobspred_NS = dSSTobspred(1,:);  dSSTobsfit_NS = dSSTobsfit(1,:);  dSSTmodelpred_NS = dSSTmodelpred(1,:);
dSSTobspred_EW = dSSTobspred(2,:);  dSSTobsfit_EW = dSSTobsfit(2,:);  dSSTmodelpred_EW = dSSTmodelpred(2,:);
dSSTobspred_UD = dSSTobspred(3,:);  dSSTobsfit_UD = dSSTobsfit(3,:);  dSSTmodelpred_UD = dSSTmodelpred(3,:);

tobspred_NS = tobspred(1,:);  tobspred_EW = tobspred(2,:);  tobspred_UD = tobspred(3,:);
dobsN_pd = dobspred(1,:) + dSSTobspred_NS;
dobsE_pd = dobspred(2,:) + dSSTobspred_EW;
dobsU_pd = dobspred(3,:) + dSSTobspred_UD;
dcalpredN_pd = dcalpred(1,:) + dSSTmodelpred_NS;
dcalpredE_pd = dcalpred(2,:) + dSSTmodelpred_EW;
dcalpredU_pd = dcalpred(3,:) + dSSTmodelpred_UD;
VRcalpred_NS = VRcalpred(1,:);  VRcalpred_EW = VRcalpred(2,:);  VRcalpred_UD = VRcalpred(3,:);
AScalpred_NS = AScalpred(1,:);  AScalpred_EW = AScalpred(2,:);  AScalpred_UD = AScalpred(3,:);

obsfilter  = @(Dinp) [movmean(Dinp(1:7),1),  movmean(Dinp(8:12),7),  movmean(Dinp(13:end),14)];
obsfilterw = @(Dinp) [movmean(Dinp(1:7),14), movmean(Dinp(8:12),14), movmean(Dinp(13:end),14)];

## ///// Visualization of timeseries prediction ////////////////////////////// #
figure(92); clf; set(gcf, 'name', 'Observations and model timeseries');
gold = [255 215 0] / 255;

# ---- North-south component ----------------------------
subplot(3,1,1); hold on; box on; grid on;
p1=plot(tobspred_NS, dobsN_pd, 'k*');                        # Obs data
p2 = plot(tobsfit(1,:), dobsfit(1,:) + dSSTobsfit_NS, '*');  # Obs data in fitting period
set(p2, 'Color', gold);
##plot(tobspred_NS, obsfilter(Nobspd), 'g');                 # Mean of Obs data
p3=plot(t_pred, dcalpredN_pd, 'r', 'LineWidth', 2);          # Modeled timeseries
p4=plot(t_pred, VRcalpred_NS, 'b');                          # Modeled VR component
p5=plot(t_pred, AScalpred_NS, 'c');                          # Modeled AS component
p6=plot(t_pred, dSSTmodelpred_NS, 'm--');                    # Modeled steady-state component (Vt)
title(gnsssitename{1}, 'FontSize', 18);
ylabel('Displacement (m)', 'FontSize', 14);

# ---- East-West component -----------------------------
subplot(3,1,2); hold on; box on; grid on;
p1=plot(tobspred_EW, dobsE_pd, 'k*');
p2 = plot(tobsfit(2,:), dobsfit(2,:) + dSSTobsfit_EW, '*');
set(p2, 'Color', gold);
##plot(tobspred_EW, obsfilter(Eobspd), 'g');
p3=plot(t_pred, dcalpredE_pd, 'r', 'LineWidth', 2);
p4=plot(t_pred, VRcalpred_EW, 'b');
p5=plot(t_pred, AScalpred_EW, 'c');
p6=plot(t_pred, dSSTmodelpred_EW, 'm--');
title(gnsssitename{2}, 'FontSize', 18);
ylabel('Displacement (m)', 'FontSize', 14);

# ---- Up-Down component ------------------------------
subplot(3,1,3); hold on; box on; grid on;
p1=plot(tobspred_UD, dobsU_pd, 'k*');
p2 = plot(tobsfit(3,:), dobsfit(3,:) + dSSTobsfit_UD, '*');
set(p2, 'Color', gold);
##plot(tobspred_UD, obsfilterw(Uobspd), 'g');
p3=plot(t_pred, dcalpredU_pd, 'r', 'LineWidth', 2);
p4=plot(t_pred, VRcalpred_UD, 'b');
p5=plot(t_pred, AScalpred_UD, 'g');
p6=plot(t_pred, dSSTmodelpred_UD, 'm--');
title(gnsssitename{3}, 'FontSize', 18);
xlabel('Years after earthquake', 'FontSize', 14);
ylabel('Displacement (m)', 'FontSize', 14);

leg=legend([p1, p3, p4, p5, p6], {'Obs', 'Model','VR','AS', 'SST'});
set(leg, 'Location', 'southoutside', 'Orientation', 'horizontal');
