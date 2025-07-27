% MATLAB script for Time Series Forecasting using ARIMA
clear;close all;clc;
%% 1.Load and Prepare Data
opts=detectImportOptions('Download Data - STOCK_US_XNAS_ACGL.csv');
variableNames=opts.VariableNames;
dateColumnIndex=find(strcmp(variableNames,'Date'));
if~isempty(dateColumnIndex)
    opts.VariableTypes{dateColumnIndex}='datetime';
else
    warning('Date column not found.Check names.');
end
opts.DataLines=[2,Inf];
stock_data=readtable('Download Data - STOCK_US_XNAS_ACGL.csv',opts);
dates=stock_data.Date;
df_close=stock_data.Close;
df_close(isnan(df_close))=0;
%% 2.Visualize the Data
figure;
plot(dates,df_close);
grid on;xlabel('Date');ylabel('Close Prices');
title('ARCH CAPITAL GROUP Closing Price');
legend('Close Price');
figure;
ksdensity(df_close);
title('Distribution of Closing Price (KDE)');
xlabel('Close Price');ylabel('Density');grid on;
%% 3.Test for Stationarity using Augmented Dickey-Fuller Test
rolmean=movmean(df_close,12);
rolstd=movstd(df_close,12);
figure;
hold on;
plot(dates,df_close,'b','DisplayName','Original');
plot(dates,rolmean,'r','LineWidth',1.5,'DisplayName','Rolling Mean');
plot(dates,rolstd,'k','LineWidth',1.5,'DisplayName','Rolling Std');
hold off;
legend('show');title('Rolling Mean and Standard Deviation');
xlabel('Date');ylabel('Price');grid on;
[h,pValue,stat,cValue]=adftest(df_close);
disp("Results of Dickey-Fuller Test:");
fprintf('Test Statistic: %f\n',stat);
fprintf('p-value: %f\n',pValue);
fprintf('Critical Value (1%%): %f\n',cValue(1));
fprintf('Critical Value (5%%): %f\n',cValue(2));
fprintf('Critical Value (10%%): %f\n',cValue(3));
if h==0
    disp('Result:Series is non-stationary.');
else
    disp('Result:Series is stationary.');
end
%% 4.Decompose and Transform Data
df_log=log(df_close);
df_log(isinf(df_log))=nan;
df_log=fillmissing(df_log,'previous');
figure;
hold on;
plot(dates,movmean(df_log,12),'r','DisplayName','Mean');
plot(dates,movstd(df_log,12),'k','DisplayName','Standard Deviation');
hold off;
title('Moving Average of Log-Transformed Price');
xlabel('Date');ylabel('Log Price');legend('show');grid on;
%% 5.Split Data into Training and Testing Sets
split_point=floor(0.9*length(df_log));
train_data=df_log(1:split_point);
test_data=df_log(split_point+1:end);
test_dates=dates(split_point+1:end);
figure;
hold on;
plot(dates(1:split_point),train_data,'g','DisplayName','Train Data');
plot(test_dates,test_data,'b','DisplayName','Test Data');
hold off;
grid on;xlabel('Dates');ylabel('Log Closing Prices');
title('Train and Test Data Split');legend('show');
%% 6.Find Optimal ARIMA Parameters (Auto ARIMA equivalent)
disp('Finding best ARIMA(p,d,q) model...');
best_aic=inf;
best_p=0;
best_q=0;
d=1;
for p=0:3
    for q=0:3
        model_spec=arima(p,d,q);
        try
            [~,~,logL]=estimate(model_spec,train_data,'Display','off');
            num_params=p+q+1;
            [aic,~]=aicbic(logL,num_params,length(train_data));
            if aic<best_aic
                best_aic=aic;
                best_p=p;
                best_q=q;
            end
        catch ME
            continue;
        end
    end
end
fprintf('Best model:ARIMA(%d,%d,%d) with AIC = %f\n',best_p,d,best_q,best_aic);
%% 7.Build and Fit Final ARIMA Model
p=1;d=1;q=2;
final_model=arima(p,d,q);
disp('Fitting final ARIMA(1,1,2) model...');
EstMdl=estimate(final_model,train_data);
disp(EstMdl);
%% 8.Forecast Stock Prices
num_forecast_steps=length(test_data);
[fc,fc_se]=forecast(EstMdl,num_forecast_steps,'Y0',train_data);
conf_alpha=0.05;
z_score=norminv(1-conf_alpha/2);
conf_interval=z_score*fc_se;
upper_bound=fc+conf_interval;
lower_bound=fc-conf_interval;
%% 9.Visualize the Forecast vs Actual Data
figure;
hold on;
fill([test_dates;flipud(test_dates)],[lower_bound;flipud(upper_bound)],'k',...
    'FaceAlpha',0.10,'EdgeColor','none','DisplayName','95% Confidence Interval');
plot(dates(1:split_point),train_data,'b','LineWidth',1,'DisplayName','Training Data');
plot(test_dates,test_data,'Color',[0.8500 0.3250 0.0980],'LineWidth',1.5,'DisplayName','Actual Stock Price');
plot(test_dates,fc,'Color',[0.4660 0.6740 0.1880],'LineWidth',1.5,'DisplayName','Predicted Stock Price');
hold off;
title('ARCH CAPITAL GROUP Stock Price Prediction (Log Scale)');
xlabel('Time');ylabel('Log Stock Price');
legend('Location','northwest','FontSize',8);grid on;axis tight;
%% 10.Report Performance Metrics
mse=immse(test_data,fc);
mae=mae(test_data,fc);
rmse=sqrt(mse);
mape=mean(abs((fc-test_data)./test_data));
disp('Performance Metrics (on log-transformed data):');
fprintf('MSE: %f\n',mse);
fprintf('MAE: %f\n',mae);
fprintf('RMSE: %f\n',rmse);
fprintf('MAPE: %f\n',mape);
%% 11.Residual Analysis
disp('Performing Residual Analysis...');
residuals=infer(EstMdl,train_data);
figure;
subplot(2,1,1);
autocorr(residuals);
title('ACF of Residuals');
subplot(2,1,2);
parcorr(residuals);
title('PACF of Residuals');
lags=1:20;
[h_lb,p_lb,~,~]=lbqtest(residuals,lags);
fprintf('\nLjung-Box Q-Test Results (for residuals):\n');
for i=1:length(lags)
    if h_lb(i)==0
        fprintf('At lag %d:White noise (p-value=%f).\n',lags(i),p_lb(i));
    else
        fprintf('At lag %d:Not white noise (p-value=%f).\n',lags(i),p_lb(i));
    end
end