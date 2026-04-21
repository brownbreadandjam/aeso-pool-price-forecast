


##### AESO POOL PRICE PREDICTION #####

##### 1. INIT #####

if(!require(data.table)){install.packages("data.table"); library(data.table)}
if(!require(lubridate)){install.packages("lubridate"); library(lubridate)}
if(!require(dplyr)){install.packages("dplyr"); library(dplyr)}
if(!require(xgboost)){install.packages("xgboost");library(xgboost)}
if(!require(ggplot2)){install.packages("ggplot2");library(ggplot2)}
if(!require(forecast)){install.packages("forecast");library(forecast)}
if(!require(tseries)){install.packages("tseries");library(tseries)}
if(!require(zoo)){install.packages("zoo"); library(zoo)}
if(!require(patchwork)){install.packages("patchwork");library(patchwork)}

setwd("C:/Users/naved/Downloads")
aeso_data <- fread("Pooled_Main.csv")
set.seed(1)




##### 2. TRANSFORM #####

# Parsing Datetime
aeso_data[, Datetime := dmy_hm(Datetime)]
setorder(aeso_data, Datetime)

# Setting price cap 
aeso_data[, price := Price]
aeso_data[price == 999.99, price := NA]

# Histogram to preview the raw prices
ggplot(aeso_data, aes(price)) + geom_histogram(bins = 100, fill="seagreen")


# Data is heavily concentrated around 0-200 dollars with non-finite ranges
# So we must adjust for NA or zero values with shifted price
# And we need log transformation to normalize the prices
aeso_data[, price := na.approx(price, na.rm = FALSE)]
aeso_data[is.na(price), price := Price]
price_shift <- median(aeso_data$price, na.rm = TRUE) * 0.1 + 1
aeso_data[, price_shifted := pmax(price + price_shift, price_shift)]


# Setting price to log format due to heavily tailed data
aeso_data[, log_price := log(price_shifted)]


# Confirming data for time-series elements
ggplot(aeso_data, aes(Datetime, price_shifted)) +
  geom_line(color = "seagreen", linewidth = 0.3)

# Checking for seasonality trends
ts_raw <- ts(aeso_data$price_shifted, frequency = 24*7*30)
plot(decompose(ts_raw))





##### 3. FEATURES #####

# Creating calendar features
aeso_data[, `:=`(hour = hour(Datetime), wday = lubridate::wday(Datetime), month = month(Datetime))]
aeso_data[, `:=`(
  is_weekend = as.integer(wday %in% c(1, 7)),
  hour_f = factor(hour),
  wday_f = factor(wday),
  month_f = factor(month))]


# Creating hourly lags to train the models on testing and forecasting
lag_hours <- c(1, 2, 12, 24, 48, 168)
for (lag_n in lag_hours) {
  aeso_data[, paste0("lag_", lag_n) := shift(price_shifted, lag_n, type = "lag")]}

# Creating Alberta Internal load and Rolling Means 
aeso_data[, `:=`(
  ail_lag_1 = shift(AIL, 1), ail_lag_24 = shift(AIL, 24),
  roll_mean_24 = frollmean(price_shifted, 24),
  roll_sd_24 = frollapply(price_shifted, 24, sd))]
# Ravg30 did not have any significant impact on the results, thus we omitted rolling average lags


# Creating spike features to predict outlier in price shifts
spike_threshold <- quantile(aeso_data$price, 0.9, na.rm = TRUE) + price_shift
aeso_data[, spike_count_24 := frollapply(price_shifted, 24, function(x) sum(x > spike_threshold))]
aeso_data[, is_spike := as.integer(price_shifted > spike_threshold)]

# Calculating differences in spikes to train patterns
aeso_data[, hrs_since_spike := {
  out <- integer(.N)
  counter <- 168L
  for (i in seq_len(.N)) {
    counter <- if (is_spike[i]) 0L else min(counter + 1L, 168L)
    out[i] <- counter}
  out}]


# Creating combined features for XGBoost and ARIMA input
combined_features <- c(
  "is_weekend", "lag_1","lag_2", "lag_12", "lag_24","lag_48","lag_168",
  "ail_lag_1", "ail_lag_24", "roll_mean_24",
  "roll_sd_24", "spike_count_24","hrs_since_spike")

# Back transformation formula to change back log prices to real prices
back_transform <- function(lp){pmax(pmin(exp(lp) - price_shift, 1000), 0.01)}




##### 4. SPLIT #####

# Setting forecast horizon to last 30 days (hourly)
forecast_horizon_hours <- 24 * 30
split_cutoff <- max(aeso_data$Datetime) - hours(forecast_horizon_hours)

# Splitting data into train & test set
model_dt <- na.omit(aeso_data)
train_dt <- model_dt[Datetime <= split_cutoff]
test_dt  <- model_dt[Datetime >  split_cutoff]

# Number of data in each set
nrow(train_dt) ; nrow(test_dt)




##### 5. STATIONARITY #####

# Plotting ACF and PACF
par(mfrow = c(1, 2))
acf(train_dt$log_price, lag.max = 168, main = "ACF log-price")
pacf(train_dt$log_price, lag.max = 168, main = "PACF log-price")
par(mfrow = c(1, 1))

# Signs of recurring pattern may further suggest seasonality

# Checking stationarity with ADF test
print(adf.test(train_dt$log_price))

# p-value seems to be much smaller 0.01, meaning data is stationary and may not need differencing.
# But we will use auto.arima to check whether data require difference or not.





##### 6. MODELS #####

# A) Linear Regression

# This is our baseline model to check how our lags are performing
# After running multiple linear models, we kept the lags & features that had significant impact or were essential for training patterns. 
linear_model <- lm(log_price ~ ., data = aeso_data)

# We will use some of the lags with high significance in the other models
summary(linear_model)

# Using back transformation formula for linear prediction
linear_pred <- back_transform(predict(linear_model, test_dt))




# B) ARIMA

# Using auto.arima to check for difference & seasonality variables
arima_model <- auto.arima(train_dt$log_price, stepwise = FALSE, approximation = FALSE)

# Results show data was differenced indeed, with an AIC value of 48K
# And seasonality variables were detected as we assumed
summary(arima_model)

# Using back transformation formula for ARIMA prediction
arima_pred <- back_transform(as.numeric(forecast(arima_model, h = nrow(test_dt))$mean))




# C) ARIMAX
arimax_features <- setdiff(combined_features,
  c("roll_mean_24", "roll_sd_24", "spike_count_24", "hrs_since_spike"))

arimax_model <- auto.arima(
  train_dt$log_price, xreg = as.matrix(train_dt[, ..arimax_features]),
  stepwise = FALSE, approximation = FALSE)


# ARIMAX has a better AIC of (46K) and lower errors overall than simple ARIMA
summary(arimax_model)

# Using back transformation formula for ARIMAX with external regressors
arimax_pred <- back_transform(as.numeric(forecast(arimax_model, xreg = as.matrix(test_dt[, ..arimax_features]))$mean))



# D) XGBoost

# Creating train and test matrices with the XGB features we created before 
train_matrix <- xgb.DMatrix(as.matrix(train_dt[, ..combined_features]), label = train_dt$log_price)
test_matrix  <- xgb.DMatrix(as.matrix(test_dt[, ..combined_features]),  label = test_dt$log_price)

# Setting XGB parameters in a list as input
xgb_params <- list(
  objective = "reg:squarederror", eta = 0.01, max_depth = 5,
  min_child_weight = 5, subsample = 0.8, colsample_bytree = 0.8, gamma = 0.1)

# Creating XGBoost with all features since this type of model can evaluate precisely with higher number of variables.  
xgb_model <- xgb.train(
  params = xgb_params,  data = train_matrix, nrounds = 2000, early_stopping_rounds = 50,
  watchlist = list(train = train_matrix, test = test_matrix), verbose = 0)


# Using back transformation from the test matrix for XGBoost
xgb_pred_log <- predict(xgb_model, test_matrix)
xgb_pred <- back_transform(xgb_pred_log)





##### 7. PERFORMANCE #####

# Creating performance metric for comparison
perf_metric <- function(actual, predicted, model_name, train_actual, train_predicted) {
  idx_te <- !is.na(actual) & !is.na(predicted) & is.finite(actual) & is.finite(predicted)
  idx_tr <- !is.na(train_actual) & !is.na(train_predicted) & is.finite(train_actual) & is.finite(train_predicted)
  
  # Setting naive benchmark to compare against test and train set
  test_mae  <- mean(abs(actual[idx_te] - predicted[idx_te]))
  train_mae <- mean(abs(train_actual[idx_tr] - train_predicted[idx_tr]))
  naive_mae <- mean(abs(diff(train_actual[idx_tr])))
  
  # Calculating RMSE, MAE,and accuracy from actual train and test set index 
  data.table(Model = model_name,
             Train_RMSE = round(sqrt(mean((train_actual[idx_tr] - train_predicted[idx_tr])^2)), 3),
             Test_RMSE = round(sqrt(mean((actual[idx_te] - predicted[idx_te])^2)), 3),
             Train_MAE = round(train_mae, 3), Test_MAE = round(test_mae, 3), MASE = round(test_mae / naive_mae, 3),
             Accuracy = paste0(round((1 - test_mae / mean(actual[idx_te])) * 100, 1), "%"))}

# Converting those log prices to real prices
train_xgb_pred    <- back_transform(predict(xgb_model, train_matrix))
train_linear_pred <- back_transform(predict(linear_model, train_dt))
train_arima_pred  <- back_transform(as.numeric(fitted(arima_model)))
train_arimax_pred <- back_transform(as.numeric(fitted(arimax_model)))


# Turning comparison into a table
results <- rbind(
  perf_metric(test_dt$price, linear_pred, "Linear", train_dt$price, train_linear_pred),
  perf_metric(test_dt$price, arima_pred, "ARIMA", train_dt$price, train_arima_pred),
  perf_metric(test_dt$price, arimax_pred, "ARIMAX", train_dt$price, train_arimax_pred),
  perf_metric(test_dt$price, xgb_pred, "XGBoost", train_dt$price, train_xgb_pred))

setorder(results, Test_MAE)
print(results)

# XGBoost is clearly the best model across all measures, especially with 77.5% acccuracy on real-life current data.
# It implies the XG model is able to predict within just below $5, as depicted by the Test MAE (4.939).
# So, if pool prices were $30 on 28 February 2026, the model should predict within 25-35 dollars.

# Linear is the second best, hinting how traditional tests work better with time-series, even with all the features.
# ARIMAX outperformed only ARIMA, presumably due to the lag inputs (selectively chosen after iterations).
# But only XGBoost provides an accuracy and measure of error worth pursuing for test and future predictions.
# Yet the other models (especially linear) were crucial in creating and tuning the lags and features.



# Plotting Actual vs XGBoost — Feb 02 to Mar 02, 2026
test_plot_dt <- data.table(Datetime = test_dt$Datetime, Actual = test_dt$price, XGBoost = xgb_pred)

ggplot(test_plot_dt, aes(x = Datetime)) +
  geom_ribbon(aes(ymin = 0, ymax = Actual), fill = "seagreen", alpha = 0.4) +
  geom_line(aes(y = XGBoost), color = "navy", linewidth = 0.7) +
  labs(title = "Actual vs XGBoost — Test Period")


# Plotting XGBoost residuals during test period
test_plot_dt[, residual := Actual - XGBoost]

ggplot(test_plot_dt, aes(x = Datetime, y = residual)) +
  geom_line(color = "navy", linewidth = 0.5, alpha = 0.8) +
  labs(title = "XGBoost Residuals - Test Period")





##### 8. FORECASTING #####

# We wanted to forecast next hour, next day, and next week pool prices
# But doing so without any other future variables limited our options
# Thus we opted to use the actual mean in a recursive loop to predict a stable baseline

forecast_horizon <- 24 * 7
last_datetime <- max(model_dt$Datetime)

# Creating future datetime grid
future_dt <- data.table(Datetime = seq(from = last_datetime + hours(1), by = "hour", length.out = forecast_horizon))


# Creating future calendar features
future_dt[, hour := hour(Datetime)]
future_dt[, wday := lubridate::wday(Datetime)]
future_dt[, month := month(Datetime)]
future_dt[, `:=`(is_weekend = as.integer(wday %in% c(1, 7)))]


# Using price shifted model to create a baseline mean for future prices
recent_mean <- mean(tail(model_dt$price_shifted, 168))

future_dt[, `:=`( lag_1 = recent_mean, lag_2 = recent_mean, lag_3 = recent_mean, lag_6 = recent_mean,
  lag_12 = recent_mean, lag_24 = tail(model_dt$price_shifted, 1), lag_48 = tail(model_dt$price_shifted, 1),
  lag_168 = recent_mean, ail_lag_1 = tail(model_dt$AIL, 1), ail_lag_24 = tail(model_dt$AIL, 24)[1],
  roll_mean_24 = recent_mean, roll_mean_168 = recent_mean, roll_sd_24 = sd(tail(model_dt$price_shifted, 24)),
  spike_count_24 = 0, hrs_since_spike = 24L)]

history_price <- model_dt$price_shifted
history_ail <- model_dt$AIL

# Creating a recursive loop for price and AIL to predict future pool prices

for (i in seq_len(forecast_horizon)) {
  n_hist <- length(history_price)
  
  # Inputting the loop within the future features we just created
  future_dt[i, `:=`(
    lag_1 = pmax(history_price[n_hist], price_shift),
    lag_2 = pmax(ifelse(n_hist >= 2, history_price[n_hist-1], history_price[n_hist]), price_shift),
    lag_6 = pmax(ifelse(n_hist >= 6, history_price[n_hist-5], history_price[n_hist]), price_shift),
    lag_12 = pmax(ifelse(n_hist >= 12, history_price[n_hist-11], history_price[n_hist]), price_shift),
    lag_24 = pmax(ifelse(n_hist >= 24, history_price[n_hist-23], tail(model_dt$price_shifted, 1)), price_shift),
    lag_48 = pmax(ifelse(n_hist >= 48, history_price[n_hist-47], recent_mean), price_shift),
    lag_168 = pmax(ifelse(n_hist >= 168, history_price[n_hist-167], recent_mean), price_shift),
    ail_lag_1 = tail(history_ail, 1),
    ail_lag_24 = ifelse(n_hist >= 24, history_ail[n_hist-23], tail(history_ail, 24)[1]),
    roll_mean_24 = mean(tail(history_price, min(24, n_hist))),
    roll_mean_168 = mean(tail(history_price, min(168, n_hist))),
    roll_sd_24 = sd(tail(history_price, min(24, n_hist))),
    spike_count_24 = sum(tail(history_price, min(24, n_hist)) > spike_threshold),
    hrs_since_spike = min(168L, 24L + ifelse(sum(tail(history_price, min(168, n_hist)) > spike_threshold) > 0, 0L, 24L)))]
  
  # Predicting future prices with log and features
  future_features <- as.matrix(future_dt[i, ..combined_features])
  future_log_price <- predict(xgb_model, xgb.DMatrix(future_features))
  
  # Adjusting future prices for our shifted price measures
  future_price_shifted_raw <- exp(future_log_price)
  future_price_shifted <- 0.7 * future_price_shifted_raw + 0.3 * future_dt[i, lag_24]
  future_price_shifted <- pmax(pmin(future_price_shifted, recent_mean * 10), price_shift) 
  
  # Converting future prices back from log to raw
  future_price <- back_transform(log(future_price_shifted))
  
  # Binding the future prices together
  future_dt[i, `:=`(log_price = future_log_price,
    price_shifted = future_price_shifted, Forecast = future_price)]
  
  history_price <- c(history_price, future_price_shifted)}


# Summarizing the forecast values
forecast_table_7d <- future_dt[, .(Datetime, Forecast, hour, wday, lag_24, spike_count_24, price_shifted)]
print(head(forecast_table_7d, 24))




# Plotting the forecast with Actual vs XGboost 
plot_data <- rbind(
  data.table(Datetime = tail(model_dt$Datetime, 168), Price = model_dt$price[tail(1:nrow(model_dt), 168)], Series = "Actual"),
  data.table(Datetime = future_dt$Datetime, Price = future_dt$Forecast, Series = "XGBoost"))

forecast_plot <- ggplot(plot_data, aes(Datetime, Price, color = Series)) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "seagreen", "XGBoost" = "navy")) +
  labs(title = "Last Week Actual vs Next Week Forecast")

print(forecast_plot)


# The plot shows a smoothed tendency due to the recursive loop from the actual mean.
# Since, the model is predicting values close to usual prices, we can iterate for improvements.
# This gives us a wide range of opportunity to try out different sorts of parameter tuning or combine with other models.

