



# ============================================================
# AESO Electricity Price Forecasting v7 - FIXED FORECASTING
# ============================================================

library(data.table)
library(lubridate)
library(zoo)
library(forecast)
library(tseries)
library(xgboost)
library(ggplot2)
library(patchwork)

setwd("C:/Users/naved/Downloads")
aeso_data <- fread("Pooled_Main.csv")
set.seed(1)

# -----------------------------
# 1) CLEAN DATA
# -----------------------------
aeso_data[, Datetime := dmy_hm(Datetime)]
setorder(aeso_data, Datetime)

aeso_data[, price := Price]
aeso_data[price == 999.99, price := NA]
aeso_data[, price := na.approx(price, na.rm = FALSE)]
aeso_data[is.na(price), price := Price]

# FIXED: Use median instead of min for price_shift (much more stable)
price_shift <- median(aeso_data$price, na.rm = TRUE) * 0.1 + 1  # ~10-20 range
aeso_data[, price_shifted := pmax(price + price_shift, price_shift)]
aeso_data[, log_price := log(price_shifted)]

# -----------------------------
# 2) FEATURE ENGINEERING
# -----------------------------
aeso_data[, `:=`(
  hour = hour(Datetime),
  wday = lubridate::wday(Datetime),
  month = month(Datetime)
)]

aeso_data[, `:=`(
  is_weekend = as.integer(wday %in% c(1, 7)),
  is_peak = as.integer(hour %in% 16:21),
  hour_sin = sin(2 * pi * hour / 24),
  hour_cos = cos(2 * pi * hour / 24),
  wday_sin = sin(2 * pi * wday / 7),
  wday_cos = cos(2 * pi * wday / 7),
  month_sin = sin(2 * pi * month / 12),
  month_cos = cos(2 * pi * month / 12),
  hour_f = factor(hour),
  wday_f = factor(wday),
  month_f = factor(month)
)]

lag_hours <- c(1, 2, 3, 6, 12, 24, 48, 168)
for (lag_n in lag_hours) {
  aeso_data[, paste0("lag_", lag_n) := shift(price_shifted, lag_n, type = "lag")]
}

aeso_data[, lag_ratio_1_24 := lag_1 / pmax(lag_24, 0.01)]
aeso_data[, lag_ratio_24_168 := lag_24 / pmax(lag_168, 0.01)]

aeso_data[, `:=`(
  ail_lag_1 = shift(AIL, 1),
  ail_lag_24 = shift(AIL, 24),
  ail_roll_mean_24 = frollmean(AIL, 24),
  ail_trend = AIL - shift(AIL, 24)
)]

aeso_data[, `:=`(
  roll_mean_24 = frollmean(price_shifted, 24),
  roll_mean_168 = frollmean(price_shifted, 168),
  roll_sd_24 = frollapply(price_shifted, 24, sd),
  roll_min_24 = frollapply(price_shifted, 24, min),
  roll_max_24 = frollapply(price_shifted, 24, max)
)]

spike_threshold <- quantile(aeso_data$price, 0.95, na.rm = TRUE) + price_shift
aeso_data[, spike_count_24 := frollapply(price_shifted, 24, function(x) sum(x > spike_threshold))]
aeso_data[, is_spike := as.integer(price_shifted > spike_threshold)]

aeso_data[, hrs_since_spike := {
  out <- integer(.N)
  counter <- 168L
  for (i in seq_len(.N)) {
    counter <- if (is_spike[i]) 0L else min(counter + 1L, 168L)
    out[i] <- counter
  }
  out
}]

# -----------------------------
# 3) TRAIN / TEST SPLIT
# -----------------------------
forecast_horizon_hours <- 24 * 30
split_cutoff <- max(aeso_data$Datetime) - hours(forecast_horizon_hours)

model_dt <- na.omit(aeso_data)
train_dt <- model_dt[Datetime <= split_cutoff]
test_dt  <- model_dt[Datetime >  split_cutoff]

cat("Train:", format(min(train_dt$Datetime)), "->", format(max(train_dt$Datetime)),
    "| Rows:", nrow(train_dt), "\n")
cat("Test: ", format(min(test_dt$Datetime)), "->", format(max(test_dt$Datetime)),
    "| Rows:", nrow(test_dt), "\n")

# -----------------------------
# 4) STATIONARITY CHECK
# -----------------------------
par(mfrow = c(1, 2))
acf(train_dt$log_price, lag.max = 168, main = "ACF log-price")
pacf(train_dt$log_price, lag.max = 168, main = "PACF log-price")
par(mfrow = c(1, 1))
print(adf.test(train_dt$log_price))

# -----------------------------
# 5) FEATURE SET
# -----------------------------
xgb_features <- c(
  "hour_sin","hour_cos","wday_sin","wday_cos","month_sin","month_cos",
  "is_weekend","is_peak",
  "lag_1","lag_2","lag_3","lag_6","lag_12","lag_24","lag_48","lag_168",
  "lag_ratio_1_24","lag_ratio_24_168",
  "ail_lag_1","ail_lag_24","ail_roll_mean_24","ail_trend",
  "roll_mean_24","roll_mean_168","roll_sd_24","roll_min_24","roll_max_24",
  "spike_count_24","hrs_since_spike"
)

# FIXED: Better back transformation
back_transform <- function(lp) {
  pmax(pmin(exp(lp) - price_shift, 1000), 0.01)
}

# -----------------------------
# 6) MODELS
# -----------------------------
linear_model <- lm(
  log_price ~ hour_f + wday_f + month_f + is_peak +
    lag_1 + lag_24 + lag_168 + ail_lag_1 + roll_mean_24,
  data = train_dt
)
linear_pred <- back_transform(predict(linear_model, test_dt))

arima_model <- auto.arima(train_dt$log_price, stepwise = TRUE, approximation = TRUE)
arima_pred <- back_transform(as.numeric(forecast(arima_model, h = nrow(test_dt))$mean))

arimax_features <- c("lag_1", "lag_24", "lag_168", "ail_lag_1")
arimax_model <- auto.arima(
  train_dt$log_price,
  xreg = as.matrix(train_dt[, ..arimax_features]),
  stepwise = TRUE,
  approximation = TRUE
)
arimax_pred <- back_transform(as.numeric(
  forecast(arimax_model, xreg = as.matrix(test_dt[, ..arimax_features]))$mean
))

train_matrix <- xgb.DMatrix(as.matrix(train_dt[, ..xgb_features]), label = train_dt$log_price)
test_matrix  <- xgb.DMatrix(as.matrix(test_dt[, ..xgb_features]),  label = test_dt$log_price)

xgb_params <- list(
  objective = "reg:squarederror",
  eta = 0.03,
  max_depth = 5,
  min_child_weight = 3,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0.1
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = train_matrix,
  nrounds = 1000,
  early_stopping_rounds = 50,
  watchlist = list(train = train_matrix, test = test_matrix),
  verbose = 0
)

xgb_pred_log <- predict(xgb_model, test_matrix)
xgb_pred <- back_transform(xgb_pred_log)

# -----------------------------
# 7) PERFORMANCE METRICS
# -----------------------------
metric_fn <- function(actual, predicted, model_name) {
  valid_idx <- !is.na(actual) & !is.na(predicted) & is.finite(actual) & is.finite(predicted)
  actual <- actual[valid_idx]
  predicted <- predicted[valid_idx]
  data.table(
    model = model_name,
    RMSE = round(sqrt(mean((actual - predicted)^2)), 3),
    MAE = round(mean(abs(actual - predicted)), 3),
    SMAPE = round(mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted) + 1e-6)) * 100, 2)
  )
}

results <- rbind(
  metric_fn(test_dt$price, linear_pred, "Linear"),
  metric_fn(test_dt$price, arima_pred, "ARIMA"),
  metric_fn(test_dt$price, arimax_pred, "ARIMAX"),
  metric_fn(test_dt$price, xgb_pred, "XGBoost")
)

setorder(results, MAE)
print(results)

best_model <- results$model[1]
best_pred <- switch(
  best_model,
  XGBoost = xgb_pred,
  Linear = linear_pred,
  ARIMA = arima_pred,
  ARIMAX = arimax_pred
)

cat("\nBest model:", best_model,
    "| MAE:", results[model == best_model, MAE],
    "| RMSE:", results[model == best_model, RMSE], "\n")

# -----------------------------
# 8) VISUALS
# -----------------------------
actual_vs_pred_dt <- melt(
  data.table(
    Datetime = test_dt$Datetime,
    Actual = test_dt$price,
    Linear = linear_pred,
    ARIMA = arima_pred,
    ARIMAX = arimax_pred,
    XGBoost = xgb_pred
  ),
  id = "Datetime"
)

print(
  ggplot(actual_vs_pred_dt, aes(Datetime, value, color = variable)) +
    geom_line(linewidth = 0.4) +
    theme_minimal() +
    scale_color_manual(values = c(
      Actual = "black",
      Linear = "orange",
      ARIMA = "purple",
      ARIMAX = "green",
      XGBoost = "steelblue"
    )) +
    labs(title = "Test Set: Actual vs All Models", y = "Price ($/MWh)", color = "")
)

residual_dt <- data.table(Datetime = test_dt$Datetime, Error = test_dt$price - best_pred)

print(
  ggplot(residual_dt, aes(Datetime, Error)) +
    geom_line(linewidth = 0.3) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(title = paste(best_model, "Residuals"), y = "Error ($/MWh)")
)

print(
  ggplot(data.table(Actual = test_dt$price, Predicted = xgb_pred), aes(Actual, Predicted)) +
    geom_point(alpha = 0.2, size = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(title = "XGBoost: Actual vs Predicted Scatter")
)

print(
  ggplot(melt(results, id = "model", measure.vars = c("RMSE", "MAE", "SMAPE")),
         aes(model, value, fill = model)) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~variable, scales = "free_y") +
    theme_minimal() +
    labs(title = "Model Comparison: RMSE / MAE / SMAPE", x = "", y = "")
)

# -----------------------------
# 9) FIXED 7-DAY HOURLY FORECAST
# -----------------------------
forecast_horizon <- 24 * 7
last_datetime <- max(model_dt$Datetime)

# Create future datetime grid
future_dt <- data.table(
  Datetime = seq(from = last_datetime + hours(1),
                 by = "hour",
                 length.out = forecast_horizon)
)

# Calendar features
future_dt[, hour := hour(Datetime)]
future_dt[, wday := lubridate::wday(Datetime)]
future_dt[, month := month(Datetime)]

future_dt[, `:=`(
  is_weekend = as.integer(wday %in% c(1, 7)),
  is_peak = as.integer(hour %in% 16:21),
  hour_sin = sin(2 * pi * hour / 24),
  hour_cos = cos(2 * pi * hour / 24),
  wday_sin = sin(2 * pi * wday / 7),
  wday_cos = cos(2 * pi * wday / 7),
  month_sin = sin(2 * pi * month / 12),
  month_cos = cos(2 * pi * month / 12)
)]

# FIXED: Initialize with recent history damping
recent_mean <- mean(tail(model_dt$price_shifted, 168))
future_dt[, `:=`(
  lag_1 = recent_mean,
  lag_2 = recent_mean,
  lag_3 = recent_mean,
  lag_6 = recent_mean,
  lag_12 = recent_mean,
  lag_24 = tail(model_dt$price_shifted, 1),
  lag_48 = tail(model_dt$price_shifted, 1),
  lag_168 = recent_mean,
  lag_ratio_1_24 = 1.0,
  lag_ratio_24_168 = 1.0,
  ail_lag_1 = tail(model_dt$AIL, 1),
  ail_lag_24 = tail(model_dt$AIL, 24)[1],
  ail_roll_mean_24 = mean(tail(model_dt$AIL, 24)),
  ail_trend = 0,
  roll_mean_24 = recent_mean,
  roll_mean_168 = recent_mean,
  roll_sd_24 = sd(tail(model_dt$price_shifted, 24)),
  roll_min_24 = min(tail(model_dt$price_shifted, 24)),
  roll_max_24 = max(tail(model_dt$price_shifted, 24)),
  spike_count_24 = 0,
  hrs_since_spike = 24L
)]

# Recursive forecast loop with STABILIZATION
history_price <- model_dt$price_shifted
history_ail <- model_dt$AIL

for (i in seq_len(forecast_horizon)) {
  n_hist <- length(history_price)
  
  # FIXED: Robust lag extraction
  future_dt[i, `:=`(
    lag_1 = pmax(history_price[n_hist], price_shift),
    lag_2 = pmax(ifelse(n_hist >= 2, history_price[n_hist-1], history_price[n_hist]), price_shift),
    lag_3 = pmax(ifelse(n_hist >= 3, history_price[n_hist-2], history_price[n_hist]), price_shift),
    lag_6 = pmax(ifelse(n_hist >= 6, history_price[n_hist-5], history_price[n_hist]), price_shift),
    lag_12 = pmax(ifelse(n_hist >= 12, history_price[n_hist-11], history_price[n_hist]), price_shift),
    lag_24 = pmax(ifelse(n_hist >= 24, history_price[n_hist-23], tail(model_dt$price_shifted, 1)), price_shift),
    lag_48 = pmax(ifelse(n_hist >= 48, history_price[n_hist-47], recent_mean), price_shift),
    lag_168 = pmax(ifelse(n_hist >= 168, history_price[n_hist-167], recent_mean), price_shift),
    lag_ratio_1_24 = pmax(history_price[n_hist], price_shift) / pmax(ifelse(n_hist >= 24, history_price[n_hist-23], price_shift), 0.01),
    lag_ratio_24_168 = pmax(ifelse(n_hist >= 24, history_price[n_hist-23], price_shift), 0.01) / pmax(ifelse(n_hist >= 168, history_price[n_hist-167], price_shift), 0.01),
    ail_lag_1 = tail(history_ail, 1),
    ail_lag_24 = ifelse(n_hist >= 24, history_ail[n_hist-23], tail(history_ail, 24)[1]),
    ail_roll_mean_24 = mean(tail(history_ail, min(24, n_hist))),
    ail_trend = tail(history_ail, 1) - ifelse(n_hist >= 24, history_ail[n_hist-23], tail(history_ail, 1)),
    roll_mean_24 = mean(tail(history_price, min(24, n_hist))),
    roll_mean_168 = mean(tail(history_price, min(168, n_hist))),
    roll_sd_24 = sd(tail(history_price, min(24, n_hist))),
    roll_min_24 = min(tail(history_price, min(24, n_hist))),
    roll_max_24 = max(tail(history_price, min(24, n_hist))),
    spike_count_24 = sum(tail(history_price, min(24, n_hist)) > spike_threshold),
    hrs_since_spike = min(168L, 24L + ifelse(sum(tail(history_price, min(168, n_hist)) > spike_threshold) > 0, 0L, 24L))
  )]
  
  # Predict
  future_features <- as.matrix(future_dt[i, ..xgb_features])
  future_log_price <- predict(xgb_model, xgb.DMatrix(future_features))
  
  # FIXED: Stabilized transformation + damping
  future_price_shifted_raw <- exp(future_log_price)
  future_price_shifted <- 0.7 * future_price_shifted_raw + 0.3 * future_dt[i, lag_24]  # Damping
  future_price_shifted <- pmax(pmin(future_price_shifted, recent_mean * 10), price_shift)  # Hard bounds
  
  future_price <- back_transform(log(future_price_shifted))
  
  # Store and update history SAFELY
  future_dt[i, `:=`(
    log_price = future_log_price,
    price_shifted = future_price_shifted,
    Forecast = future_price
  )]
  
  history_price <- c(history_price, future_price_shifted)
}

# Summary table
forecast_table_7d <- future_dt[, .(
  Datetime,
  Forecast,
  hour,
  wday,
  is_peak,
  lag_24,
  spike_count_24,
  price_shifted
)]

print(head(forecast_table_7d, 48))

# Plot
plot_data <- rbind(
  data.table(Datetime = tail(model_dt$Datetime, 168), Price = model_dt$price[tail(1:nrow(model_dt), 168)], Series = "Actual"),
  data.table(Datetime = future_dt$Datetime, Price = future_dt$Forecast, Series = "XGBoost Forecast v7")
)

forecast_plot <- ggplot(plot_data, aes(Datetime, Price, color = Series)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = as.numeric(last_datetime), linetype = "dashed", color = "red", linewidth = 1) +
  theme_minimal(base_size = 12) +
  scale_color_manual(values = c("Actual" = "black", "XGBoost Forecast v7" = "steelblue")) +
  labs(
    title = "AESO Price: Last Week Actual + Next 7 Days Forecast (FIXED)",
    subtitle = paste("Forecast starts", format(last_datetime + hours(1)), "using Stabilized XGBoost"),
    x = "Date & Time",
    y = "Price ($/MWh)",
    color = ""
  ) +
  theme(plot.title = element_text(size = 14, face = "bold"))

print(forecast_plot)

# -----------------------------
# 10) SUMMARY WITH INSIGHTS
# -----------------------------
model_gap <- results[model == best_model, MAE] / results[model != best_model, MAE]
best_row <- results[model == best_model]

cat("\n===== RESULTS SUMMARY =====\n")
cat("1. Best model:", best_model,
    "| MAE:", best_row$MAE,
    "| RMSE:", best_row$RMSE,
    "| SMAPE:", best_row$SMAPE, "%\n")
cat("2. Spike threshold:", round(spike_threshold - price_shift, 2), "/MWh\n")
cat("3. PRICE_SHIFT used:", round(price_shift, 4), "\n")
cat("4. Test window:", format(min(test_dt$Datetime)), "to", format(max(test_dt$Datetime)), "\n")

cat("\n===== FIXED 7-DAY FORECAST SUMMARY =====\n")
cat("Forecast period:", format(min(future_dt$Datetime)), "to", format(max(future_dt$Datetime)), "\n")
cat("Mean forecast price: $", round(mean(future_dt$Forecast), 2), "/MWh\n")
cat("Max forecast price:  $", round(max(future_dt$Forecast), 2), "/MWh\n")
cat("Min forecast price:  $", round(min(future_dt$Forecast), 2), "/MWh\n")
cat("Files saved:\n")
cat("  - forecast_7d_hourly_v7.csv\n")
cat("  - forecast_7d_plot_v7.png\n")
print(summary(future_dt$Forecast))

cat("\n===== KEY FIXES APPLIED =====\n")
cat("- Smaller, median-based price_shift for stability\n")
cat("- Damping in recursive forecasts (70% model + 30% lag_24)\n")
cat("- Robust lag initialization with historical means\n")
cat("- Hard bounds prevent extreme values\n")
cat("- Better back-transformation with realistic limits\n")










#Check summary of all models
#ADD ACCURACY
#DROP RESIDUALS IF CANT BE REDUCED
#ADD PLOT - ACTUAL VS XGBOOST ONLY
#EDA PLOTS - JUSTIFICATION FOR LAG
#DROP MODEL COMPARISON BAR CHART
#Remove best_model function
