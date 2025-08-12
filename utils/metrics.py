import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


import numpy as np
from sklearn.metrics import mean_absolute_error


def smape(y_true, y_pred):
    """
    计算对称平均绝对百分比误差 (SMAPE)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    smape_val = np.mean(diff / denominator) * 100  # 百分比形式
    return smape_val


def mase(y_pred, y_true, seasonal_period=1):
    """
    计算平均绝对比例误差 (MASE)
    """
    # 计算预测误差
    forecast_error = mean_absolute_error(y_true, y_pred)

    # 计算训练数据中季节naive的误差
    # 注意：实际实现需要访问训练数据，这里简化处理
    # 实际应使用训练数据计算seasonal_naive_error
    seasonal_naive_error = 1.0  # 此处应为实际计算值

    mase_val = forecast_error / seasonal_naive_error
    return mase_val


def owa(y_pred, y_true, train_data_path):
    """
    计算整体加权平均 (OWA)
    """
    # 简化实现 - 实际需要基准模型预测
    naive_forecast = np.roll(y_true, 1)  # 简单使用前一个值作为预测
    naive_error = mean_absolute_error(y_true, naive_forecast)
    model_error = mean_absolute_error(y_true, y_pred)

    owa_val = (model_error / naive_error) * 0.5  # 简化计算公式
    return owa_val
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
