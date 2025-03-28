# services/epoch_training_service.py

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def train_sgd_regressor_with_epochs(
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 5,
        batch_size: int = 32
):
    """
    Обучаем линейную регрессию методом SGD по эпохам:
     - Инициализируем SGDRegressor
     - На каждой эпохе случайно перемешиваем данные,
       делим на батчи, для каждого батча вызываем partial_fit
     - Отслеживаем средний loss за эпоху.

    По окончании вычисляем MSE, R2 на всей выборке (X,y).

    Возвращаем словарь:
      - model: обученный SGDRegressor
      - scaler_X, scaler_y: StandardScaler для входов и целевых
      - epoch_losses: список средних значений loss за эпоху
      - final_mse: MSE на всей выборке
      - final_r2: R^2 на всей выборке
    """

    # Инициализируем модель
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        max_iter=1,  # мы управляем эпохами вручную
        learning_rate="constant",
        eta0=0.0001  # начальный шаг
    )

    # Масштабирование
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    y_reshaped = y.values.reshape(-1, 1)

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_reshaped).ravel()

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    epoch_losses = []

    for epoch in range(epochs):
        np.random.shuffle(indices)
        X_scaled = X_scaled[indices]
        y_scaled = y_scaled[indices]

        num_batches = int(np.ceil(n_samples / batch_size))
        losses_in_epoch = []

        for batch_i in range(num_batches):
            start_i = batch_i * batch_size
            end_i = min(start_i + batch_size, n_samples)

            X_batch = X_scaled[start_i:end_i]
            y_batch = y_scaled[start_i:end_i]

            # partial_fit
            model.partial_fit(X_batch, y_batch)

            # Вычислим loss на этом батче
            y_pred_batch = model.predict(X_batch)
            batch_loss = 0.5 * np.mean((y_pred_batch - y_batch) ** 2)
            losses_in_epoch.append(batch_loss)

        epoch_loss_mean = np.mean(losses_in_epoch)
        epoch_losses.append(epoch_loss_mean)

    # По завершении эпох оценим MSE, R^2 на всей выборке (в масштабированном виде)
    y_pred_all_scaled = model.predict(X_scaled)
    # Считаем MSE, R2 в «скейленом» пространстве
    # Чтобы вернуть результат в исходном масштабе, придётся
    # инвертировать y_pred_all_scaled обратно. Сделаем это:
    y_pred_all = scaler_y.inverse_transform(y_pred_all_scaled.reshape(-1, 1)).ravel()
    y_true_all = y.values  # изначальные не-scaled

    final_mse = mean_squared_error(y_true_all, y_pred_all)
    final_r2 = r2_score(y_true_all, y_pred_all)

    return {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "epoch_losses": epoch_losses,
        "final_mse": final_mse,
        "final_r2": final_r2
    }
