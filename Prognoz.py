import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.family'] = 'DejaVu Sans'

num_hidden_neurons = 25
window_size = 11
epochs = 1000
forecast_steps = 10

t_end_values = [2, 4, 6, 8]

errors_list = []
forecasts_all = []
forecast_times_all = []

t_full = np.arange(-2, 8 + forecast_steps * 0.05, 0.05)
y_full = t_full**3 - 2*t_full**2 + 2

for t_end in t_end_values:
    t = np.arange(-2, t_end, 0.05)
    y = t**3 - 2*t**2 + 2
    end_t = t[-1]
    step = t[1] - t[0]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train = []
    y_train = []
    
    for i in range(len(t) - window_size):
        X_train.append(y_scaled[i:i+window_size])
        y_train.append(y_scaled[i+window_size])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = keras.Sequential([
        keras.layers.LSTM(num_hidden_neurons, activation='tanh', input_shape=(window_size, 1)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    
    forecast = []
    input_window = y_scaled[-window_size:]
    
    for _ in range(forecast_steps):
        prediction = model.predict(input_window.reshape(1, window_size, 1), verbose=0)
        forecast.append(prediction[0, 0])
        input_window = np.append(input_window[1:], prediction)
    
    forecast_inverse = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    
    forecast_t = end_t + step * np.arange(1, forecast_steps + 1)
    actual_values = forecast_t**3 - 2*forecast_t**2 + 2
    error = np.abs((forecast_inverse - actual_values) / actual_values) * 100
    
    errors_list.append({
        't_end': t_end,
        'forecast': forecast_inverse,
        'actual_values': actual_values,
        'error': error
    })
    
    forecasts_all.append(forecast_inverse)
    forecast_times_all.append(forecast_t)
    
    print(f"\nІнтервал спостереження: [-2, {t_end}]")
    print(f"Прогнозовані значення: {forecast_inverse}")
    print(f"Реальні значення: {actual_values}")
    print(f"Похибка (%): {error}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Втрати на тренуванні', color='green')
    plt.xlabel('Епоха')
    plt.ylabel('Втрати')
    plt.title(f'Процес навчання моделі LSTM (інтервал спостереження [-2, {t_end}])')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Таблиця похибки прогнозу:")
    for i in range(forecast_steps):
        print(f"{i+1}. Точне значення: {actual_values[i]:.4f}, Прогноз: {forecast_inverse[i]:.4f}, Похибка: {error[i]:.2f}%")

plt.figure(figsize=(12, 7))
plt.plot(t_full, y_full, label='Функція $t^3 - 2t^2 + 2$', color='blue')

colors = ['red', 'green', 'orange', 'purple']
line_styles = ['--', '-.', ':', '-']

for i, t_end in enumerate(t_end_values):
    plt.plot(forecast_times_all[i], forecasts_all[i], label=f'Прогноз при t_end={t_end}', color=colors[i], linestyle=line_styles[i], marker='o')

plt.xlabel('t')
plt.ylabel('Значення')
plt.title('Прогнозування функції $t^3 - 2t^2 + 2$ за допомогою LSTM\nпри різних інтервалах спостереження')
plt.legend()
plt.grid(True)
plt.show()

mean_errors = [np.mean(errors['error']) for errors in errors_list]
t_end_values = [errors['t_end'] for errors in errors_list]

plt.figure(figsize=(10, 6))
plt.plot(t_end_values, mean_errors, marker='o', linestyle='-', color='purple')
plt.xlabel('Кінець інтервалу спостереження t_end')
plt.ylabel('Середня відносна похибка (%)')
plt.title('Залежність похибки прогнозу від довжини інтервалу спостереження')
plt.grid(True)
plt.show()