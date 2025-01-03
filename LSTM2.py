import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input

def lstm_model(time_steps, future_steps, num_epochs=50):
    # 读取数据，选择目标列和特征列
    df = pd.read_csv('day.csv', index_col='date')
    selected_features = ['cnt', 'temp', 'hum', 'windspeed']
    df = df[selected_features]

    # 转换索引为日期格式
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(df)

    # 数据集构造函数
    def generate_data(dataset, t_steps=1):
        features, target = [], []
        for i in range(len(dataset) - t_steps):
            features.append(dataset[i:i + t_steps, 1:])  # 使用除目标列以外的所有特征
            target.append(dataset[i + t_steps, 0])  # 目标列是 'cnt'
        return np.array(features), np.array(target)

    # 构造训练集和测试集
    X, y = generate_data(normalized_data, t_steps=time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 打印数据形状以供检查
    print(f"训练集 X 形状: {X_train.shape}, 测试集 X 形状: {X_test.shape}")

    # LSTM 模型定义
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)  # 输出预测目标值
    ])

    # 模型编译
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 模型训练
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(X_test, y_test))

    # 模型预测
    predictions = model.predict(X_test)

    # 反归一化预测结果和实际值
    predictions_rescaled = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), len(selected_features) - 1)))))[:, 0]
    y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), len(selected_features) - 1)))))[:, 0]

    # 未来预测
    last_sequence = normalized_data[-time_steps:, 1:]
    future_predictions = []

    for _ in range(future_steps):
        next_prediction = model.predict(last_sequence[np.newaxis, :, :])
        future_predictions.append(next_prediction[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = np.append(next_prediction[0], [0] * (last_sequence.shape[1] - 1))

    # 反归一化未来预测结果
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_rescaled = scaler.inverse_transform(np.hstack((future_predictions, np.zeros((len(future_predictions), len(selected_features) - 1)))))[:, 0]

    # 生成未来预测日期
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=future_steps + 1, freq='D')[1:]

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(y_test_rescaled):], y_test_rescaled, label='实际租赁数', color='blue')
    plt.plot(df.index[-len(y_test_rescaled):], predictions_rescaled, label='预测租赁数', color='red')
    plt.plot(future_dates, future_predictions_rescaled, label='未来预测', color='green', linestyle='--')
    plt.legend()
    plt.title('租赁总数预测（LSTM模型）')
    plt.show()
