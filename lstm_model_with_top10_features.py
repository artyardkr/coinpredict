
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 피처 선택
try:
    df = pd.read_csv('integrated_data_full.csv', parse_dates=['Date'], index_col='Date')
except FileNotFoundError:
    print("오류: integrated_data_full.csv 파일을 찾을 수 없습니다.")
    exit()

features = [
    'Close', 'EMA5_close', 'SMA5_close', 'OBV', 'RSI', 'Williams_R', 
    'BB_low', 'CCI', 'Stoch_K', 'bc_miners_revenue', 'daily_return'
]

# 피처 존재 여부 확인
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"오류: 다음 피처를 데이터에서 찾을 수 없습니다: {', '.join(missing_features)}")
    exit()

data = df[features].copy()

# 2. 데이터 전처리
# 결측치 처리 (초기 이동평균 등 계산으로 인한 NaN 제거)
data.dropna(inplace=True)

if data.empty:
    print("오류: 데이터를 처리한 후 남은 데이터가 없습니다. 결측치가 너무 많을 수 있습니다.")
    exit()

# 데이터 분할 (80% 훈련, 20% 테스트)
training_data_len = int(np.ceil(len(data) * .8))
training_data = data.iloc[:training_data_len]
testing_data = data.iloc[training_data_len:]

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_data = scaler.fit_transform(training_data)
scaled_testing_data = scaler.transform(testing_data)

# 3. 시계열 데이터 생성
sequence_length = 60

def create_sequences(data, seq_len):
    x = []
    y = []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i, :])
        y.append(data[i, 0]) # 'Close' is the first column
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(scaled_training_data, sequence_length)
x_test, y_test = create_sequences(scaled_testing_data, sequence_length)

if x_train.shape[0] == 0 or x_test.shape[0] == 0:
    print("오류: 훈련 또는 테스트 시퀀스를 생성할 수 없습니다. 데이터가 너무 적습니다.")
    exit()
    
# 4. LSTM 모델 구축
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# 5. 모델 컴파일 및 훈련
model.compile(optimizer='adam', loss='mean_squared_error')
print("모델 훈련을 시작합니다...")
history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test), verbose=1)
print("모델 훈련이 완료되었습니다.")

# 6. 예측 수행
predictions = model.predict(x_test)

# 예측 결과를 원래 스케일로 되돌리기 위해 더미 배열 생성
dummy_predictions = np.zeros((len(predictions), len(features)))
dummy_predictions[:, 0] = predictions.ravel()
inversed_predictions = scaler.inverse_transform(dummy_predictions)[:, 0]

# 실제 값을 원래 스케일로 되돌리기
dummy_y_test = np.zeros((len(y_test), len(features)))
dummy_y_test[:, 0] = y_test.ravel()
inversed_y_test = scaler.inverse_transform(dummy_y_test)[:, 0]


# 7. 결과 시각화 및 저장
plt.figure(figsize=(16, 8))
plt.title('LSTM Model - Close Price Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

# 실제 테스트 데이터의 날짜 인덱스 가져오기
test_dates = testing_data.index[sequence_length:]

# 실제값과 예측값 plot
plt.plot(test_dates, inversed_y_test, label='Actual Price')
plt.plot(test_dates, inversed_predictions, label='Predicted Price')

plt.legend(loc='lower right')
plt.grid(True)

# 그래프 저장
output_filename = 'lstm_performance.png'
plt.savefig(output_filename)
print(f"예측 결과 그래프를 '{output_filename}' 파일로 저장했습니다.")

# 학습 손실 시각화 (선택 사항)
plt.figure(figsize=(16, 8))
plt.title('Model Training Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.grid(True)
plt.savefig('lstm_training_loss.png')
print("모델 학습 손실 그래프를 'lstm_training_loss.png' 파일로 저장했습니다.")
