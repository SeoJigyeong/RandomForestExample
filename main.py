import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. 데이터 생성 (가상의 데이터)
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=100, freq='W')
marketing_spend = np.random.normal(loc=5000, scale=1000, size=100)  # 마케팅 지출
price_index = np.random.normal(loc=100, scale=10, size=100)           # 가격 지수

# 매출은 마케팅 지출과 가격 지수에 영향을 받는다고 가정하고 생성
sales = 20000 + 3 * marketing_spend - 150 * price_index + np.random.normal(loc=0, scale=3000, size=100)

# 데이터프레임 생성
data = pd.DataFrame({
    "Date": dates,
    "MarketingSpend": marketing_spend,
    "PriceIndex": price_index,
    "Sales": sales
})
data.set_index("Date", inplace=True)

# 2. 특징(feature)과 타깃(target) 변수 설정
X = data[["MarketingSpend", "PriceIndex"]]
y = data["Sales"]

# 3. 학습 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 랜덤 포레스트 회귀 모델 생성 및 학습
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. 테스트 데이터에 대한 예측 및 평가
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 6. 실제 매출과 예측 매출 시각화
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.xlabel("실제 매출")
plt.ylabel("예측 매출")
plt.title("실제 매출 vs 예측 매출")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # y=x 선
plt.show()
