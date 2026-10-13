import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


# 1. Генерація випадкового набору даних

np.random.seed()
n = 1000
x = np.random.uniform(0, 100, 1000)
y = 3.8 * x + np.random.normal(0, 20, 1000)
x = x.reshape(-1, 1)


# 2. Нормалізація даних методом Z-score

x_mean = x.mean()
x_std = x.std()
x_normalized = (x - x_mean) / x_std


# 3. Розділення на навчальну та тестову вибірки

x_train, x_test, y_train, y_test = train_test_split(
    x_normalized,
    y,
    test_size=0.2
)

# 4. Навчання KNN-регресорадля різних значень k

k_values = range(1, 50)

r2_scores = []

for k in k_values:

    knn = KNeighborsRegressor(n_neighbors=k)

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    r2 = r2_score(y_test, y_pred)

    r2_scores.append(r2)

    print(f"k = {k:2d} | R2 = {r2:.4f}")


# 5. Вибір найкращого k

best_index = np.argmax(r2_scores)
best_k = list(k_values)[best_index]
best_r2 = r2_scores[best_index]

print("Найкраще значення k:", best_k)
print("Найкращий коефіцієнт R2:", round(best_r2, 4))

best_model = KNeighborsRegressor(n_neighbors=best_k)
best_model.fit(x_train, y_train)

y_pred = best_model.predict(x_test)

# 6. Візуалізація результатів

sort_index = np.argsort(x_test[:, 0])
x_test_sorted = x_test[sort_index]
y_test_sorted = y_test[sort_index]
y_pred_sorted = y_pred[sort_index]

# Графік згенерованого набору даних
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Згенерований набір даних")
plt.grid(True)
plt.show(block=False)

# Графік тестової та дослідної вибірок
plt.figure(figsize=(10, 6))
plt.scatter(
    x_train,
    y_train,
    s=25,
    label='Дослідна вибірка'
)

plt.scatter(
    x_test,
    y_test,
    s=25,
    label='Тестова вибірка'
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Тестова та дослідна вибірки")
plt.grid(True)
plt.show(block=False)

# Графік залежності R2 від k
plt.figure(figsize=(10, 5))
plt.plot(k_values, r2_scores, marker='o')
plt.xlabel("k")
plt.ylabel("R2")
plt.title("Залежність якості KNN-регресії від k")
plt.grid(True)

plt.axvline(
    best_k,
    linestyle="--",
    label=f"Найкраще k = {best_k}"
)

plt.legend()
plt.show(block=False)

# Графік фактичних і прогнозованих значень
plt.figure(figsize=(10, 6))

plt.scatter(
    x_test_sorted,
    y_test_sorted,
    label="Фактичні значення",
    alpha=0.6
)

plt.plot(
    x_test_sorted,
    y_pred_sorted,
    linewidth=2,
    label="KNN прогноз"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"KNN-регресія: k = {best_k}, R2 = {best_r2:.4f}")
plt.legend()
plt.grid(True)
plt.show()