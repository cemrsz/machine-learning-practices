## 1. Gerekli kütüphaneleri içe aktar
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from helper import plot_boundary
from prettytable import PrettyTable
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

## 2. Görüntüleme ayarlarını yap
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 20)

plt.rcParams["figure.figsize"] = (12, 8)

## 3. Train ve test dosyalarını DataFrame olarak oku
elect_train = pd.read_csv("election_train.csv")
elect_test = pd.read_csv("election_test.csv")

elect_train.head()

## 4. İlk model için X ve y değişkenlerini hazırla
### edTest(test_data)

# Train datasından predictor değişkenleri seç
X_train = elect_train[["minority", "bachelor"]]

# Test datasından predictor değişkenleri seç
X_test = elect_test[["minority", "bachelor"]]

# Train datasından hedef değişkeni seç
y_train = elect_train["won"]

# Test datasından hedef değişkeni seç
y_test = elect_test["won"]

## 5. Derinliği 2 ve 10 olan Decision Tree modellerini eğit
### edTest(test_models)

# Derinliği 2 olan Decision Tree classifier oluştur
dt1 = DecisionTreeClassifier(max_depth=2)

# Modeli train datası üzerinde eğit
dt1.fit(X_train, y_train)

# Derinliği 10 olan Decision Tree classifier oluştur
dt2 = DecisionTreeClassifier(max_depth=10)

# Modeli train datası üzerinde eğit
dt2.fit(X_train, y_train)

## 6. Decision boundary görselleştir
plot_boundary(elect_train, dt1, dt2)

## 7. Daha fazla predictor ile yeni X değişkenlerini oluştur

# Kullanılacak predictor kolonları
pred_cols = [
    'minority',
    'density',
    'hispanic',
    'obesity',
    'female',
    'income',
    'bachelor',
    'inactivity'
]

# Train datasından predictor değişkenleri seç
X_train = elect_train[pred_cols]

# Test datasından predictor değişkenleri seç
X_test = elect_test[pred_cols]

# Train datasından hedef değişkeni seç
y_train = elect_train["won"]

# Test datasından hedef değişkeni seç
y_test = elect_test["won"]

## 8. Derinliği 2, 10 ve 15 olan modelleri oluştur ve eğit

# Derinliği 2 olan Decision Tree classifier oluştur
dt1 = DecisionTreeClassifier(max_depth=2)

# Derinliği 10 olan Decision Tree classifier oluştur
dt2 = DecisionTreeClassifier(max_depth=10)

# Derinliği 15 olan Decision Tree classifier oluştur
dt3 = DecisionTreeClassifier(max_depth=15)

# Modelleri train datası üzerinde eğit
dt1.fit(X_train, y_train)
dt2.fit(X_train, y_train)
dt3.fit(X_train, y_train)

## 9. Train ve test accuracy değerlerini hesapla
### edTest(test_accuracy)

# Derinliği 2 olan modelin train ve test accuracy değerleri
dt1_train_acc = dt1.score(X_train, y_train)
dt1_test_acc = dt1.score(X_test, y_test)

# Derinliği 10 olan modelin train ve test accuracy değerleri
dt2_train_acc = dt2.score(X_train, y_train)
dt2_test_acc = dt2.score(X_test, y_test)

# Derinliği 15 olan modelin train ve test accuracy değerleri
dt3_train_acc = dt3.score(X_train, y_train)
dt3_test_acc = dt3.score(X_test, y_test)

## 10. Accuracy skorlarını tablo halinde göster
pt = PrettyTable()

pt.field_names = [
    'Max Depth',
    'Number of Features',
    'Train Accuracy',
    'Test Accuracy'
]

pt.add_row([
    2,
    len(pred_cols),
    round(dt1_train_acc, 4),
    round(dt1_test_acc, 4)
])

pt.add_row([
    10,
    len(pred_cols),
    round(dt2_train_acc, 4),
    round(dt2_test_acc, 4)
])

pt.add_row([
    15,
    len(pred_cols),
    round(dt3_train_acc, 4),
    round(dt3_test_acc, 4)
])

print(pt)
