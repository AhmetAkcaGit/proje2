#%% Kullanılan kütüphaneler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef, recall_score, precision_score, confusion_matrix
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns

#%% Veri setini dahil etme işlemi
# Veri setini fetch_ucirepo kullanarak indiriyoruz.
heart_disease=fetch_ucirepo(id=15)

# Özellikleri ve hedefleri tanımlıyoruz.
X=heart_disease.data.features
y=heart_disease.data.targets

#%% Veri ön işleme
# Eksik değerleri doldurma
imputer=SimpleImputer(strategy='mean') # Eksik değerleri eksik değerlerin ortalaması ile dolduruyoruz.
X_imputed=imputer.fit_transform(X)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test=train_test_split(X_imputed, y, test_size=0.2, random_state=42)

#%% Gaussian Naive Bayes modelini oluşturma ve eğitme
gnb=GaussianNB()
gnb.fit(X_train, y_train.values.ravel()) # .values.reval() ifadesi, y_train'in tek boyutlu bir dizi olmasını sağlar.

#%% Test seti üzerinde tahmin yapma
y_pred = gnb.predict(X_test)

#%% Modelin doğruluğunu değerlendirme
accuracy=accuracy_score(y_test, y_pred)
f1=f1_score(y_test, y_pred, pos_label=4)
mcc=matthews_corrcoef(y_test, y_pred)
recall=recall_score(y_test, y_pred, pos_label=4)
precision=precision_score(y_test, y_pred, pos_label=4)
conf_matrix=confusion_matrix(y_test, y_pred)

#%% Sonuçları yazdırma
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Matthews Correlation Coefficient:", mcc)
print("Recall:", recall)
print("Precision:", precision)
print("Classification Report:\n", classification_report(y_test, y_pred))

#%% Karışıklık matrisini görselleştirme
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["İyi Huylu", "Kötü Huylu"], yticklabels=["İyi Huylu", "Kötü Huylu"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()