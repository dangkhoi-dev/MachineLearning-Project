# PHÂN TÍCH NGUY CƠ OVERFITTING - CREDIT CARD FRAUD DETECTION

## 🚨 NHỮNG DẤU HIỆU OVERFITTING NGHIÊM TRỌNG

### 1. **IMBALANCED DATA CỰC KỲ NGHIÊM TRỌNG**
- **Tỉ lệ**: 99.83% giao dịch hợp lệ vs 0.17% gian lận
- **Vấn đề**: Mô hình có thể đạt 99.8% accuracy chỉ bằng cách dự đoán TẤT CẢ là "không gian lận"
- **Nguy cơ**: Mô hình "học thuộc lòng" pattern của class majority

### 2. **THIẾU VALIDATION SET**
```python
# Chỉ có train/test split
X_train, X_test, y_train, y_test = train_test_split(...)
```
- **Vấn đề**: Không có validation set để monitor overfitting
- **Hậu quả**: Không biết khi nào mô hình bắt đầu overfit

### 3. **SỬ DỤNG ACCURACY LÀM METRIC CHÍNH**
- **Trong imbalanced data**: Accuracy là metric gây hiểu lầm
- **Ví dụ**: Model dự đoán tất cả = 0 → Accuracy = 99.8%
- **Thực tế**: Model hoàn toàn vô dụng cho fraud detection

### 4. **KHÔNG CÓ REGULARIZATION**
```python
# Decision Tree không có max_depth, min_samples_split
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)

# Logistic Regression không có regularization
model = LogisticRegression()
```

### 5. **KHÔNG CÓ CROSS-VALIDATION**
- Chỉ test trên 1 split duy nhất
- Kết quả có thể may mắn hoặc thiên vị

## ⚠️ KẾT QUẢ TỪ NOTEBOOK MẪU CHO THẤY OVERFITTING

### Decision Tree Results:
```
Class 0: precision=1.00, recall=1.00, f1=1.00
Class 1: precision=0.83, recall=0.81, f1=0.82
Overall accuracy: 1.00
```
→ **NGHI NGỜ OVERFITTING** - Perfect score cho class 0

### Logistic Regression Results:
```
Training accuracy: 0.935
Test accuracy: 0.919
```
→ **Suy giảm performance** từ train → test

## 🔧 CÁCH PHÁT HIỆN OVERFITTING

### 1. **So sánh Train vs Test Performance**
```python
# Train accuracy
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

# Test accuracy  
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"Train: {train_accuracy:.4f}")
print(f"Test: {test_accuracy:.4f}")
print(f"Gap: {train_accuracy - test_accuracy:.4f}")
```

### 2. **Sử dụng Metrics phù hợp cho Imbalanced Data**
```python
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Precision, Recall, F1 cho class thiểu số
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
print(f"Fraud Precision: {precision[1]:.4f}")
print(f"Fraud Recall: {recall[1]:.4f}")
print(f"Fraud F1: {f1[1]:.4f}")

# AUC Score
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC: {auc:.4f}")
```

### 3. **Learning Curves**
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='f1'
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train F1')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation F1')
plt.legend()
plt.show()
```

## 🛡️ GIẢI PHÁP CHỐNG OVERFITTING

### 1. **Cross-Validation với Stratification**
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print(f"CV F1 Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 2. **Regularization**
```python
# Decision Tree với constraints
dtree = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# Logistic Regression với regularization
model = LogisticRegression(C=0.1, penalty='l2')
```

### 3. **Xử lý Imbalanced Data**
```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class Weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
model = LogisticRegression(class_weight='balanced')
```

### 4. **Early Stopping & Validation Split**
```python
from sklearn.model_selection import train_test_split

# Chia thành train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp)

# Monitor validation performance
```

## 🎯 METRICS ĐÚNG CHO FRAUD DETECTION

### Thứ tự ưu tiên:
1. **Recall (Sensitivity)** - Phát hiện được bao nhiêu % fraud thực tế
2. **Precision** - Trong những dự đoán fraud, bao nhiêu % là đúng
3. **F1-Score** - Cân bằng giữa Precision và Recall
4. **AUC-ROC** - Khả năng phân biệt giữa fraud và non-fraud
5. **Accuracy** - Chỉ tham khảo, không nên làm metric chính

### Confusion Matrix Analysis:
```
                Predicted
                0    1
Actual    0   TN   FP  ← Type I Error (False Alarm)
          1   FN   TP  ← Type II Error (Missed Fraud) - NGUY HIỂM!
```

**Trong fraud detection: Type II Error (FN) nghiêm trọng hơn Type I Error (FP)**
