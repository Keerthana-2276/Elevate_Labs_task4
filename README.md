
# 🚢 Titanic Data Preprocessing and Outlier Removal

This project demonstrates basic data preprocessing on the Titanic dataset using Python and Pandas in Google Colab.

---

## 📁 Dataset
We use the Titanic dataset containing passenger information such as age, fare, gender, class, etc.

---

## 🛠️ Tasks Performed

### 1. 🔍 Import Dataset & Basic Info
- Load the CSV file using pandas.
- Check the first few rows using `df.head()`.
- View data types and null values using `df.info()` and `df.isnull().sum()`.

```python
import pandas as pd

df = pd.read_csv('/content/Titanic-Dataset.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
```

---

### 2. 🧼 Handle Missing Values
- Fill missing **numerical** values (`Age`) using **mean**.
- Fill missing **categorical** values (`Embarked`) using **mode**.

```python
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

---

### 3. 🔢 Encode Categorical Features
Convert categorical columns to numerical using **Label Encoding**.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])            # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])  # C=0, Q=1, S=2 (may vary)
```

---

### 4. 📐 Standardize Numerical Features
Standardize `Age` and `Fare` so they are on the same scale using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
print("\n📐 Standardized 'Age' and 'Fare':\n", df[['Age', 'Fare']].head())
```

ℹ️ **Standardization Formula**:  
`z = (x - μ) / σ`  
This ensures the values have mean = 0 and standard deviation = 1.

---

### 5. 📊 Visualize & Remove Outliers (IQR Method)
Use **Interquartile Range (IQR)** method to detect and remove outliers from `Age` and `Fare`.

```python
for col in ['Age', 'Fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
```

```python
print("\n✅ Dataset shape after removing outliers:", df.shape)
```

🧹 This removes values that are too high or too low compared to the normal data range.

---

## ✅ Final Output

```python
print(df.shape)
```
Shows the new shape of the dataset after cleaning and outlier removal.

---

## 💡 Summary of Preprocessing Steps

| Step              | Description                                      |
|-------------------|--------------------------------------------------|
| 1. Import         | Load and explore the Titanic dataset             |
| 2. Missing Values | Fill missing Age with mean, Embarked with mode  |
| 3. Encoding       | Convert text (categorical) columns to numbers    |
| 4. Standardization| Scale Age and Fare using StandardScaler          |
| 5. Outliers       | Remove extreme values using IQR method           |

---

## 👩‍💻 Tools Used
- Python  
- Pandas  
- Scikit-learn  
- Google Colab  

---

## 📎 Notes
This preprocessing prepares the dataset for:

- 🚀 Machine Learning models  
- 📊 Exploratory Data Analysis (EDA)  
- 🤖 Classification tasks like survival prediction  

---

## 📌 Sample Output Preview

```python
✅ Dataset shape after removing outliers: (XXX, YYY)
```
Replace `XXX` and `YYY` with your actual dataset shape after preprocessing.
