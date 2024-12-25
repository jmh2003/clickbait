# 支持向量机

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import glob, os

# 读取 JSON 数据
json_files = glob.glob(os.path.join("data", "*.json"))
df_list = [pd.read_json(f) for f in json_files]
df = pd.concat(df_list, ignore_index=True)

# 调整列名
df.rename(columns={"article_title": "text", "clickbait": "label"}, inplace=True)
df = df[["text", "label"]]

# 向量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"].astype(str))
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
