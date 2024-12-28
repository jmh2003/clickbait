import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import glob, os
import json
from langdetect import detect
from googletrans import Translator

# 初始化翻译器
translator = Translator()
#朴素贝叶斯


# 读取 JSON 数据
json_files = glob.glob(os.path.join("data", "*.json"))
df_list = [pd.read_json(f) for f in json_files]
df = pd.concat(df_list, ignore_index=True)

# 调整列名
df.rename(columns={"article_title": "text", "clickbait": "label"}, inplace=True)
df = df[["text", "label"]]

# df_records = df.to_dict(orient="records")
# with open("data/merged_data.json", "w", encoding="utf-8") as f:
#     json.dump(df_records, f, ensure_ascii=False, indent=2)

# 向量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"].astype(str))
y = df["label"]

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train.shape:",X_train.shape)
print("X_test.shape:",X_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape", y_test.shape)

# 训练模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

while True:
    print("请输入标题<<")
    text = input()
        # 检测语言
    lang = detect(text)
    if lang != 'en':
        # 翻译成英文
        translated = translator.translate(text, src=lang, dest='en')
        translated_text = translated.text
        print(f"翻译后的文本: {translated_text}")
        print(f"原来文本的语言是：{lang}")
    else:
        translated_text = text

    # if not text:
    #     break  # 输入为空时退出循环
    text_vector = vectorizer.transform([translated_text])
    predict = model.predict(text_vector)
    probabilities = model.predict_proba(text_vector)
    
    # 获取每个类别的概率
    prob_non_clickbait = probabilities[0][0]
    prob_clickbait = probabilities[0][1]
    
    print(f"这个是预测结果 of '{text}': {predict[0]}")
    print(f"概率分布 - 非标题党: {prob_non_clickbait * 100:.2f}%, 标题党: {prob_clickbait * 100:.2f}%")

