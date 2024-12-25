import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import glob, os, json
from langdetect import detect
from googletrans import Translator

# 初始化翻译器
translator = Translator()

# 朴素贝叶斯

# 读取 JSON 数据
json_files = glob.glob(os.path.join("data", "*.json"))
df_list = [pd.read_json(f) for f in json_files]
df = pd.concat(df_list, ignore_index=True)

# 调整列名
df.rename(columns={"article_title": "text", "clickbait": "label"}, inplace=True)
df = df[["text", "label", "article_url"]]  # 保留 article_url

# 向量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"].astype(str))
y = df["label"]

# 划分训练和测试集，同时保留测试集的文本数据和链接
X_train, X_test, y_train, y_test, url_train, url_test = train_test_split(
    X, y, df["article_url"], test_size=0.2, random_state=42
)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

# 训练模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 读取 Excel 文件的第一列
excel_file = "test_data/今日头条.xlsx"  # 替换为你的 Excel 文件路径
try:
    df_excel = pd.read_excel(excel_file, usecols=[0])  # 读取第一列
    titles = df_excel.iloc[:, 0].dropna().tolist()  # 获取第一列并转换为列表
except Exception as e:
    print(f"读取 Excel 文件时发生错误: {e}")
    titles = []

print("读取的标题数:", len(titles))

if not titles:
    print("没有标题可供处理。")
    exit()

# 分离英语和非英语标题
english_titles = []
non_english_titles = []
original_indices = []  # 存储非英语标题的原始索引

for idx, text in enumerate(titles):
    if not isinstance(text, str):
        print(f"跳过非字符串类型的输入: {text}")
        continue
    try:
        lang = detect(text)
    except Exception as e:
        print(f"检测语言时出错的标题 '{text}': {e}")
        continue

    if lang != 'en':
        non_english_titles.append(text)
        original_indices.append(idx)
        english_titles.append(None)  # 占位
    else:
        english_titles.append(text)

print(english_titles)
print(non_english_titles)


# 批量翻译非英语标题
translated_non_english = []
if non_english_titles:
    print("开始翻译！")
    try:
        print(non_english_titles[12])
        print(type(non_english_titles[12]))
        print(type(non_english_titles))
        print(non_english_titles)
        # exit()
        for title in non_english_titles:
            trans_title= translator.translate(title, src='auto', dest='en')
            translated_non_english.append(trans_title)
        print("翻译的结果是：")
        print(translated_non_english)
        exit()    
        # 移除 None 和空字符串
        non_english_titles = [title for title in non_english_titles if isinstance(title, str) and title.strip()]
        # exit()
        translations = translator.translate(non_english_titles, src='auto', dest='en')
        print("这个是翻译的结果：",translations)
        # exit()

        # 如果只有一个翻译结果，确保是列表
        if isinstance(translations, list):
            translated_non_english = [translation.text for translation in translations]
        else:
            translated_non_english = [translations.text]
    except Exception as e:
        print(f"批量翻译时发生错误: {e}")

print(translated_non_english)

# 将翻译后的标题填回原列表
for idx, translated_text in zip(original_indices, translated_non_english):
    english_titles[idx] = translated_text
    print(f"原文: {titles[idx]}")
    print(f"翻译后的文本: {translated_text}\n")

# 移除所有None值（如果有的话）
translated_titles = [text for text in english_titles if text is not None]

if not translated_titles:
    print("没有翻译后的标题可供预测。")
    exit()

# 向量化所有翻译后的标题
text_vectors = vectorizer.transform(translated_titles)

# 批量预测
predictions = model.predict(text_vectors)
probabilities = model.predict_proba(text_vectors)

# 打印结果
for original_text, translated_text, prediction, probs in zip(titles, translated_titles, predictions, probabilities):
    prob_non_clickbait, prob_clickbait = probs
    print(f"原文: {original_text}")
    print(f"翻译后的文本: {translated_text}")
    print(f"预测结果: {'标题党' if prediction == 1 else '非标题党'}")
    print(f"概率分布 - 非标题党: {prob_non_clickbait * 100:.2f}%, 标题党: {prob_clickbait * 100:.2f}%\n")

