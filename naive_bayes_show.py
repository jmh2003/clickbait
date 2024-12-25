import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import glob, os, json

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
X_train, X_test, y_train, y_test, text_train, text_test, url_train, url_test = train_test_split(
    X, y, df["text"], df["article_url"], test_size=0.2, random_state=42
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

# 将测试结果保存为HTML
test_results = pd.DataFrame({
    '文本': text_test.reset_index(drop=True),
    '实际标签': y_test.reset_index(drop=True),
    '预测标签': y_pred,
    '文章链接': url_test.reset_index(drop=True)
})

# 根据预测标签分组
grouped = test_results.groupby('预测标签')

# 创建HTML内容
html_content = '''
<html>
<head>
    <meta charset="UTF-8">
    <title>分类结果</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            color: #333333;
        }
        h2 {
            color: #333333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
        }
        th, td {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 12px;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        a {
            text-decoration: none;
            color: #1a0dab;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>分类结果</h1>
'''

for label, group in grouped:
    html_content += f'<h2>类别 {label}</h2>\n'
    html_content += '<table>\n<thead>\n<tr>\n'
    html_content += '<th>文本</th>\n<th>实际标签</th>\n<th>预测标签</th>\n</tr>\n</thead>\n<tbody>\n'
    for index, row in group.iterrows():
        text = row['文本']
        actual = row['实际标签']
        predicted = row['预测标签']
        url = row['文章链接']
        html_content += f'<tr>\n<td><a href="{url}" target="_blank">{text}</a></td>\n<td>{actual}</td>\n<td>{predicted}</td>\n</tr>\n'
    html_content += '</tbody>\n</table>\n'

html_content += '''
</body>
</html>
'''

# 保存为HTML文件
with open("data/test_results.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("分类结果已保存为 data/test_results.html")