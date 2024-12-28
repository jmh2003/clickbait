import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import glob, os, json
import torch
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

def load_training_data():  # 加载训练数据集
    base_path = "data/clickbait_detection_dataset"
    train_data = pd.read_json(os.path.join(base_path, "train.json"))
    test_data = pd.read_json(os.path.join(base_path, "test.json"))
    val_data = pd.read_json(os.path.join(base_path, "val.json"))
    df = pd.concat([train_data, test_data, val_data], ignore_index=True)
    return df

def load_news_data():   # 加载新闻数据集
    news_path = "news"
    news_files = glob.glob(os.path.join(news_path, "*_translated.json"))
    all_news = []
    
    for file in news_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            source = os.path.basename(file).replace('_translated.json', '')
            for item in data:
                item['source'] = source
            all_news.extend(data)
    return pd.DataFrame(all_news)

def train_and_predict_single_model(model_name, model_class, X_train, X_test, y_train, y_test, vectorizer, news_df, batch_size=1000):
    print(f"\n=== 开始训练 {model_name} 模型 ===")
    try:
        # 针对SVC特殊处理  模型初始化
        if model_name == "支持向量机":
            model = SVC(kernel='linear', probability=True, random_state=42)
        else:
            model = model_class()
        
        # 确保数据格式正确
        X_train = X_train.tocsr()
        X_test = X_test.tocsr()
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # 添加训练进度条
        print("开始训练...")
        n_samples = X_train.shape[0]
        batch_count = (n_samples + batch_size - 1) // batch_size
        
        if hasattr(model, "partial_fit"):   # 增量训练
            with tqdm(total=n_samples, desc=f"{model_name} 训练进度") as pbar:
                classes = np.unique(y_train)
                for i in range(0, n_samples, batch_size):
                    end_idx = min(i + batch_size, n_samples)
                    batch_X = X_train[i:end_idx]
                    batch_y = y_train[i:end_idx]
                    model.partial_fit(batch_X, batch_y, classes=classes)
                    pbar.update(end_idx - i)
        else:
            print(f"模型不支持增量训练，使用全量训练...")
            model.fit(X_train, y_train)
        
        # 评估模型
        print("\n评估模型...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} 准确率:", accuracy_score(y_test, y_pred))
        print(f"{model_name} 分类报告:")
        print(classification_report(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        
        
        # 预测新闻标题
        results = []
        print("\n开始预测新闻标题...")
        
        for i in tqdm(range(0, len(news_df), batch_size), desc=f"{model_name} 预测进度"):
            batch_df = news_df.iloc[i:i + batch_size]
            titles = batch_df['translated_title'].fillna('').tolist()
            title_vectors = vectorizer.transform(titles)
            
            predictions = model.predict(title_vectors)
            
            # 处理概率预测
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(title_vectors)
                prob_values = [float(p[1]) for p in probabilities]
            else:
                # 对于不支持概率预测的模型，使用决策函数值
                decision_values = model.decision_function(title_vectors)
                prob_values = [float((v + 1) / 2) for v in decision_values]
            
            for j, (title, orig_title, source, url) in enumerate(zip(
                batch_df['translated_title'],
                batch_df['original_title'],
                batch_df['source'],
                batch_df['url']
            )):
                if pd.notna(title):
                    results.append({
                        "title": str(title),
                        "original_title": str(orig_title),
                        "url": str(url),
                        "source": str(source),
                        "prediction": int(predictions[j]),
                        "probability": prob_values[j]
                    })
        
        # 保存结果
        output_path = f"{model_name}_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"\n{model_name} 预测结果已保存至 {output_path}")
        
    except Exception as e:
        print(f"{model_name} 处理失败: {str(e)}")
        raise
    finally:
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return accuracy, precision, recall, f1_score

def main():
    # 加载数据
    final_accuracies = []
    print("加载训练数据...")
    train_df = load_training_data()
    print("加载新闻数据...")
    news_df = load_news_data()
    
    # 数据预处理
    print("数据向量化...")
    vectorizer = CountVectorizer(max_features=10000)  # 限制特征数量
    X = vectorizer.fit_transform(train_df["text"])
    y = train_df["label"]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 逐个训练和预测模型
    models = [
        ("朴素贝叶斯", MultinomialNB),
        ("随机森林", RandomForestClassifier),
        ("逻辑回归", LogisticRegression),
        ("支持向量机", SVC)
    ]
    
    for model_name, model_class in models:
        try:
            accuracy,precision, recall, f1_score = train_and_predict_single_model(model_name, model_class, X_train, X_test, y_train, y_test,vectorizer, news_df)
            final_accuracies.append((model_name,accuracy,precision, recall, f1_score))
        except Exception as e:
            print(f"{model_name} 处理失败: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_results = []

    for i in tqdm(range(0, len(news_df), 1000), desc="最终结果进度"):
        batch_df = news_df.iloc[i:i + 1000]
        titles = batch_df['translated_title'].fillna('').tolist()
        title_vectors = vectorizer.transform(titles)
    
        predictions = {}
        for model_name, model_class in models:
            try:
                model = model_class()
                model.fit(X_train, y_train)
                predictions[model_name] = model.predict(title_vectors)
            except Exception as e:
                print(f"{model_name} 处理失败: {e}")
    
        for j, (title, orig_title, source, url) in enumerate(zip(
            batch_df['translated_title'],
            batch_df['original_title'],
            batch_df['source'],
            batch_df['url']
        )):
            if pd.notna(title):
                # 统计每个模型对当前标题的预测结果
                model_predictions = {model_name: pred[j] for model_name, pred in predictions.items()}
                # 如果有超过一半的模型预测为标题党，则最终结果也将其标记为标题党
                if sum(model_predictions.values()) > len(models) / 2:
                    final_results.append({
                        "title": str(title),
                        "original_title": str(orig_title),
                        "source": str(source),
                        "url": str(url),
                        "results":"标题党"
                    })
                else:
                    final_results.append({
                        "title": str(title),
                        "original_title": str(orig_title),
                        "source": str(source),
                        "url": str(url),
                        "results":"新闻"
                    })

    # 将最终结果写入JSON文件
    with open("final_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 提取模型名称和各个指标
    model_names = [item[0] for item in final_accuracies]
    accuracies = [item[1] for item in final_accuracies]
    precisions = [item[2] for item in final_accuracies]
    recalls = [item[3] for item in final_accuracies]
    f1_scores = [item[4] for item in final_accuracies]

    # 设置条形图的位置和宽度
    x = np.arange(len(model_names))
    width = 0.2

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制条形图
    rects1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy')
    rects2 = ax.bar(x - 0.5*width, precisions, width, label='Precision')
    rects3 = ax.bar(x + 0.5*width, recalls, width, label='Recall')
    rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score')

    # 添加标签和标题
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    # 添加数值标签到每个条形上
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.5f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    ax.set_ylim(0, 0.95)
    # 调整布局以防止标签重叠
    fig.tight_layout()

    # 显示图形
    plt.show()

    # 筛选出 results 为 "标题党" 的标题
    clickbait_titles = [(item['original_title'],item['url']) for item in final_results if item.get('results') == "标题党"]
    news_titles = [(item['original_title'],item['url']) for item in final_results if item.get('results') == "新闻"]
    # 创建 HTML 内容
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clickbait and News Titles</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 20px;
            }
            .container {
                display: flex;
                justify-content: space-between;
                width: 100%;
                max-width: 1200px;
            }
            .column {
                flex: 1;
                padding: 0 10px;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                background-color: #fff;
                margin: 10px 0;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            a {
                text-decoration: none;
                flex-grow: 1; /* 让链接占据剩余空间 */
            }
            a:hover {
                text-decoration: underline;
            }
            .clickbait a {
                color: #FF6347; /* 标题党的颜色 */
            }
            .news a {
                color: #4682B4; /* 新闻的颜色 */
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="column">
                <h1>Clickbait Titles</h1>
                <ul class="clickbait">
    """

    # 添加每个标题党的标题和链接到 HTML 内容中
    for title, url in clickbait_titles:
        html_content += f'        <li><a href="{url}">{title}</a></li>\n'

    # 结束 Clickbait 部分的 HTML 内容
    html_content += """
                </ul>
            </div>
            <div class="column">
                <h1>News Titles</h1>
                <ul class="news">
    """

    # 添加每个新闻的标题和链接到 HTML 内容中
    for title, url in news_titles:
        html_content += f'        <li><a href="{url}">{title}</a></li>\n'

    # 结束 News 部分的 HTML 内容
    html_content += """
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    # 将 HTML 内容写入文件
    with open("clickbait_and_news_titles.html", "w", encoding="utf-8") as file:
        file.write(html_content)

    print("HTML 文件已成功生成！")  

if __name__ == "__main__":
    main()