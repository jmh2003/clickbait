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
import gc

def load_training_data():
    base_path = "data/clickbait_detection_dataset"
    train_data = pd.read_json(os.path.join(base_path, "train.json"))
    test_data = pd.read_json(os.path.join(base_path, "test.json"))
    val_data = pd.read_json(os.path.join(base_path, "val.json"))
    df = pd.concat([train_data, test_data, val_data], ignore_index=True)
    return df

def load_news_data():
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

def train_and_predict_single_model(model_name, model, X_train, X_test, y_train, y_test, vectorizer, news_df):
    print(f"\n=== 开始训练 {model_name} 模型 ===")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    print(f"{model_name} 准确率:", accuracy_score(y_test, y_pred))
    print(f"{model_name} 分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 预测新闻标题
    results = []
    for title, source in zip(news_df['translated_title'], news_df['source']):
        if isinstance(title, str):
            title_vector = vectorizer.transform([title])
            prediction = model.predict(title_vector)[0]
            probability = model.predict_proba(title_vector)[0][1]
            
            results.append({
                "title": title,
                "source": source,
                "prediction": int(prediction),
                "probability": float(probability)
            })
    
    # 保存结果
    output_path = f"{model_name}_predictions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n{model_name} 预测结果已保存至 {output_path}")
    
    # 打印统计信息
    for source in news_df['source'].unique():
        source_results = [r for r in results if r['source'] == source]
        clickbait_count = sum(1 for r in source_results if r['prediction'] == 1)
        total = len(source_results)
        print(f"\n新闻源 {source} 的统计信息:")
        print(f"标题党比例 = {clickbait_count/total*100:.2f}% ({clickbait_count}/{total})")
    
    # 清理内存
    del model
    gc.collect()

def main():
    # 加载数据
    print("加载训练数据...")
    train_df = load_training_data()
    print("加载新闻数据...")
    news_df = load_news_data()
    
    # 数据预处理
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_df["text"])
    y = train_df["label"]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 逐个训练和预测模型
    models = [
        ("朴素贝叶斯", MultinomialNB()),
        ("随机森林", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("逻辑回归", LogisticRegression(random_state=42, max_iter=1000)),
        ("支持向量机", SVC(kernel='linear', probability=True, random_state=42))
    ]
    
    for model_name, model in models:
        train_and_predict_single_model(
            model_name, 
            model, 
            X_train, 
            X_test, 
            y_train, 
            y_test,
            vectorizer, 
            news_df
        )
        gc.collect()  # 强制垃圾回收

if __name__ == "__main__":
    main()