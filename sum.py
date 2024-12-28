import json
from collections import defaultdict

def analyze_news_sources():
    # 读取JSON文件
    with open('final_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 创建统计字典
    stats = defaultdict(lambda: {"标题党": 0, "新闻": 0, "总数": 0})
    
    # 统计各来源的数量
    for item in results:
        source = item['source']
        result_type = item['results']
        stats[source][result_type] += 1
        stats[source]["总数"] += 1
    
    # 打印统计结果
    print("\n=== 各新闻来源标题党统计 ===")
    print(f"{'来源':<10} {'标题党':<10} {'新闻':<10} {'总数':<10} {'标题党比例':<10}")
    print("-" * 50)
    
    for source, counts in stats.items():
        clickbait = counts["标题党"]
        news = counts["新闻"]
        total = counts["总数"]
        ratio = clickbait / total * 100 if total > 0 else 0
        
        print(f"{source:<10} {clickbait:<10} {news:<10} {total:<10} {ratio:.2f}%")
    
    # 保存结果到文件
    with open('source_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(dict(stats), f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    analyze_news_sources()