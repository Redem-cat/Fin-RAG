"""
知识图谱模块功能测试脚本

测试范围：
1. kg_crawler - 数据采集（API + 新闻爬虫）
2. kg_builder - 知识图谱构建（实体抽取 + Neo4j 写入）
3. knowledge_graph - Neo4j 连接与查询

运行方式：
    python test_kg_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import json

TEST_OUTPUT_DIR = Path("test_output")
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name, success, detail=""):
    status = "OK" if success else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"       {detail}")


# ==================== 1. 测试 kg_crawler.api_sources ====================
def test_api_sources():
    print_section("1. 测试金融数据 API 采集 (kg_crawler.api_sources)")

    try:
        # 绕过 __init__.py，避免 schedule 模块缺失导致导入失败
        from src.kg_crawler.api_sources import FinancialDataSource
        ds = FinancialDataSource()
        print_result("导入 FinancialDataSource", True)
    except Exception as e:
        print_result("导入 FinancialDataSource", False, str(e))
        return

    # 1.1 测试获取公司列表
    try:
        companies = ds.get_all_companies()
        print_result("获取公司列表", len(companies) > 0,
                     f"获取到 {len(companies)} 家公司")
        if companies:
            print(f"       示例: {companies[0].get('name', 'N/A')} ({companies[0].get('code', 'N/A')})")
    except Exception as e:
        print_result("获取公司列表", False, str(e))

    # 1.2 测试获取单个公司信息
    try:
        info = ds.get_company_detail("600519")
        print_result("获取公司详情", info is not None,
                     f"{info.name if info else 'N/A'} ({info.sector if info else 'N/A'})")
    except Exception as e:
        print_result("获取公司详情", False, str(e))

    # 1.3 测试获取行业分类
    try:
        sectors = ds.get_sectors()
        print_result("获取行业分类", len(sectors) > 0,
                     f"获取到 {len(sectors)} 个行业")
        if sectors:
            print(f"       示例: {sectors[0].name}")
    except Exception as e:
        print_result("获取行业分类", False, str(e))

    # 1.4 测试获取实时行情
    try:
        quote = ds.get_realtime_quote("600519")
        print_result("获取实时行情", bool(quote),
                     f"价格: {quote.get('price', 'N/A')}" if quote else "无数据")
    except Exception as e:
        print_result("获取实时行情", False, str(e))

    # 1.5 测试获取财经新闻
    try:
        news = ds.get_news(count=3)
        print_result("获取财经新闻", len(news) > 0,
                     f"获取到 {len(news)} 条新闻")
        if news:
            print(f"       示例: {news[0].title[:40]}...")
            # 保存结果
            with open(TEST_OUTPUT_DIR / "api_news.json", "w", encoding="utf-8") as f:
                json.dump([n.to_dict() for n in news], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print_result("获取财经新闻", False, str(e))


# ==================== 2. 测试 kg_crawler.news_crawler ====================
def test_news_crawler():
    print_section("2. 测试新闻爬虫 (kg_crawler.news_crawler)")

    try:
        from src.kg_crawler.news_crawler import get_news_crawler
        crawler = get_news_crawler()
        print_result("导入 FinancialNewsCrawler", True)
    except Exception as e:
        print_result("导入 FinancialNewsCrawler", False, str(e))
        return

    # 2.1 测试爬取新闻（限制数量避免太慢）
    try:
        news_list = crawler.crawl_all(max_count_per_source=3)
        print_result("爬取财经新闻", len(news_list) > 0,
                     f"爬取到 {len(news_list)} 条新闻")
        if news_list:
            print(f"       示例: {news_list[0].title[:40]}...")
            crawler.save_news_to_json(news_list, TEST_OUTPUT_DIR / "crawled_news.json")
    except Exception as e:
        print_result("爬取财经新闻", False, str(e))

    # 2.2 测试缓存统计
    try:
        stats = crawler.get_statistics()
        print_result("获取缓存统计", True,
                     f"历史记录 {stats['total_urls']} 条")
    except Exception as e:
        print_result("获取缓存统计", False, str(e))


# ==================== 3. 测试 kg_builder.entity_extractor ====================
def test_entity_extractor():
    print_section("3. 测试实体抽取 (kg_builder.entity_extractor)")

    try:
        from src.kg_builder.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        print_result("导入 EntityExtractor", True)
    except Exception as e:
        print_result("导入 EntityExtractor", False, str(e))
        return

    # 3.1 测试规则抽取
    test_text = """
    贵州茅台（股票代码：600519）今日发布公告，
    董事长丁雄军表示公司将继续深耕白酒行业。
    2024年营收达到1500亿元，同比增长18%。
    科技行业和新能源板块表现活跃。
    """

    try:
        entities, relations = extractor.extract_from_text(test_text)
        print_result("规则抽取实体", len(entities) > 0,
                     f"抽取到 {len(entities)} 个实体, {len(relations)} 个关系")
        for e in entities:
            print(f"       - {e.name} ({e.entity_type}, conf={e.confidence})")
    except Exception as e:
        print_result("规则抽取实体", False, str(e))

    # 3.2 测试从新闻 dict 抽取
    try:
        from src.kg_builder.entity_extractor import extract_entities_from_news
        news_data = {
            "title": "比亚迪发布新款电动车",
            "content": "比亚迪（002594）今日发布新款电动车，董事长王传福出席发布会。新能源汽车行业迎来新机遇。"
        }
        entities, relations = extract_entities_from_news(news_data)
        print_result("从新闻 dict 抽取实体", len(entities) > 0,
                     f"抽取到 {len(entities)} 个实体, {len(relations)} 个关系")
    except Exception as e:
        print_result("从新闻 dict 抽取实体", False, str(e))


# ==================== 4. 测试 Neo4j 连接 ====================
def test_neo4j_connection():
    print_section("4. 测试 Neo4j 连接 (knowledge_graph)")

    try:
        from src.knowledge_graph import get_neo4j_connection, create_schema_constraints
        conn = get_neo4j_connection()
        print_result("导入并创建连接", True)
    except Exception as e:
        print_result("导入并创建连接", False, str(e))
        return False

    # 4.1 测试连接状态
    try:
        connected = conn.is_connected()
        print_result("Neo4j 连接状态", connected,
                     "已连接" if connected else "未连接（请检查 Docker 容器）")
        if not connected:
            print("       提示: 运行 start.bat 启动 Neo4j 容器")
            return False
    except Exception as e:
        print_result("Neo4j 连接状态", False, str(e))
        return False

    # 4.2 测试创建 Schema
    try:
        ok = create_schema_constraints()
        print_result("创建 Schema 约束", ok)
    except Exception as e:
        print_result("创建 Schema 约束", False, str(e))

    # 4.3 测试查询
    try:
        result = conn.execute_query("MATCH (n) RETURN count(n) as count")
        count = result[0]["count"] if result else 0
        print_result("执行 Cypher 查询", True, f"当前共有 {count} 个节点")
    except Exception as e:
        print_result("执行 Cypher 查询", False, str(e))

    return True


# ==================== 5. 测试 KGWriter ====================
def test_kg_writer():
    print_section("5. 测试知识图谱写入 (kg_builder.kg_writer)")

    try:
        from src.kg_builder.kg_writer import get_kg_writer
        writer = get_kg_writer()
        print_result("导入 KGWriter", True)
    except Exception as e:
        print_result("导入 KGWriter", False, str(e))
        return

    # 5.1 测试写入公司
    try:
        ok = writer.write_company({
            "code": "TEST001",
            "name": "测试科技公司",
            "sector": "科技",
            "market": "沪市",
            "market_cap": 100.5,
            "list_date": "2020-01-01",
            "description": "用于测试的公司节点"
        })
        print_result("写入公司节点", ok)
    except Exception as e:
        print_result("写入公司节点", False, str(e))

    # 5.2 测试写入行业
    try:
        ok = writer.write_sector({
            "name": "测试行业",
            "code": "TEST_SECTOR",
            "description": "用于测试的行业节点"
        })
        print_result("写入行业节点", ok)
    except Exception as e:
        print_result("写入行业节点", False, str(e))

    # 5.3 测试写入关系
    try:
        from src.knowledge_graph import get_neo4j_connection
        conn = get_neo4j_connection()
        conn.execute_write("""
            MATCH (c:Company {code: 'TEST001'})
            MATCH (s:Sector {name: '测试行业'})
            MERGE (c)-[:BELONGS_TO]->(s)
        """)
        print_result("写入关系 BELONGS_TO", True)
    except Exception as e:
        print_result("写入关系 BELONGS_TO", False, str(e))

    # 5.4 验证写入结果
    try:
        from src.knowledge_graph import get_neo4j_connection
        conn = get_neo4j_connection()
        result = conn.execute_query("""
            MATCH (c:Company {code: 'TEST001'})-[:BELONGS_TO]->(s:Sector)
            RETURN c.name as company, s.name as sector
        """)
        if result:
            print_result("验证写入结果", True,
                         f"{result[0]['company']} -> {result[0]['sector']}")
        else:
            print_result("验证写入结果", False, "未找到关系")
    except Exception as e:
        print_result("验证写入结果", False, str(e))


# ==================== 6. 清理测试数据 ====================
def cleanup_test_data():
    print_section("6. 清理测试数据")

    try:
        from src.knowledge_graph import get_neo4j_connection
        conn = get_neo4j_connection()
        if conn.is_connected():
            conn.execute_write("MATCH (c:Company {code: 'TEST001'}) DETACH DELETE c")
            conn.execute_write("MATCH (s:Sector {name: '测试行业'}) DETACH DELETE s")
            print_result("删除测试节点", True)
        else:
            print_result("删除测试节点", False, "Neo4j 未连接")
    except Exception as e:
        print_result("删除测试节点", False, str(e))

    # 清理测试输出文件
    for f in TEST_OUTPUT_DIR.iterdir():
        f.unlink()
    TEST_OUTPUT_DIR.rmdir()
    print_result("清理测试文件", True)


# ==================== 主入口 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  知识图谱模块功能测试")
    print("=" * 60)

    test_api_sources()
    test_news_crawler()
    test_entity_extractor()
    neo4j_ok = test_neo4j_connection()

    if neo4j_ok:
        test_kg_writer()
        cleanup_test_data()
    else:
        print("\n  [!] 跳过 Neo4j 写入测试（服务未启动）")

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)
