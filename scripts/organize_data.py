"""
数据整理脚本
将法律法规分门别类，保留与项目相关的，移走无关的
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

# 金融相关关键词
FINANCE_KEYWORDS = [
    # 金融核心
    "证券", "股票", "期货", "基金", "银行", "保险", "信托", "资产管理",
    "金融", "投资", "融资", "信贷", "贷款", "存款", "利息", "利率",
    "支付", "清算", "结算", "托管", "保证金", "杠杆", "配资",
    # 具体法规
    "证券法", "基金法", "银行法", "保险法", "信托法", "公司法",
    "上市公司", " IPO", "注册制", "退市", "减持", "回购",
    # 金融产品
    "ETF", "LOF", "FOF", "QDII", "REITs", "ABS", "REPO",
    # 监管
    "证监会", "银保监会", "央行", "金融监管", "合规", "反洗钱",
    # 违规违法
    "内幕交易", "操纵市场", "虚假陈述", "欺诈", "违规", "处罚",
]

# 投资案例关键词
CASE_KEYWORDS = [
    "投资", "理财", "收益", "亏损", "案例", "分析", "报告",
    "基金", "股票", "债券", "理财产品",
]

# 政策宏观关键词
POLICY_KEYWORDS = [
    "政策", "通知", "指导意见", "决定", "规定", "办法",
    "国务院", "财政部", "发改委", "央行", "银保监会",
    "LPR", "利率", "货币政策", "财政政策",
]


def classify_file(filename: str) -> str:
    """根据文件名判断应该放到哪个目录"""
    name_lower = filename.lower()
    
    # 检查是否匹配金融关键词
    finance_score = sum(1 for kw in FINANCE_KEYWORDS if kw in name_lower)
    
    # 检查是否匹配案例关键词
    case_score = sum(1 for kw in CASE_KEYWORDS if kw in name_lower)
    
    # 检查是否匹配政策关键词
    policy_score = sum(1 for kw in POLICY_KEYWORDS if kw in name_lower)
    
    # 优先级判断
    if finance_score > 0:
        # 进一步细分
        if any(kw in name_lower for kw in ["证券", "股票", "期货", "上市公司", "注册制"]):
            return "01_金融法规/证券期货"
        elif any(kw in name_lower for kw in ["基金", "资管", "理财"]):
            return "01_金融法规/基金资管"
        elif any(kw in name_lower for kw in ["银行", "保险", "信托"]):
            return "01_金融法规/银行保险"
        elif any(kw in name_lower for kw in ["支付", "清算", "结算", "托管"]):
            return "01_金融法规/支付金融"
        return "01_金融法规"
    
    if case_score > 1:  # 需要多个关键词匹配
        return "02_投资案例"
    
    if policy_score > 1:
        return "03_政策法规/宏观政策"
    
    # 民法商法相关
    if any(kw in name_lower for kw in ["合同", "公司", "合伙", "侵权", "民事", "商法"]):
        return "04_民商法"
    
    return "05_待清理"


def organize_folder(source_folder: str, dry_run: bool = True):
    """整理一个文件夹"""
    source_path = Path(source_folder)
    data_root = Path("f:/py/Bishe/langchain-ollama-elasticsearch/data")
    
    stats = {
        "moved": [],
        "total": 0,
    }
    
    # 遍历所有md文件
    for file_path in source_path.rglob("*.md"):
        if file_path.name.startswith("_"):
            continue
            
        stats["total"] += 1
        target_dir = classify_file(file_path.stem)
        target_path = data_root / target_dir / file_path.name
        
        if dry_run:
            print(f"【{file_path.parent.name}】 -> 【{target_dir}】: {file_path.name}")
            stats["moved"].append((str(file_path), str(target_path)))
        else:
            # 创建目标目录
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # 复制文件
            shutil.copy2(file_path, target_path)
            stats["moved"].append((str(file_path), str(target_path)))
            
    return stats


def organize_all(data_root: str = "f:/py/Bishe/langchain-ollama-elasticsearch/data", dry_run: bool = True):
    """整理所有数据"""
    data_path = Path(data_root)
    
    # 要整理的源目录（排除新创建的目录）
    source_dirs = [
        "行政法规",
        "部门规章", 
        "经济法",
        "司法解释",
        "行政法",
        "社会法",
        "刑法",
        "宪法相关法",
        "案例",
    ]
    
    all_stats = {"total": 0, "moved": []}
    
    for dir_name in source_dirs:
        dir_path = data_path / dir_name
        if dir_path.exists():
            print(f"\n{'='*50}")
            print(f"正在整理: {dir_name}")
            print(f"{'='*50}")
            stats = organize_folder(str(dir_path), dry_run)
            all_stats["total"] += stats["total"]
            all_stats["moved"].extend(stats["moved"])
    
    print(f"\n\n{'='*50}")
    print(f"总计: {all_stats['total']} 个文件")
    print(f"将移动到新目录")
    
    return all_stats


if __name__ == "__main__":
    import sys
    
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        dry_run = False
        print("!!! 正式执行模式！将实际移动文件")
    else:
        print(">>> 预览模式，使用 --execute 参数实际执行")
    
    organize_all(dry_run=dry_run)
