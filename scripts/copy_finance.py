"""复制金融相关法规到新目录"""
import os
import shutil
from pathlib import Path

FINANCE_KEYWORDS = [
    '证券', '股票', '期货', '基金', '银行', '保险', '信托', '资产',
    '金融', '投资', '融资', '信贷', '支付', '清算', '托管',
    '上市', 'IPO', '注册制', '退市', '回购', '减持',
    '内幕', '操纵', '违规', '处罚', '欺诈',
    '理财', '债券', 'ABS', 'REITs',
    '央行', '银保监', '证监会', '合规', '反洗钱',
]

DATA_ROOT = Path(r"f:\py\Bishe\langchain-ollama-elasticsearch\data")
TARGET = DATA_ROOT / "01_金融法规"

# 确保目标目录存在
TARGET.mkdir(parents=True, exist_ok=True)

# 扫描源目录
SOURCE_DIRS = [
    DATA_ROOT / "行政法规",
    DATA_ROOT / "部门规章",
    DATA_ROOT / "经济法",
    DATA_ROOT / "司法解释",
]

copied_count = 0
for source_dir in SOURCE_DIRS:
    if not source_dir.exists():
        continue
    print(f"扫描目录: {source_dir.name}")
    for f in source_dir.rglob("*.md"):
        if f.name.startswith("_"):
            continue
        if any(kw in f.stem for kw in FINANCE_KEYWORDS):
            target_file = TARGET / f.name
            shutil.copy2(f, target_file)
            copied_count += 1
            print(f"  + {f.name}")

print(f"\n完成! 共复制 {copied_count} 个文件到 {TARGET}")
