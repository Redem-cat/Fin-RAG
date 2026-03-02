import os
import shutil
from pathlib import Path

# 项目根目录
root = Path(r"F:\py\Bishe\langchain-ollama-elasticsearch\data")

# 要删除的目录和文件
delete_list = [
    "DLC",
    "其他",
    "宪法",
    "宪法相关法",
    "刑法",
    "行政法",
    "行政法规",
    "部门规章",
    "经济法",
    "社会法",
    "诉讼法",
    "民法典",
    "诉讼与非诉讼程序法",
    "华夏成长混合基金.md",
    "users.db"
]

print("开始删除...")
for item in delete_list:
    path = root / item
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"[OK] 已删除目录: {item}")
            else:
                path.unlink()
                print(f"[OK] 已删除文件: {item}")
        except Exception as e:
            print(f"[X] 删除失败 {item}: {e}")
    else:
        print(f"[-] 不存在: {item}")

print("\n删除完成!")
print("\n保留的数据目录:")
for item in root.iterdir():
    if item.is_dir():
        count = len(list(item.rglob("*.md")))
        print(f"  [DIR] {item.name}/ ({count} md文件)")
    else:
        print(f"  [FILE] {item.name}")
