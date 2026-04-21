"""
锐思文本分析 API 模块 - 与 FinRAG 系统集成

提供中国上市公司财经文本、政府工作报告、美国上市公司报告、
财经资讯、房产信息、研究报告、股吧评论等文本数据的获取和检索。
"""

import json
import traceback
from typing import Optional, Dict, List, Any
from datetime import datetime


# =========================
# 配置
# =========================
RESSET_USERNAME = "sysu"
RESSET_PASSWORD = "sysu"

# 数据类型定义
CN_REPORT_TYPES = [
    "年度报告", "第一季度报告", "第二季度报告", "第三季度报告",
    "问询函及回复说明", "IPO招股说明书", "内部控制评价报告",
    "业绩说明会全文", "社会责任报告", "上市公司重大事项公告",
    "审计报告", "风险管理业务公告", "上市公司典型案例",
]

US_REPORT_TYPES = ["10K", "10Q", "424B"]

RESEARCH_TYPES = ["宏观分析", "行业分析", "证券市场研究", "公司研究", "期货研究", "晨会汇编"]

FORUM_TYPES = ["东方财富", "雪球"]

REAL_ESTATE_TYPES = [
    "中拍网_拍卖公告", "中拍网_竞买须知",
    "北交互联_拍卖公告", "北交互联_竞买须知",
    "工行融e购_拍卖公告", "工行融e购_竞买须知",
    "公拍网_拍卖公告", "公拍网_竞买须知",
    "京东拍卖_拍卖公告", "京东拍卖_竞买须知",
    "人民法院诉讼资产网_拍卖公告", "人民法院诉讼资产网_竞买须知",
]

# 行政区域代码（常用）
REGION_CODES = {
    "国务院": "100100",
    "北京市": "100101",
    "天津市": "100102",
    "河北省": "100103",
    "山西省": "100104",
    "内蒙古自治区": "100105",
    "辽宁省": "100106",
    "吉林省": "100107",
    "黑龙江省": "100108",
    "上海市": "100109",
    "江苏省": "100110",
    "浙江省": "100111",
    "安徽省": "100112",
    "福建省": "100113",
    "江西省": "100114",
    "山东省": "100115",
    "河南省": "100116",
    "湖北省": "100117",
    "湖南省": "100118",
    "广东省": "100119",
    "广西壮族自治区": "100120",
    "海南省": "100121",
    "重庆市": "100122",
    "四川省": "100123",
    "贵州省": "100124",
    "云南省": "100125",
    "西藏自治区": "100126",
    "陕西省": "100127",
    "甘肃省": "100128",
    "青海省": "100129",
    "宁夏回族自治区": "100130",
    "新疆维吾尔自治区": "100131",
}


# =========================
# 连接管理
# =========================
class RessetConnection:
    """锐思 API 连接管理器"""

    def __init__(self, username: str = None, password: str = None):
        self.username = username or RESSET_USERNAME
        self.password = password or RESSET_PASSWORD
        self._login_id = None
        self._reportdata = None
        self._available = None

    @property
    def is_available(self) -> bool:
        """检查锐思 API 是否可用"""
        if self._available is not None:
            return self._available
        try:
            self._ensure_login()
            self._available = self._login_id is not None
        except Exception:
            self._available = False
        return self._available

    def _ensure_login(self):
        """确保已登录"""
        if self._login_id is not None:
            return

        try:
            from resset.report import reportdata
            self._reportdata = reportdata
            self._login_id = reportdata.ressetLogin(self.username, self.password)
            print(f"[Resset] 登录成功, 用户ID: {self._login_id}")
        except ImportError:
            print("[Resset] 锐思 API 未安装。请执行: pip install https://rtas.resset.com/txtPath/resset-0.9.8-py3-none-any.whl")
            raise
        except Exception as e:
            print(f"[Resset] 登录失败: {e}")
            raise

    def get_permission(self) -> str:
        """获取账号当前剩余下载数量"""
        self._ensure_login()
        return self._reportdata.get_Permission(self._login_id)

    def get_content_data(
        self,
        code: str,
        content_type: str = "part",
        data_type: str = "年度报告",
        year: str = None,
    ) -> List[Dict]:
        """
        获取文本数据（统一接口）

        Args:
            code: 股票代码 / 行政区域代码 / 'None'
            content_type: 'part'（剔除表格图片）或 'all'（全文）
            data_type: 数据类型
            year: 报告年份

        Returns:
            文本数据列表
        """
        self._ensure_login()

        if year is None:
            year = str(datetime.now().year - 1)

        try:
            result = self._reportdata.get_Content_data(
                self._login_id, code, content_type, data_type, str(year)
            )
            return result if result else []
        except Exception as e:
            print(f"[Resset] 获取数据失败: {e}")
            return []

    def get_id_data(
        self,
        code: str,
        content_type: str = "part",
        data_type: str = "东方财富",
        year: str = None,
    ) -> List:
        """
        获取文本数据标题信息（用于大数据量场景，先获取 ID 再按 ID 获取内容）

        Args:
            code: 股票代码 / 'None'
            content_type: 'part' 或 'all'
            data_type: 数据类型
            year: 报告年份

        Returns:
            ID 和标题列表
        """
        self._ensure_login()

        if year is None:
            year = str(datetime.now().year - 1)

        try:
            result = self._reportdata.get_ID_data(
                self._login_id, code, content_type, data_type, str(year)
            )
            return result if result else []
        except Exception as e:
            print(f"[Resset] 获取 ID 数据失败: {e}")
            return []

    def get_content_by_id(
        self,
        data_id: str,
        content_type: str = "part",
        data_type: str = "东方财富",
        year: str = None,
    ) -> List[Dict]:
        """
        根据 ID 获取文本数据信息（大数据量场景的第二步）

        Args:
            data_id: 数据 ID
            content_type: 'part' 或 'all'
            data_type: 数据类型
            year: 报告年份

        Returns:
            文本数据列表
        """
        self._ensure_login()

        if year is None:
            year = str(datetime.now().year - 1)

        try:
            result = self._reportdata.get_ContentByID(
                self._login_id, data_id, content_type, data_type, str(year)
            )
            return result if result else []
        except Exception as e:
            print(f"[Resset] 根据 ID 获取数据失败: {e}")
            return []


# =========================
# 全局单例
# =========================
_resset_conn: Optional[RessetConnection] = None


def get_resset_connection(username: str = None, password: str = None) -> RessetConnection:
    """获取锐思 API 连接单例"""
    global _resset_conn
    if _resset_conn is None:
        _resset_conn = RessetConnection(username, password)
    return _resset_conn


def check_resset_available() -> tuple:
    """检查锐思 API 是否可用"""
    try:
        conn = get_resset_connection()
        available = conn.is_available
        if available:
            return True, "锐思文本分析 API 已连接"
        else:
            return False, "锐思 API 登录失败"
    except ImportError:
        return False, "锐思 API 未安装 (pip install resset)"
    except Exception as e:
        return False, f"锐思 API 不可用: {e}"


# =========================
# 便捷查询函数
# =========================
def get_cn_company_report(
    stock_code: str,
    data_type: str = "年度报告",
    year: str = None,
    content_type: str = "part",
) -> List[Dict]:
    """
    获取中国上市公司财经文本信息

    Args:
        stock_code: 股票代码，如 "000002"
        data_type: 数据类型，默认 "年度报告"
        year: 报告年份
        content_type: "part" 或 "all"

    Returns:
        文本数据列表
    """
    conn = get_resset_connection()
    return conn.get_content_data(stock_code, content_type, data_type, year)


def get_government_report(
    region_code: str = "100100",
    year: str = None,
    content_type: str = "part",
) -> List[Dict]:
    """
    获取政府工作文本信息

    Args:
        region_code: 行政区域代码，默认 "100100"（国务院）
        year: 报告年份
        content_type: "part" 或 "all"

    Returns:
        文本数据列表
    """
    conn = get_resset_connection()
    return conn.get_content_data(region_code, content_type, "政府工作报告", year)


def get_us_company_report(
    stock_code: str,
    data_type: str = "10K",
    year: str = None,
    content_type: str = "part",
) -> List[Dict]:
    """
    获取美国上市公司财经文本信息

    Args:
        stock_code: 美国股票代码，如 "AMZN"
        data_type: "10K" / "10Q" / "424B"
        year: 报告年份
        content_type: "part" 或 "all"

    Returns:
        文本数据列表
    """
    conn = get_resset_connection()
    return conn.get_content_data(stock_code, content_type, data_type, year)


def get_financial_news(
    year: str = None,
    content_type: str = "part",
) -> List[Dict]:
    """
    获取财经资讯文本信息

    Args:
        year: 报告年份 (2017-2023)
        content_type: "part" 或 "all"

    Returns:
        文本数据列表
    """
    conn = get_resset_connection()
    return conn.get_content_data("None", content_type, "新闻资讯", year)


def get_research_report(
    data_type: str = "公司研究",
    year: str = None,
    content_type: str = "part",
) -> List[Dict]:
    """
    获取研究报告文本信息

    Args:
        data_type: 研究类型
        year: 报告年份 (2017-2023)
        content_type: "part" 或 "all"

    Returns:
        文本数据列表
    """
    conn = get_resset_connection()
    return conn.get_content_data("None", content_type, data_type, year)


def get_forum_posts(
    data_type: str = "东方财富",
    year: str = None,
    content_type: str = "part",
    max_items: int = 20,
) -> List[Dict]:
    """
    获取股吧评论文本信息（大数据量，使用两步获取）

    Args:
        data_type: "东方财富" 或 "雪球"
        year: 报告年份 (2000-2023)
        content_type: "part" 或 "all"
        max_items: 最大获取条数

    Returns:
        文本数据列表
    """
    conn = get_resset_connection()

    # 第一步：获取 ID 列表
    id_list = conn.get_id_data("None", content_type, data_type, year)

    if not id_list:
        return []

    # 解析 ID 列表（格式为 "id_title" 的字符串数组）
    ids = []
    for item in id_list:
        if isinstance(item, str) and "_" in item:
            data_id = item.split("_")[0]
            ids.append(data_id)
        elif isinstance(item, dict) and "id" in item:
            ids.append(item["id"])

    # 限制数量
    ids = ids[:max_items]

    # 第二步：根据 ID 获取内容
    results = []
    for data_id in ids:
        content = conn.get_content_by_id(data_id, content_type, data_type, year)
        if content:
            results.extend(content)

    return results


def get_real_estate_info(
    data_type: str = "京东拍卖_拍卖公告",
    year: str = None,
    content_type: str = "part",
) -> List[Dict]:
    """
    获取房产文本信息

    Args:
        data_type: 房产数据类型
        year: 报告年份 (2017-2023)
        content_type: "part" 或 "all"

    Returns:
        文本数据列表
    """
    conn = get_resset_connection()
    return conn.get_content_data("None", content_type, data_type, year)


# =========================
# RAG 集成：格式化文本供检索
# =========================
def format_resset_content(data: List[Dict], data_type: str) -> str:
    """
    将锐思 API 返回的数据格式化为可读文本，供 RAG 系统使用

    Args:
        data: 锐思 API 返回的数据列表
        data_type: 数据类型名称

    Returns:
        格式化后的文本
    """
    if not data:
        return f"未获取到 {data_type} 相关数据。"

    parts = []
    for i, item in enumerate(data[:5], 1):  # 最多取5条
        title = item.get("title", "")
        if isinstance(title, list):
            title = title[0] if title else ""

        # 获取内容字段
        content = (
            item.get("part_content") or item.get("all_content") or
            item.get("Content") or item.get("content") or item.get("announcement") or ""
        )
        if isinstance(content, list):
            content = content[0] if content else ""

        # 截断过长内容
        if len(content) > 3000:
            content = content[:3000] + "...(内容已截断)"

        name = item.get("name", "")
        if isinstance(name, list):
            name = name[0] if name else ""

        code = item.get("code", "")
        if isinstance(code, list):
            code = code[0] if code else ""

        year = item.get("year", "")
        if isinstance(year, list):
            year = year[0] if year else ""

        release_time = item.get("releaseTime") or item.get("release_time") or ""
        if isinstance(release_time, list):
            release_time = release_time[0] if release_time else ""

        parts.append(
            f"### {data_type} - {title}\n"
            f"- 代码: {code}  名称: {name}  年份: {year}\n"
            f"- 发布时间: {release_time}\n\n"
            f"{content}"
        )

    return "\n\n---\n\n".join(parts)


def get_resset_context(question: str) -> str:
    """
    根据用户问题智能获取锐思数据并格式化为 RAG 上下文

    Args:
        question: 用户问题

    Returns:
        格式化的上下文字符串
    """
    question_lower = question.lower()

    # 判断数据类型
    data_type = None
    query_fn = None
    extra_kwargs = {}

    # 中国上市公司年报
    if any(kw in question_lower for kw in ["年报", "年度报告", "上市公司报告"]):
        data_type = "年度报告"
        query_fn = get_cn_company_report
    elif any(kw in question_lower for kw in ["季报", "第一季度", "第三季度"]):
        data_type = "第三季度报告"
        query_fn = get_cn_company_report
    elif any(kw in question_lower for kw in ["半年报", "第二季度", "中期报告"]):
        data_type = "第二季度报告"
        query_fn = get_cn_company_report
    # 政府工作报告
    elif any(kw in question_lower for kw in ["政府工作报告", "国务院报告", "政府报告"]):
        data_type = "政府工作报告"
        query_fn = get_government_report
    # 问询函
    elif any(kw in question_lower for kw in ["问询函", "回复说明"]):
        data_type = "问询函及回复说明"
        query_fn = get_cn_company_report
    # IPO 招股书
    elif any(kw in question_lower for kw in ["招股说明书", "ipo", "招股书"]):
        data_type = "IPO招股说明书"
        query_fn = get_cn_company_report
    # 美股报告
    elif any(kw in question_lower for kw in ["美股", "10k", "10q", "美国上市公司", "sec"]):
        data_type = "10K"
        query_fn = get_us_company_report
    # 财经资讯
    elif any(kw in question_lower for kw in ["财经资讯", "新闻资讯", "财经新闻"]):
        data_type = "新闻资讯"
        query_fn = get_financial_news
    # 研究报告
    elif any(kw in question_lower for kw in ["研究报告", "行业分析", "宏观分析", "公司研究", "券商研报"]):
        data_type = "公司研究"
        query_fn = get_research_report
        # 检查具体类型
        for rt in RESEARCH_TYPES:
            if rt in question:
                extra_kwargs["data_type"] = rt
                data_type = rt
                break
    # 股吧评论
    elif any(kw in question_lower for kw in ["股吧", "东方财富", "雪球", "股民评论", "散户"]):
        data_type = "东方财富"
        query_fn = get_forum_posts
        if "雪球" in question:
            extra_kwargs["data_type"] = "雪球"
            data_type = "雪球"
    # 房产信息
    elif any(kw in question_lower for kw in ["房产", "拍卖", "法拍", "司法拍卖"]):
        data_type = "京东拍卖_拍卖公告"
        query_fn = get_real_estate_info

    if data_type is None or query_fn is None:
        return ""

    try:
        # 提取年份
        import re
        year_match = re.search(r"(20\d{2})年?", question)
        year = year_match.group(1) if year_match else None

        # 提取股票代码
        code_match = re.search(r"(\d{6})", question)
        stock_code = code_match.group(1) if code_match else "000002"

        # 根据数据类型选择参数
        if data_type == "政府工作报告":
            result = query_fn(year=year)
        elif data_type in ["新闻资讯"]:
            result = query_fn(year=year)
        elif data_type in RESEARCH_TYPES + ["东方财富", "雪球"]:
            result = query_fn(year=year, **extra_kwargs)
        elif data_type in [rt for sub in [REAL_ESTATE_TYPES] for rt in sub] or "拍卖" in data_type:
            result = query_fn(year=year, **extra_kwargs)
        elif data_type in US_REPORT_TYPES:
            # 美股
            us_code_match = re.search(r"([A-Z]{2,5})", question.upper())
            us_code = us_code_match.group(1) if us_code_match else "AMZN"
            result = query_fn(stock_code=us_code, data_type=data_type, year=year)
        else:
            # 中国上市公司
            result = query_fn(stock_code=stock_code, data_type=data_type, year=year)

        if result:
            formatted = format_resset_content(result, data_type)
            return f"【锐思文本分析 - {data_type}】\n\n{formatted}"
        else:
            return f"【锐思文本分析】未找到 {data_type} 相关数据。"

    except Exception as e:
        return f"【锐思文本分析】查询失败: {e}"


def should_trigger_resset(user_input: str) -> bool:
    """检查是否应该触发锐思数据功能"""
    resset_keywords = [
        "年报", "季报", "年度报告", "季度报告", "问询函", "招股说明书",
        "政府工作报告", "政府报告", "ipo", "10k", "10q",
        "财经资讯", "财经新闻", "新闻资讯",
        "研究报告", "行业分析", "宏观分析", "公司研究", "券商研报",
        "股吧", "东方财富", "雪球", "股民评论",
        "房产", "拍卖", "法拍",
        "锐思", "resset",
        "内控报告", "社会责任报告", "审计报告",
        "业绩说明会",
    ]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in resset_keywords)
