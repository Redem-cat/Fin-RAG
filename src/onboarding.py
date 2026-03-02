"""
首次使用引导模块
通过对话框进行多步骤引导
"""
from typing import Dict, Any, Optional, List
from src.auth import (
    get_user_profile,
    update_user_profile,
    create_user_profile,
    needs_onboarding
)


# 引导对话步骤
ONBOARDING_STEPS: List[Dict[str, Any]] = [
    {
        "step": 0,
        "field": None,  # 欢迎步骤，不收集数据
        "prompt": "请回复任意内容开始",
        "message": """# 欢迎使用 FinRAG 智能投顾助手！

我是您的智能金融助手，可以帮助您：

- 📚 法规查询：回答您关于金融法规的问题
- 📊 投资顾问：提供市场数据分析和投资建议  
- ⚠️ 合规提示：帮助您了解投资风险

我可以查询实时金融数据（股票、基金、指数等），并结合知识库为您提供专业的金融问答服务。

请告诉我您的称呼，我该怎么称呼您？"""
    },
    {
        "step": 1,
        "field": "display_name",
        "prompt": "请告诉我您的称呼：",
        "message": "好的 {name}！请问您有什么投资经验？比如您是刚开始接触投资，还是已经有一定的投资经历了？"
    },
    {
        "step": 2,
        "field": "investment_experience", 
        "prompt": "请回复您的投资经验：",
        "message": "了解！最后请问您对哪些投资领域感兴趣？比如股票、基金、债券、黄金、ETF等，可以告诉我感兴趣的领域。"
    },
    {
        "step": 3,
        "field": "interested_areas",
        "prompt": "请回复您感兴趣的投资领域：",
        "message": None  # 最后一步不返回新消息
    }
]


def get_current_step(user_id: int) -> int:
    """获取用户当前引导步骤"""
    profile = get_user_profile(user_id)
    if not profile:
        return 0
    return profile.get("onboarding_step", 0)


def get_onboarding_message(user_id: int) -> Dict[str, Any]:
    """获取当前引导步骤的消息"""
    if not get_user_profile(user_id):
        create_user_profile(user_id)
    
    current_step = get_current_step(user_id)
    
    if current_step < 0:
        current_step = 0
    if current_step >= len(ONBOARDING_STEPS):
        current_step = len(ONBOARDING_STEPS) - 1
    
    step_config = ONBOARDING_STEPS[current_step]
    profile = get_user_profile(user_id)
    
    message = step_config.get("message", "")
    if message and "{name}" in message:
        name = profile.get("display_name", "用户") if profile else "用户"
        message = message.replace("{name}", name)
    
    return {
        "step": current_step,
        "total_steps": len(ONBOARDING_STEPS),
        "message": message,
        "prompt": step_config.get("prompt", ""),
        "is_complete": current_step >= len(ONBOARDING_STEPS) - 1,
        "next_step": current_step + 1 if current_step < len(ONBOARDING_STEPS) - 1 else None
    }


def process_onboarding_response(user_id: int, user_response: str) -> Dict[str, Any]:
    """处理用户对引导的回复"""
    current_step = get_current_step(user_id)
    step_config = ONBOARDING_STEPS[current_step]
    field = step_config.get("field")
    
    # 解析并保存用户回复
    if field:
        value = _parse_field_response(field, user_response)
        update_user_profile(user_id, **{field: value})
    
    next_step = current_step + 1
    
    if next_step >= len(ONBOARDING_STEPS):
        update_user_profile(
            user_id,
            onboarding_step=next_step,
            has_completed_onboarding=1
        )
        return {
            "is_complete": True,
            "message": "设置完成！您可以开始使用了。有什么我可以帮您的吗？"
        }
    
    update_user_profile(user_id, onboarding_step=next_step)
    return get_onboarding_message(user_id)


def _parse_field_response(field: str, response: str) -> str:
    """解析用户回复为对应字段值"""
    response = response.strip()
    
    if field == "display_name":
        return response.split("\n")[0].strip()[:50] or "用户"
    
    elif field == "investment_experience":
        if any(kw in response for kw in ["新手", "没有", "刚开始", "第一次"]):
            return "新手"
        elif any(kw in response for kw in ["1", "一年"]):
            return "初级"
        elif any(kw in response for kw in ["3", "三年", "几年"]):
            return "中级"
        elif any(kw in response for kw in ["5", "五年", "多年", "专业"]):
            return "高级"
        return "新手"
    
    elif field == "interested_areas":
        areas = []
        keywords = {
            "股票": "股票",
            "基金": "基金", 
            "债券": "债券",
            "黄金": "黄金",
            "期货": "期货",
            "ETF": "ETF"
        }
        for kw, value in keywords.items():
            if kw in response:
                areas.append(value)
        return ",".join(areas) if areas else "股票,基金"
    
    return response


def needs_onboarding(user_id: int) -> bool:
    """检查是否需要完成引导"""
    profile = get_user_profile(user_id)
    if not profile:
        return True
    return not profile.get("has_completed_onboarding", False)


def should_show_onboarding(user_id: int) -> bool:
    """检查是否应该显示引导消息"""
    return needs_onboarding(user_id)


def mark_onboarding_complete(user_id: int) -> bool:
    """标记引导完成"""
    return update_user_profile(
        user_id,
        has_completed_onboarding=1,
        onboarding_step=len(ONBOARDING_STEPS)
    )


def is_onboarding_complete(user_id: int) -> bool:
    """检查引导是否完成"""
    return not needs_onboarding(user_id)
