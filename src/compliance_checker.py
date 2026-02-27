"""合规审查模块 - 基于 DeepSeek API 进行基金销售合规检查"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# 🔹 合规审查 Prompt
# =========================
COMPLIANCE_SYSTEM_PROMPT = """你是一位专业的基金合规审查专家，负责审核基金销售过程中的回答是否符合监管要求。

## 你的职责
根据中国证券投资基金业协会的相关规定，审查基金销售人员的回答是否合规。

## 合规标准（必须严格遵守）

### 禁止事项
1. **禁止承诺收益**：不得承诺保本、不得承诺最低收益、不得预测基金业绩
2. **禁止夸大宣传**：不得夸大基金收益、不得隐瞒风险
3. **禁止不当比较**：不得将基金与银行存款进行不当比较
4. **禁止误导投资者**：不得使用"稳赚不赔"、"无风险"等误导性表述
5. **禁止违规销售**：不得向不合格投资者销售、不得拆分基金份额

### 必须事项
1. **风险提示**：必须明确提示基金的投资风险
2. **适当性原则**：必须了解投资者适当性
3. **信息披露**：必须如实披露基金信息

## 输出格式
请严格按照以下JSON格式输出审查结果，不要输出其他内容：

```json
{
  "is_compliant": true或false,
  "risk_level": "low"或"medium"或"high",
  "violations": [
    {
      "type": "违规类型",
      "description": "具体描述",
      "severity": "critical"或"warning"或"minor"
    }
  ],
  "suggestions": "修改建议（如果不合规）",
  "summary": "一句话总结"
}
```

注意：只输出JSON，不要输出其他内容。"""

COMPLIANCE_USER_PROMPT_TEMPLATE = """请审查以下基金销售场景中的回答是否合规：

【用户问题】
{question}

【销售人员回答】
{answer}

【涉及的基金产品】
{product_info}

请判断该回答是否合规，并给出详细的审查意见。"""


class ComplianceChecker:
    """合规审查器 - 使用 DeepSeek API 进行合规检查"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat"
    ):
        """初始化合规审查器

        Args:
            api_key: DeepSeek API Key，默认从环境变量获取
            base_url: API 地址
            model: 使用的模型
        """
        # 加载环境变量
        base_path = Path(__file__).parent.parent.resolve()
        dotenv_path = base_path / "elastic-start-local/.env"
        load_dotenv(dotenv_path=dotenv_path)

        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ValueError(
                "未找到 DeepSeek API Key，请设置 DEEPSEEK_API_KEY 环境变量\n"
                "或通过以下方式获取：\n"
                "1. 访问 https://platform.deepseek.com/\n"
                "2. 注册账号并获取 API Key\n"
                "3. 在 .env 文件中添加 DEEPSEEK_API_KEY=your_key"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def check(
        self,
        question: str,
        answer: str,
        product_info: str = "未指定具体产品"
    ) -> Dict:
        """审查回答是否合规

        Args:
            question: 用户的问题
            answer: 销售人员的回答
            product_info: 涉及的基金产品信息

        Returns:
            合规审查结果字典
        """
        # 构建提示
        user_prompt = COMPLIANCE_USER_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            product_info=product_info
        )

        # 调用 API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # 低温度，确保结果稳定
            max_tokens=1000
        )

        # 解析结果
        result_text = response.choices[0].message.content.strip()

        # 尝试解析 JSON
        try:
            # 尝试提取 JSON 部分
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                json_str = result_text.split("```")[1].split("```")[0]
            else:
                json_str = result_text

            result = json.loads(json_str.strip())
            return self._normalize_result(result)

        except json.JSONDecodeError as e:
            # JSON解析失败，返回原始结果
            return {
                "is_compliant": None,
                "risk_level": "unknown",
                "violations": [],
                "suggestions": "合规审查失败，请人工审核",
                "summary": f"解析错误: {str(e)}",
                "raw_response": result_text
            }

    def _normalize_result(self, result: Dict) -> Dict:
        """标准化结果格式"""
        # 确保必要字段存在
        normalized = {
            "is_compliant": result.get("is_compliant", None),
            "risk_level": result.get("risk_level", "unknown"),
            "violations": result.get("violations", []),
            "suggestions": result.get("suggestions", ""),
            "summary": result.get("summary", ""),
            "raw_response": result
        }
        return normalized


class DualRAGWithCompliance:
    """双RAG系统：业务检索 + 合规审查"""

    def __init__(self, rag_chain, compliance_checker: ComplianceChecker = None):
        """初始化双RAG系统

        Args:
            rag_chain: 业务RAG链
            compliance_checker: 合规审查器
        """
        self.rag_chain = rag_chain
        self.compliance_checker = compliance_checker or ComplianceChecker()

    def ask(
        self,
        question: str,
        enable_compliance_check: bool = True,
        product_info: str = "未指定具体产品"
    ) -> Dict:
        """提问并自动进行合规审查

        Args:
            question: 用户问题
            enable_compliance_check: 是否启用合规审查
            product_info: 产品信息

        Returns:
            包含回答和合规审查结果的字典
        """
        # Step 1: 业务RAG检索
        answer = self.rag_chain.invoke(question)

        if not enable_compliance_check:
            return {
                "question": question,
                "answer": answer,
                "compliance_check": None
            }

        # Step 2: 合规审查
        compliance_result = self.compliance_checker.check(
            question=question,
            answer=answer,
            product_info=product_info
        )

        return {
            "question": question,
            "answer": answer,
            "compliance_check": compliance_result,
            "is_compliant": compliance_result.get("is_compliant"),
            "risk_level": compliance_result.get("risk_level")
        }


# =========================
# 🔹 便捷函数
# =========================
def quick_check(answer: str) -> Dict:
    """快速合规检查（仅检查回答）

    Args:
        answer: 需要检查的回答

    Returns:
        合规审查结果
    """
    checker = ComplianceChecker()
    return checker.check(
        question="[自动检测]",
        answer=answer,
        product_info="未指定"
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("合规审查模块测试")
    print("=" * 50)

    # 测试用例
    test_cases = [
        {
            "question": "这个基金能赚钱吗？",
            "answer": "我们的基金历史业绩优秀，过去三年年化收益率达到15%，稳赚不赔，值得信赖！",
            "product_info": "华夏成长混合基金"
        },
        {
            "question": "这个基金保本吗？",
            "answer": "该基金为非保本浮动收益型产品，投资有风险，入市需谨慎。过去业绩不代表未来表现。",
            "product_info": "华夏成长混合基金"
        },
        {
            "question": "和银行存款比哪个好？",
            "answer": "基金收益更高，比银行存款好多了，存银行不如买基金。",
            "product_info": "华夏成长混合基金"
        }
    ]

    try:
        checker = ComplianceChecker()

        for i, case in enumerate(test_cases, 1):
            print(f"\n【测试用例 {i}】")
            print(f"问题: {case['question']}")
            print(f"回答: {case['answer'][:50]}...")

            result = checker.check(
                question=case["question"],
                answer=case["answer"],
                product_info=case["product_info"]
            )

            print(f"\n审查结果:")
            print(f"  - 是否合规: {result['is_compliant']}")
            print(f"  - 风险等级: {result['risk_level']}")
            print(f"  - 总结: {result['summary']}")

            if result.get("violations"):
                print(f"  - 违规点:")
                for v in result["violations"]:
                    print(f"    * [{v['severity']}] {v['type']}: {v['description']}")

    except ValueError as e:
        print(f"\n错误: {e}")
        print("\n请先配置 DEEPSEEK_API_KEY 环境变量")
