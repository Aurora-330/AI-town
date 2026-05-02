"""NPC好感度管理系统"""

import sys
import os

# 添加HelloAgents到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HelloAgents'))

from hello_agents import SimpleAgent, HelloAgentsLLM
from typing import Dict, Optional, Tuple, Literal
import json
import re
from pydantic import BaseModel, Field, ValidationError


class AffinityAnalysis(BaseModel):
    """好感度分析结果的固定结构"""

    should_change: bool
    change_amount: int = Field(ge=-15, le=10)
    reason: str = Field(min_length=1, max_length=10)
    sentiment: Literal["positive", "neutral", "negative"]

class RelationshipManager:
    """NPC好感度管理器
    
    功能:
    - 管理NPC与玩家的好感度 (0-100)
    - 使用LLM分析对话情感
    - 自动更新好感度
    - 提供好感度等级和修饰词
    """
    
    def __init__(self, llm: HelloAgentsLLM):
        """初始化好感度管理器
        
        Args:
            llm: HelloAgentsLLM实例
        """
        self.llm = llm
        
        # 存储每个NPC与玩家的好感度
        # 格式: {npc_name: {player_id: affinity_score}}
        self.affinity_scores: Dict[str, Dict[str, float]] = {}
        
        # 创建好感度分析Agent
        self.analyzer_agent = SimpleAgent(
            name="AffinityAnalyzer",
            llm=llm,
            system_prompt=self._create_analyzer_prompt()
        )
        
        print("💖 好感度管理系统已初始化")
    
    def _create_analyzer_prompt(self) -> str:
        """创建情感分析Agent的系统提示词"""
        return """你是一个情感分析专家,负责分析对话中的情感倾向,判断是否应该改变NPC对玩家的好感度。

【任务】
分析玩家与NPC的对话,判断是否应该改变好感度,以及改变的幅度。

【分析维度】
1. **玩家态度**: 友好/中立/不友好
2. **对话内容**: 积极/中立/消极
3. **互动质量**: 深入/一般/敷衍
4. **情感倾向**: 赞美/批评/中性

【重要区分】
- 玩家在表达自己的紧张、焦虑、害怕、压力、困惑、难过,不等于在攻击NPC。
- 玩家主动求助、请教建议、暴露脆弱状态,默认视为信任或中性互动,通常不应扣分。
- 只有当负面情绪明确指向NPC本人、NPC的能力、NPC的态度,或包含不耐烦、指责、羞辱、攻击时,才考虑负向扣分。
- “你觉得我该怎么办/先怎么说/能帮我想想吗” 这类求助句式,即使内容包含焦虑,也不应判为 negative。

【好感度变化规则】
- 赞美、感谢、请教: +3 到 +8
- 友好问候、正常交流: +1 到 +3
- 普通闲聊、中性话题: 0
- 批评、质疑、不耐烦: -3 到 -8
- 侮辱、攻击、恶意: -8 到 -15

【输出规则】
不要输出JSON,不要解释,不要补充说明。
只允许输出下面4行字段,字段名必须完全一致:
should_change=true/false
change_amount=整数
reason=简短原因
sentiment=positive/neutral/negative

【约束】
- should_change 只能是 true 或 false
- change_amount 必须是 -15 到 10 的整数
- reason 不超过10个字
- sentiment 只能是 positive / neutral / negative
- 如果 should_change=false, change_amount 必须是 0

【示例1】
should_change=true
change_amount=5
reason=友好问候
sentiment=positive

【示例2】
should_change=true
change_amount=-8
reason=批评工作
sentiment=negative

【示例3】
should_change=false
change_amount=0
reason=普通闲聊
sentiment=neutral

【示例4】
should_change=false
change_amount=0
reason=表达焦虑
sentiment=neutral

【示例5】
should_change=true
change_amount=3
reason=主动求助
sentiment=positive
"""
    
    def get_affinity(self, npc_name: str, player_id: str = "player") -> float:
        """获取好感度 (0-100)
        
        Args:
            npc_name: NPC名称
            player_id: 玩家ID
            
        Returns:
            好感度值 (0-100)
        """
        if npc_name not in self.affinity_scores:
            self.affinity_scores[npc_name] = {}
        
        if player_id not in self.affinity_scores[npc_name]:
            self.affinity_scores[npc_name][player_id] = 50.0  # 初始好感度50
        
        return self.affinity_scores[npc_name][player_id]
    
    def set_affinity(self, npc_name: str, affinity: float, player_id: str = "player"):
        """设置好感度
        
        Args:
            npc_name: NPC名称
            affinity: 好感度值 (0-100)
            player_id: 玩家ID
        """
        if npc_name not in self.affinity_scores:
            self.affinity_scores[npc_name] = {}
        
        # 限制在0-100范围内
        affinity = max(0.0, min(100.0, affinity))
        self.affinity_scores[npc_name][player_id] = affinity
    
    def analyze_and_update_affinity(
        self,
        npc_name: str,
        player_message: str,
        npc_response: str,
        player_id: str = "player"
    ) -> Dict:
        """分析对话并更新好感度
        
        Args:
            npc_name: NPC名称
            player_message: 玩家消息
            npc_response: NPC回复
            player_id: 玩家ID
            
        Returns:
            分析结果字典
        """
        prompt = self._build_analysis_prompt(npc_name, player_message, npc_response)
        
        try:
            analysis = self._analyze_with_retry(
                npc_name=npc_name,
                player_message=player_message,
                npc_response=npc_response,
                prompt=prompt
            )
            analysis = self._apply_affinity_guardrails(
                player_message=player_message,
                npc_response=npc_response,
                analysis=analysis
            )
            
            if analysis["should_change"]:
                # 更新好感度
                current_affinity = self.get_affinity(npc_name, player_id)
                new_affinity = current_affinity + analysis["change_amount"]
                new_affinity = max(0.0, min(100.0, new_affinity))  # 限制在0-100

                self.set_affinity(npc_name, new_affinity, player_id)

                # 获取好感度等级
                old_level = self.get_affinity_level(current_affinity)
                new_level = self.get_affinity_level(new_affinity)

                # 注意: 打印日志已移到agents.py中,避免重复输出

                return {
                    "changed": True,
                    "old_affinity": current_affinity,
                    "new_affinity": new_affinity,
                    "change_amount": analysis["change_amount"],
                    "reason": analysis["reason"],
                    "sentiment": analysis.get("sentiment", "neutral"),
                    "old_level": old_level,
                    "new_level": new_level
                }
            else:
                return {
                    "changed": False,
                    "affinity": self.get_affinity(npc_name, player_id),
                    "reason": analysis["reason"],
                    "sentiment": analysis.get("sentiment", "neutral")
                }
        
        except Exception as e:
            print(f"❌ 好感度分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "changed": False,
                "affinity": self.get_affinity(npc_name, player_id),
                "reason": "分析失败",
                "sentiment": "neutral"
            }

    def _build_analysis_prompt(self, npc_name: str, player_message: str, npc_response: str) -> str:
        """构建首轮分析提示"""
        return f"""请分析以下对话:

玩家: {player_message}
{npc_name}: {npc_response}

请严格按4行字段输出结果。
"""

    def _build_retry_prompt(self, prompt: str, invalid_response: str, error_message: str) -> str:
        """构建重试提示"""
        return f"""{prompt}

你上一次的输出不合法,错误原因是: {error_message}

上一次输出:
{invalid_response}

请重新输出,只保留以下4行,不要加任何解释:
should_change=true/false
change_amount=整数
reason=简短原因
sentiment=positive/neutral/negative
"""

    def _analyze_with_retry(
        self,
        npc_name: str,
        player_message: str,
        npc_response: str,
        prompt: str
    ) -> Dict:
        """执行分析、校验和一次自动重试"""
        response = self.analyzer_agent.run(prompt)

        try:
            return self._parse_analysis(response)
        except Exception as first_error:
            print(f"⚠️ 首次好感度结构化输出失败: {first_error}. 原始响应: {response[:120]}")

            retry_prompt = self._build_retry_prompt(prompt, response, str(first_error))
            retry_response = self.analyzer_agent.run(retry_prompt)

            try:
                return self._parse_analysis(retry_response)
            except Exception as retry_error:
                print(f"⚠️ 重试后仍解析失败: {retry_error}. 原始响应: {retry_response[:120]}")
                return self._rule_based_fallback(npc_name, player_message, npc_response)

    def _parse_analysis(self, response: str) -> Dict:
        """解析并校验分析结果"""
        candidate = self._extract_analysis_candidate(response)
        validated = AffinityAnalysis.model_validate(candidate)

        normalized = validated.model_dump()
        normalized["reason"] = normalized["reason"][:10]

        if not normalized["should_change"]:
            normalized["change_amount"] = 0
            if normalized["sentiment"] != "neutral":
                normalized["sentiment"] = "neutral"

        return normalized

    def _apply_affinity_guardrails(
        self,
        player_message: str,
        npc_response: str,
        analysis: Dict,
    ) -> Dict:
        """对明显的求助/脆弱表达做轻量护栏，避免误判成负向扣分。"""
        normalized = dict(analysis)
        player_text = (player_message or "").strip()

        if not player_text:
            return normalized

        if normalized.get("change_amount", 0) >= 0:
            return normalized

        if self._contains_hostility(player_text):
            return normalized

        if self._is_support_seeking_message(player_text):
            normalized["should_change"] = False
            normalized["change_amount"] = 0
            normalized["reason"] = "表达焦虑"
            normalized["sentiment"] = "neutral"

        return normalized

    def _extract_analysis_candidate(self, response: str) -> Dict:
        """从模型输出中提取字段值"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        extracted: Dict[str, str] = {}
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            kv_match = re.match(r"^(should_change|change_amount|reason|sentiment)\s*[:=]\s*(.+)$", stripped, re.IGNORECASE)
            if kv_match:
                key = kv_match.group(1).lower()
                value = kv_match.group(2).strip().strip('"').strip("'")
                extracted[key] = value

        if not extracted:
            raise ValueError("未提取到任何结构化字段")

        parsed: Dict[str, object] = {}
        if "should_change" in extracted:
            parsed["should_change"] = extracted["should_change"].lower() == "true"
        if "change_amount" in extracted:
            parsed["change_amount"] = int(re.search(r"-?\d+", extracted["change_amount"]).group(0))
        if "reason" in extracted:
            parsed["reason"] = extracted["reason"]
        if "sentiment" in extracted:
            parsed["sentiment"] = extracted["sentiment"].lower()

        return parsed

    def _rule_based_fallback(self, npc_name: str, player_message: str, npc_response: str) -> Dict:
        """当结构化输出持续失败时,使用轻量规则兜底"""
        text = f"{player_message} {npc_response}"

        positive_rules = [
            ("谢谢", 3, "表达感谢"),
            ("感谢", 3, "表达感谢"),
            ("厉害", 4, "赞美能力"),
            ("有帮助", 3, "认可帮助"),
            ("愿意按你的步骤", 4, "愿意配合"),
            ("继续推进", 3, "积极配合"),
            ("反馈结果", 3, "愿意反馈"),
            ("请教", 3, "主动请教"),
            ("高兴认识", 3, "友好问候"),
        ]
        negative_rules = [
            ("太烦", -5, "表达不耐"),
            ("一点用都没有", -6, "否定价值"),
            ("别再", -4, "拒绝沟通"),
            ("没意义", -5, "否定话题"),
            ("矫情", -5, "贬低感受"),
            ("闭嘴", -8, "言语攻击"),
            ("差不多就行", -4, "轻蔑否定"),
            ("别总", -3, "明显不耐"),
            ("没用", -4, "否定贡献"),
        ]

        if self._is_support_seeking_message(player_message) and not self._contains_hostility(player_message):
            return {
                "should_change": False,
                "change_amount": 0,
                "reason": "表达焦虑",
                "sentiment": "neutral"
            }

        score = 0
        reason = "普通闲聊"
        sentiment = "neutral"

        for keyword, delta, label in positive_rules:
            if keyword in text:
                score += delta
                reason = label
                sentiment = "positive"

        for keyword, delta, label in negative_rules:
            if keyword in text:
                score += delta
                reason = label
                sentiment = "negative"

        score = max(-15, min(10, score))
        if score == 0:
            return {
                "should_change": False,
                "change_amount": 0,
                "reason": "普通闲聊",
                "sentiment": "neutral"
            }

        return {
            "should_change": True,
            "change_amount": score,
            "reason": reason[:10],
            "sentiment": sentiment
        }

    def _is_support_seeking_message(self, text: str) -> bool:
        """识别以求助、请教、脆弱表达为主的输入。"""
        lowered = (text or "").strip()
        if not lowered:
            return False

        distress_markers = [
            "紧张", "焦虑", "害怕", "担心", "压力", "难受", "难过",
            "慌", "不安", "困惑", "崩溃", "心里很乱", "有点乱"
        ]
        help_markers = [
            "你觉得我该", "我该怎么", "先怎么说", "怎么办", "能帮我", "帮我想想",
            "可以怎么", "该怎么开口", "请教", "想听听你的建议", "你怎么看"
        ]

        return (
            any(marker in lowered for marker in distress_markers)
            and any(marker in lowered for marker in help_markers)
        )

    def _contains_hostility(self, text: str) -> bool:
        """识别直接指向NPC的攻击、不耐烦或羞辱。"""
        lowered = (text or "").strip()
        hostile_markers = [
            "你很烦", "你真烦", "闭嘴", "没用", "一点用都没有", "别再",
            "你懂什么", "你不懂", "别装", "少来", "废话", "矫情"
        ]
        return any(marker in lowered for marker in hostile_markers)
    
    def get_affinity_level(self, affinity: float) -> str:
        """获取好感度等级
        
        Args:
            affinity: 好感度值 (0-100)
            
        Returns:
            好感度等级名称
        """
        if affinity >= 80:
            return "挚友"
        elif affinity >= 60:
            return "亲密"
        elif affinity >= 40:
            return "友好"
        elif affinity >= 20:
            return "熟悉"
        else:
            return "陌生"
    
    def get_affinity_modifier(self, affinity: float) -> str:
        """获取好感度修饰词 (用于调整对话风格)
        
        Args:
            affinity: 好感度值 (0-100)
            
        Returns:
            对话风格修饰词
        """
        if affinity >= 80:
            return "非常热情友好,像老朋友一样亲切,愿意分享私人话题"
        elif affinity >= 60:
            return "友好热情,愿意多聊,会主动关心对方"
        elif affinity >= 40:
            return "礼貌友善,正常交流,保持专业"
        elif affinity >= 20:
            return "礼貌但略显生疏,回答简洁"
        else:
            return "冷淡疏离,不太愿意多说,回答简短"
    
    def get_all_affinities(self, player_id: str = "player") -> Dict[str, Dict]:
        """获取所有NPC的好感度信息
        
        Args:
            player_id: 玩家ID
            
        Returns:
            所有NPC的好感度信息
        """
        result = {}
        for npc_name in self.affinity_scores:
            affinity = self.get_affinity(npc_name, player_id)
            result[npc_name] = {
                "affinity": affinity,
                "level": self.get_affinity_level(affinity),
                "modifier": self.get_affinity_modifier(affinity)
            }
        return result
