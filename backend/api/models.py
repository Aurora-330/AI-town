"""数据模型定义"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """单个NPC对话请求"""
    npc_name: str = Field(..., description="NPC名称")
    message: str = Field(..., description="玩家消息")
    execution_mode: Literal["auto", "static_coordinator", "controlled_react"] = Field(
        default="auto",
        description="执行模式开关：auto / static_coordinator / controlled_react",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "npc_name": "风泠",
                "message": "你好,你在做什么?",
                "execution_mode": "auto",
            }
        }

class ChatResponse(BaseModel):
    """单个NPC对话响应"""
    npc_name: str = Field(..., description="NPC名称")
    npc_title: str = Field(..., description="NPC职位")
    message: str = Field(..., description="NPC回复")
    success: bool = Field(default=True, description="是否成功")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="时间戳")
    execution_mode: str = Field(default="auto", description="实际采用的执行模式")
    query_mode: str = Field(default="default", description="查询模式")
    react_activated: bool = Field(default=False, description="是否激活 controlled react loop")
    react_activation_rule: str = Field(default="", description="react 激活命中的规则")
    react_activation_reason: str = Field(default="", description="react 激活/回退的原因说明")
    react_step_count: int = Field(default=0, description="react loop 实际步数")
    tool_call_count: int = Field(default=0, description="本轮工具调用次数")
    input_tokens_est: int = Field(default=0, description="最终 prompt 输入 token 估算")
    latency_ms: int = Field(default=0, description="本轮处理耗时（毫秒）")
    error_type: str = Field(default="", description="错误类型标记；用于评测上下文超窗等异常")
    prompt_budget_debug: Dict[str, object] = Field(default_factory=dict, description="上下文治理观测数据")
    retrieval_metrics: Dict[str, object] = Field(default_factory=dict, description="记忆/知识检索观测数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "npc_name": "风泠",
                "npc_title": "Python工程师",
                "message": "你好!我正在写代码,调试一个多智能体系统的bug。",
                "success": True,
                "execution_mode": "auto",
                "query_mode": "default",
                "react_activated": False,
                "react_activation_rule": "",
                "react_activation_reason": "",
                "react_step_count": 0,
                "tool_call_count": 0,
                "input_tokens_est": 0,
                "latency_ms": 0,
                "error_type": "",
                "prompt_budget_debug": {},
                "retrieval_metrics": {},
            }
        }

class NPCInfo(BaseModel):
    """NPC信息"""
    name: str = Field(..., description="NPC名称")
    title: str = Field(..., description="NPC职位")
    location: str = Field(..., description="NPC位置")
    activity: str = Field(..., description="当前活动")
    available: bool = Field(default=True, description="是否可对话")

class NPCStatusResponse(BaseModel):
    """NPC状态响应"""
    dialogues: Dict[str, str] = Field(..., description="NPC当前对话内容")
    last_update: Optional[datetime] = Field(None, description="上次更新时间")
    next_update_in: int = Field(..., description="下次更新倒计时(秒)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dialogues": {
                    "风泠": "终于把这个bug修复了,测试通过!",
                    "郁米": "下周的产品评审会需要准备一下资料。",
                    "顾辰": "这个界面的配色方案还需要优化一下。"
                },
                "last_update": "2024-01-15T10:30:00",
                "next_update_in": 25
            }
        }

class NPCListResponse(BaseModel):
    """NPC列表响应"""
    npcs: List[NPCInfo] = Field(..., description="NPC列表")
    total: int = Field(..., description="NPC总数")


class DelegatePreviewRequest(BaseModel):
    """LangGraph delegate 最小骨架调试请求"""
    active_speaker: str = Field(..., description="当前前台说话的 NPC")
    message: str = Field(..., description="玩家消息")
    player_id: str = Field(default="player", description="玩家 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "active_speaker": "郁米",
                "message": "上个月那个项目的核心指标档案，你可以帮我翻一下吗？",
                "player_id": "player",
            }
        }


class DelegatePreviewResponse(BaseModel):
    """LangGraph delegate 最小骨架调试响应"""
    active_speaker: str = Field(..., description="当前前台说话的 NPC")
    message: str = Field(..., description="玩家原始消息")
    orchestration_mode: str = Field(default="direct", description="本次编排模式：direct / delegate")
    intent_type: str = Field(default="", description="路由识别出的意图类型")
    query_mode: str = Field(default="default", description="查询模式")
    needs_delegate: bool = Field(default=False, description="是否需要后台委托")
    delegate_to: str = Field(default="", description="后台委托目标 NPC")
    delegate_task: str = Field(default="", description="委托任务描述")
    final_style_owner: str = Field(default="", description="最终输出风格归属角色")
    observation_cards: List[Dict[str, object]] = Field(default_factory=list, description="节点间传递的短 observation cards")
    node_trace: List[Dict[str, object]] = Field(default_factory=list, description="节点流转轨迹")
    final_answer: str = Field(default="", description="前台角色最终组织后的输出")
    langgraph_available: bool = Field(default=False, description="当前环境是否已安装 LangGraph 并可直接用 StateGraph 运行")

    class Config:
        json_schema_extra = {
            "example": {
                "active_speaker": "郁米",
                "message": "上个月那个项目的核心指标档案，你可以帮我翻一下吗？",
                "orchestration_mode": "delegate",
                "intent_type": "archive_lookup",
                "query_mode": "recall",
                "needs_delegate": True,
                "delegate_to": "风泠",
                "delegate_task": "请帮忙查历史档案/指标事实：上个月那个项目的核心指标档案，你可以帮我翻一下吗？",
                "final_style_owner": "郁米",
                "observation_cards": [],
                "node_trace": [],
                "final_answer": "风泠刚刚帮我查到啦，……",
                "langgraph_available": False,
            }
        }


class MultiChatRequest(BaseModel):
    """多角色协作对话请求"""
    message: str = Field(..., description="玩家消息")
    mode: Literal["auto", "reactive_duo", "parallel_b", "serial_a"] = Field(
        default="auto",
        description="编排模式：auto / reactive_duo / parallel_b。serial_a 仅保留为兼容别名，会映射到 reactive_duo。",
    )
    player_id: str = Field(default="player", description="玩家 ID")
    selected_agents: List[str] = Field(default_factory=list, description="可选：手动指定参与角色，留空则由编排器决定")
    return_intermediate: bool = Field(default=True, description="是否返回中间节点输出")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "我最近工作压力好大，感觉什么都做不好，想离职。",
                "mode": "reactive_duo",
                "player_id": "player",
                "selected_agents": [],
                "return_intermediate": True,
            }
        }


class MultiChatAgentOutput(BaseModel):
    """单个角色的中间发言"""
    npc_name: str = Field(..., description="NPC 名称")
    stage: str = Field(..., description="在图中的阶段名")
    message: str = Field(..., description="该角色输出")
    query_mode: str = Field(default="default", description="该节点实际识别的 query mode")
    tool_call_count: int = Field(default=0, description="该节点工具调用次数")
    latency_ms: int = Field(default=0, description="该节点耗时（毫秒）")


class MultiChatResponse(BaseModel):
    """多角色协作对话响应"""
    mode: str = Field(..., description="实际采用的剧本模式")
    selected_agents: List[str] = Field(default_factory=list, description="参与协作的角色")
    execution_order: List[str] = Field(default_factory=list, description="执行顺序")
    aggregation_strategy: str = Field(default="", description="汇总策略")
    final_answer: str = Field(default="", description="最终输出给前端的答案")
    intermediate_outputs: List[MultiChatAgentOutput] = Field(default_factory=list, description="中间角色输出")
    node_trace: List[Dict[str, object]] = Field(default_factory=list, description="节点流转轨迹")
    success: bool = Field(default=True, description="是否成功")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="时间戳")
    langgraph_available: bool = Field(default=False, description="当前环境是否以 LangGraph StateGraph 运行")

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "reactive_duo",
                "selected_agents": ["郁米", "顾辰"],
                "execution_order": ["郁米", "顾辰"],
                "aggregation_strategy": "support_handoff_to_guchen",
                "final_answer": "先别急着把结论一步跳到离职。你现在更需要先缓一下，再把压力和长期适配度拆开看。",
                "intermediate_outputs": [
                    {
                        "npc_name": "郁米",
                        "stage": "reactive_duo_primary_郁米",
                        "message": "你先别一个人扛着，我在听。",
                        "query_mode": "default",
                        "tool_call_count": 1,
                        "latency_ms": 620,
                    }
                ],
                "node_trace": [],
                "success": True,
                "langgraph_available": True,
            }
        }
