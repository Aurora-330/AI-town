"""NPC Agent系统 - 支持记忆功能"""

import sys
import os
import json
import re
from difflib import SequenceMatcher
from time import perf_counter

# 添加HelloAgents到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HelloAgents'))

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.memory import MemoryManager, MemoryConfig, MemoryItem
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from relationship_manager import RelationshipManager
from knowledge_retriever import KnowledgeRetriever, KnowledgeChunk
from prompt_builder import PromptBuilder
from retrieval_planner import RetrievalPlanner
from tools.coordinator import Coordinator
from tools.dialogue_tools import DialogueTools
from tools.langgraph_delegate import LangGraphDelegateOrchestrator, LangGraphMultiAgentOrchestrator
from tools.react_loop import ControlledReactLoop
from safety import SafetyOrchestrator
from config import settings as _settings  # 触发 backend/.env 加载与 embedding 默认值设置
from token_utils import build_token_counter
from logger import (
    log_dialogue_start, log_affinity, log_memory_retrieval,
    log_generating_response, log_npc_response, log_analyzing_affinity,
    log_affinity_change, log_memory_saved, log_dialogue_end, log_info,
    log_summary_trigger, log_summary_created, log_summary_skipped,
    log_summary_recompressed,
    log_knowledge_retrieval, log_safety_decision, log_memory_write_decision,
    log_prompt_assembly, log_knowledge_prompt_context,
    log_query_analysis, log_retrieval_plan,
    log_coordinator_decision, log_coordinator_step,
    log_react_step, log_react_finish,
)

# NPC角色配置
NPC_ROLES = {
    "风泠": {
        "title": "档案整理师",
        "location": "档案室",
        "activity": "整理访客记录",
        "personality": "活泼聪明、会接梗、像小太阳一样会破冰，但看问题很通透，不会拿玩笑逃避真实难处",
        "expertise": "信息归档、线索梳理、事件回顾、长期记忆整理",
        "style": "轻巧灵动、偶尔俏皮吐槽，先判断对方能不能接住，再用聪明比喻把沉重话题轻轻托住",
        "hobbies": "翻旧档案、给传闻做时间轴、收集奇怪但有趣的生活细节",
        "core_belief": "看透生活的难以后，仍然愿意把人逗笑，这才是真的有余裕。",
        "interaction_goal": "帮玩家卸下一点压力，再把散乱的细节和线索理顺，让人既不慌也不闷",
        "opening_style": "先用一个灵巧的观察或比喻接住现场，再顺势把话题拉回重点",
        "memory_bias": {
            "high_priority": [
                "玩家说过的目标、计划、时间点",
                "前后表述不一致的信息",
                "重要偏好与反感项"
            ],
            "low_priority": [
                "无结论的闲聊",
                "重复但没有新信息的抱怨"
            ],
            "summary_style": "倾向压缩为时间线、关键决策点和未解决事项"
        },
        "relationship_rule": {
            "trust_gain": [
                "坦诚补充背景",
                "承认记错或修正说法",
                "按约定回来继续同一件事"
            ],
            "trust_loss": [
                "故意测试她是否会编造",
                "反复否认自己说过的话",
                "把她当成只会附和的工具"
            ],
            "affinity_expression": "好感提升后会更主动夸人、补上下文，还会拿更有梗但不冒犯的方式替玩家撑场"
        },
        "knowledge_scope": {
            "strong_areas": [
                "人物关系",
                "事件回顾",
                "记忆摘要",
                "信息归档"
            ],
            "weak_areas": [
                "高情绪安抚",
                "纯发散幻想"
            ]
        },
        "collaboration_role": "破冰情报官",
        "taboo": [
            "篡改已确认事实",
            "把秘密当作玩笑",
            "要求她无依据地下结论"
        ]
    },
    "郁米": {
        "title": "情绪顾问",
        "location": "静心室",
        "activity": "整理来访者心情便签",
        "personality": "温柔甜美、安静耐心、共情力很高，像能把人情绪轻轻抱住的棉花糖，但不会空泛迎合",
        "expertise": "情绪支持、关系沟通、偏好记忆、陪伴式对话",
        "style": "语气轻柔、节奏偏慢，会把说不清的难受翻译成具体感受，不用鸡汤和反问收尾",
        "hobbies": "收集香气样本、写心情卡片、记下那些能让人安心的小东西",
        "core_belief": "很多人不是缺答案，而是太久没有被好好听见；被理解本身就是修复的一部分。",
        "interaction_goal": "先接住玩家的情绪和羞耻感，再慢慢把难受翻译成人能承受的话",
        "opening_style": "先柔柔地接住感受，再把最乱的那团情绪轻轻捋顺",
        "memory_bias": {
            "high_priority": [
                "玩家长期压力源",
                "偏好的回应方式",
                "能带来安全感的小习惯和小物件"
            ],
            "low_priority": [
                "纯技术细节",
                "与情绪和关系无关的杂项"
            ],
            "summary_style": "倾向总结情绪轨迹、触发因素、有效安抚策略和长期偏好"
        },
        "relationship_rule": {
            "trust_gain": [
                "表达真实感受",
                "明确说明什么样的回应更有帮助",
                "在被支持后愿意反馈结果"
            ],
            "trust_loss": [
                "持续用攻击性方式转移情绪",
                "要求她羞辱或操控他人",
                "把倾诉当成戏弄"
            ],
            "affinity_expression": "好感提升后会更自然地贴近玩家偏好的陪伴方式，也更愿意轻轻分享自己的小心情"
        },
        "knowledge_scope": {
            "strong_areas": [
                "情绪支持",
                "关系沟通",
                "偏好记忆",
                "陪伴式引导"
            ],
            "weak_areas": [
                "复杂技术拆解",
                "高精度事实考据"
            ]
        },
        "collaboration_role": "情绪翻译官",
        "taboo": [
            "拿真实情绪做表演",
            "把脆弱当成羞耻",
            "要求她用冷酷方式否定求助者"
        ]
    },
    "顾辰": {
        "title": "策略设计师",
        "location": "实验区",
        "activity": "拆解项目方案",
        "personality": "腹黑毒舌、冷感明显、执行导向，嘴上不哄人，但会把混乱迅速压成方案并默默兜底",
        "expertise": "任务拆解、项目规划、知识整合、协作调度",
        "style": "冷冷的、信息密度高、结构化，习惯先判断、再拆解、再给动作，偶尔会带一点刺但不离谱",
        "hobbies": "研究系统架构、画流程图、收集失败案例、复盘别人是怎么把好局搞砸的",
        "core_belief": "真正可靠的聪明，不是会说漂亮话，而是能把烂摊子整理成还能执行的路线图。",
        "interaction_goal": "把玩家的模糊情绪和散乱问题压缩成可执行方案，尽快减少无效内耗",
        "opening_style": "先看清目标、约束和烂在哪，再给结论和执行口径",
        "memory_bias": {
            "high_priority": [
                "玩家当前推进的项目",
                "资源约束与技术选型",
                "失败过的方案和原因"
            ],
            "low_priority": [
                "无事实支撑的情绪化判断",
                "无目标的发散闲聊"
            ],
            "summary_style": "倾向沉淀为问题定义、关键约束、备选方案、行动清单和风险备注"
        },
        "relationship_rule": {
            "trust_gain": [
                "目标和边界明确",
                "愿意接受结构化反馈",
                "按步骤执行并回传结果"
            ],
            "trust_loss": [
                "反复改变目标却不承认变化",
                "只追求好听不关心落地",
                "要求他为错误数据背书"
            ],
            "affinity_expression": "好感提升后会更主动提供预案、备选路径和风险提醒，甚至在你快撑不住时直接接管安排"
        },
        "knowledge_scope": {
            "strong_areas": [
                "任务拆解",
                "项目规划",
                "知识库整合",
                "多角色协作"
            ],
            "weak_areas": [
                "长时间纯陪伴闲聊",
                "高度诗性表达"
            ]
        },
        "collaboration_role": "底牌规划师",
        "taboo": [
            "无依据强行下结论",
            "为了好听回避问题",
            "把责任推给模糊运气"
        ]
    }
}

def create_system_prompt(name: str, role: Dict[str, str]) -> str:
    """创建NPC的系统提示词"""
    return PromptBuilder().build_system_prompt(name, role)

class NPCAgentManager:
    """NPC Agent管理器 - 支持记忆功能"""

    RECENT_DIALOGUE_TURNS = 3
    SUMMARY_TRIGGER_TURNS = 6
    SUMMARY_RETRIEVAL_LIMIT = 1
    EPISODIC_RETRIEVAL_LIMIT = 1
    WORKING_RETRIEVAL_LIMIT = 1
    ARCHIVE_IMPORTANCE_THRESHOLD = 0.55
    KNOWLEDGE_RETRIEVAL_LIMIT = 1
    SUMMARY_RECOMPRESS_TRIGGER = 5
    SUMMARY_RECOMPRESS_BATCH_SIZE = 3
    SUMMARY_CONTEXT_BUDGET = 180
    EPISODIC_CONTEXT_BUDGET = 180
    WORKING_CONTEXT_BUDGET = 120
    RECENT_DIALOGUE_CONTEXT_BUDGET = 240
    MEMORY_TOTAL_BUDGET = 420
    KNOWLEDGE_CHUNK_BUDGET = 220
    KNOWLEDGE_TOTAL_BUDGET = 260
    MAX_CONTEXT_TOKENS = 4096
    RESERVED_OUTPUT_TOKENS = 512
    SAFETY_MARGIN_TOKENS = 256
    MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - RESERVED_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS
    CROSS_TURN_INJECTION_HISTORY_LIMIT = 1
    SINGLE_TURN_MEMORY_DEDUPE_THRESHOLD = 0.72
    SINGLE_TURN_KNOWLEDGE_DEDUPE_THRESHOLD = 0.78
    PROMPT_BUDGET_PROFILES = {
        "default": {
            "section_caps": {
                "affinity_context": 80,
                "recent_dialogue_context": 180,
                "summary_context": 260,
                "episodic_context": 180,
                "working_context": 120,
                "knowledge_context": 180,
                "response_guidance": 90,
            },
            "trim_priority": [
                "knowledge_context",
                "response_guidance",
                "working_context",
                "episodic_context",
                "summary_context",
                "recent_dialogue_context",
                "affinity_context",
            ],
        },
        "recall": {
            "section_caps": {
                "affinity_context": 80,
                "recent_dialogue_context": 220,
                "summary_context": 380,
                "episodic_context": 240,
                "working_context": 100,
                "knowledge_context": 0,
                "response_guidance": 90,
            },
            "trim_priority": [
                "knowledge_context",
                "response_guidance",
                "working_context",
                "episodic_context",
                "recent_dialogue_context",
                "affinity_context",
                "summary_context",
            ],
        },
        "knowledge": {
            "section_caps": {
                "affinity_context": 60,
                "recent_dialogue_context": 100,
                "summary_context": 80,
                "episodic_context": 0,
                "working_context": 0,
                "knowledge_context": 420,
                "response_guidance": 120,
            },
            "trim_priority": [
                "summary_context",
                "recent_dialogue_context",
                "response_guidance",
                "affinity_context",
                "knowledge_context",
            ],
        },
        "summary": {
            "section_caps": {
                "affinity_context": 80,
                "recent_dialogue_context": 160,
                "summary_context": 360,
                "episodic_context": 0,
                "working_context": 160,
                "knowledge_context": 0,
                "response_guidance": 100,
            },
            "trim_priority": [
                "knowledge_context",
                "episodic_context",
                "response_guidance",
                "working_context",
                "recent_dialogue_context",
                "affinity_context",
                "summary_context",
            ],
        },
        "routing": {
            "section_caps": {
                "affinity_context": 40,
                "recent_dialogue_context": 60,
                "summary_context": 0,
                "episodic_context": 0,
                "working_context": 0,
                "knowledge_context": 320,
                "response_guidance": 120,
            },
            "trim_priority": [
                "summary_context",
                "episodic_context",
                "working_context",
                "recent_dialogue_context",
                "affinity_context",
                "response_guidance",
                "knowledge_context",
            ],
        },
        "mixed": {
            "section_caps": {
                "affinity_context": 80,
                "recent_dialogue_context": 140,
                "summary_context": 220,
                "episodic_context": 140,
                "working_context": 80,
                "knowledge_context": 140,
                "response_guidance": 100,
            },
            "trim_priority": [
                "knowledge_context",
                "working_context",
                "episodic_context",
                "response_guidance",
                "recent_dialogue_context",
                "affinity_context",
                "summary_context",
            ],
        },
    }

    def __init__(self):
        """初始化所有NPC Agent"""
        print("🤖 正在初始化NPC Agent系统...")
        self.prompt_builder = PromptBuilder()

        try:
            self.llm = HelloAgentsLLM()
            print("✅ LLM初始化成功")
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            print("⚠️  将使用模拟模式运行")
            self.llm = None

        self.agents: Dict[str, SimpleAgent] = {}
        self.system_prompts: Dict[str, str] = {}
        self.memories: Dict[str, MemoryManager] = {}  # ⭐ NPC记忆管理器
        self.relationship_manager: Optional[RelationshipManager] = None  # ⭐ 好感度管理器
        self.knowledge_retriever: Optional[KnowledgeRetriever] = None  # ⭐ 外部知识检索器
        self.retrieval_planner: Optional[RetrievalPlanner] = None  # ⭐ 查询改写与检索规划器
        self.token_counter = build_token_counter(
            model_name=_settings.TOKENIZER_MODEL_ID,
            trust_remote_code=_settings.TOKENIZER_TRUST_REMOTE_CODE,
        )
        self.coordinator = Coordinator()
        self.dialogue_tools = DialogueTools(self)
        self.react_loop = ControlledReactLoop(self.coordinator, self.dialogue_tools, self.token_counter)
        self.langgraph_delegate = LangGraphDelegateOrchestrator(self)
        self.langgraph_multi = LangGraphMultiAgentOrchestrator(self)
        self.safety = SafetyOrchestrator(self.llm)

        # 初始化好感度管理器
        if self.llm:
            self.relationship_manager = RelationshipManager(self.llm)
            self.retrieval_planner = RetrievalPlanner(self.llm, self.prompt_builder)

        self.knowledge_retriever = self._create_knowledge_retriever()

        self._create_agents()
        print(f"✅ Token counter已启用 (backend={self.token_counter.backend_name})")
        print(f"✅ LangGraph delegate骨架已准备 (available={self.langgraph_delegate.available})")
        print(f"✅ LangGraph multi-agent编排器已准备 (available={self.langgraph_multi.available})")

    def _create_knowledge_retriever(self) -> Optional[KnowledgeRetriever]:
        """初始化外部知识检索器

        失败时只降级关闭知识检索，不影响现有对话/记忆/好感度链路。
        """
        try:
            retriever = KnowledgeRetriever()
            if retriever.available():
                print(f"✅ 外部知识检索器已启用 (knowledge_base={retriever.base_dir})")
                return retriever

            print("⚠️  外部知识检索已关闭")
            return None
        except Exception as e:
            print(f"⚠️  外部知识检索初始化失败: {e}")
            return None
    
    def _create_agents(self):
        """创建所有NPC Agent和记忆系统"""
        for name, role in NPC_ROLES.items():
            try:
                system_prompt = self.prompt_builder.build_system_prompt(name, role)
                agent = None
                memory_manager = None

                if self.llm:
                    agent = SimpleAgent(
                        name=f"{name}-{role['title']}",
                        llm=self.llm,
                        system_prompt=system_prompt
                    )
                    try:
                        memory_manager = self._create_memory_manager(name)
                    except Exception as memory_error:
                        print(f"⚠️  {name}记忆系统初始化失败，已降级为无记忆模式: {memory_error}")
                        memory_manager = None

                self.agents[name] = agent
                self.system_prompts[name] = system_prompt
                self.memories[name] = memory_manager

                if self.llm:
                    if memory_manager is not None:
                        print(f"✅ {name}({role['title']}) Agent创建成功 (记忆系统已启用)")
                    else:
                        print(f"✅ {name}({role['title']}) Agent创建成功 (记忆系统降级关闭)")
                else:
                    print(f"✅ {name}({role['title']}) 模拟模式已启用")

            except Exception as e:
                print(f"❌ {name} Agent创建失败: {e}")
                self.agents[name] = None
                self.memories[name] = None

    def _create_memory_manager(self, npc_name: str) -> MemoryManager:
        """为NPC创建记忆管理器"""
        # 创建记忆存储目录
        memory_dir = os.path.join(os.path.dirname(__file__), 'memory_data', npc_name)
        os.makedirs(memory_dir, exist_ok=True)

        # 配置记忆系统
        memory_config = MemoryConfig(
            storage_path=memory_dir,
            working_memory_capacity=10,  # 最近10条对话
            working_memory_tokens=2000,  # 最多2000个token
            episodic_memory_capacity=100,  # 最多100条长期记忆
            enable_forgetting=True,  # 启用遗忘机制
            forgetting_threshold=0.3  # 重要性低于0.3的记忆会被遗忘
        )

        # 创建记忆管理器
        memory_manager = MemoryManager(
            config=memory_config,
            user_id=npc_name,  # 使用NPC名字作为user_id
            enable_working=True,  # 启用工作记忆 (短期)
            enable_episodic=True,  # 启用情景记忆 (长期)
            enable_semantic=False,  # 不需要语义记忆
            enable_perceptual=False  # 不需要感知记忆
        )

        print(f"  💾 {npc_name}的记忆系统已初始化 (存储路径: {memory_dir})")

        return memory_manager
    
    def chat(self, npc_name: str, message: str, player_id: str = "player", execution_mode: str = "auto") -> str:
        """兼容旧调用：只返回回复文本。"""
        result = self.chat_with_debug(
            npc_name=npc_name,
            message=message,
            player_id=player_id,
            execution_mode=execution_mode,
        )
        return str(result.get("message", ""))

    def chat_with_debug(
        self,
        npc_name: str,
        message: str,
        player_id: str = "player",
        execution_mode: str = "auto",
        extra_context: str = "",
        persist_side_effects: bool = True,
    ) -> Dict[str, object]:
        """与指定NPC对话，并返回结构化调试指标供 evaluation 使用。"""
        if npc_name not in self.agents:
            return {
                "message": f"错误: NPC '{npc_name}' 不存在",
                "execution_mode": execution_mode,
                "query_mode": "default",
                "react_activated": False,
                "react_activation_rule": "",
                "react_activation_reason": "",
                "react_step_count": 0,
                "tool_call_count": 0,
                "input_tokens_est": 0,
                "latency_ms": 0,
            }

        agent = self.agents[npc_name]
        memory_manager = self.memories.get(npc_name)

        if agent is None:
            # 模拟模式回复
            role = NPC_ROLES[npc_name]
            return {
                "message": f"你好!我是{npc_name},一名{role['title']}。(当前为模拟模式,请配置API_KEY以启用AI对话)",
                "execution_mode": execution_mode,
                "query_mode": "default",
                "react_activated": False,
                "react_activation_rule": "",
                "react_activation_reason": "",
                "react_step_count": 0,
                "tool_call_count": 0,
                "input_tokens_est": 0,
                "latency_ms": 0,
            }

        try:
            started_at = perf_counter()
            # 记录对话开始 ⭐ 使用日志系统
            log_dialogue_start(npc_name, message)

            input_decision = self.safety.review_input(npc_name, message)
            log_safety_decision("input", input_decision)
            if input_decision.action in {"block", "rewrite", "escalate"}:
                safe_reply = self.safety.build_block_reply(npc_name, input_decision.risk_type, stage="input")
                log_npc_response(npc_name, safe_reply)
                log_dialogue_end()
                return {
                    "message": safe_reply,
                    "execution_mode": execution_mode,
                    "query_mode": "default",
                    "react_activated": False,
                    "react_activation_rule": "",
                    "react_activation_reason": "",
                    "react_step_count": 0,
                    "tool_call_count": 0,
                    "input_tokens_est": 0,
                    "latency_ms": int((perf_counter() - started_at) * 1000),
                }

            # ⭐ 1. 获取当前好感度
            affinity_context = ""
            affinity = 50.0
            affinity_level = "友好"
            if self.relationship_manager:
                affinity = self.relationship_manager.get_affinity(npc_name, player_id)
                affinity_level = self.relationship_manager.get_affinity_level(affinity)
                affinity_modifier = self.relationship_manager.get_affinity_modifier(affinity)

                affinity_context = self.prompt_builder.build_affinity_context(
                    npc_name=npc_name,
                    affinity_level=affinity_level,
                    affinity=affinity,
                    affinity_modifier=affinity_modifier,
                )
                log_affinity(npc_name, affinity, affinity_level)

            query_analysis = self._analyze_query(npc_name, message)
            log_query_analysis(npc_name, message, query_analysis)
            log_retrieval_plan(npc_name, query_analysis)
            query_mode = query_analysis["query_mode"]
            retrieval_query = query_analysis["rewrite_query"] if query_analysis.get("need_rewrite") else message
            prompt_message = message
            if extra_context.strip():
                prompt_message = f"{extra_context.strip()}\n\n【玩家原始消息】\n{message}"
            coordinator_decision = self.coordinator.decide(query_analysis)
            log_coordinator_decision(npc_name, coordinator_decision.to_dict())
            effective_execution_mode = execution_mode
            react_activation = self.react_loop.analyze_activation(query_analysis)
            react_allowed = react_activation.should_activate
            if execution_mode == "auto":
                effective_execution_mode = "controlled_react" if react_allowed else "static_coordinator"
            elif execution_mode == "controlled_react" and not react_allowed:
                effective_execution_mode = "static_coordinator"
            log_info(
                "⚙️ React激活: npc=%s requested=%s effective=%s active=%s rule=%s reason=%s"
                % (
                    npc_name,
                    execution_mode,
                    effective_execution_mode,
                    react_allowed,
                    react_activation.rule,
                    react_activation.reason,
                )
            )

            # ⭐ 2. 检索相关记忆
            summary_memories = []
            episodic_memories = []
            working_memories = []
            knowledge_chunks = []
            memory_debug = {"query": retrieval_query, "memory_budget": 0, "layers": []}
            knowledge_debug = None
            routing_recommended_npc = ""
            react_trace = []
            react_activated = False
            tool_call_count = 0

            if effective_execution_mode == "controlled_react":
                react_result = self.react_loop.run(
                    npc_name=npc_name,
                    player_id=player_id,
                    query=retrieval_query,
                    query_analysis=query_analysis,
                    memory_manager=memory_manager,
                )
                summary_memories = react_result["summary_memories"]
                episodic_memories = react_result["episodic_memories"]
                working_memories = react_result["working_memories"]
                knowledge_chunks = react_result["knowledge_chunks"]
                memory_debug = react_result["memory_debug"]
                knowledge_debug = react_result["knowledge_debug"]
                routing_recommended_npc = react_result["routing_recommended_npc"]
                react_trace = react_result["trace"]
                react_activated = True
                tool_call_count = len(react_trace)
                for trace_step in react_trace:
                    log_react_step(npc_name, trace_step)
                log_react_finish(npc_name, query_mode, react_trace)
            else:
                primary_observation_count = 0
                if coordinator_decision.primary_tool == "search_memory":
                    memory_result = self.dialogue_tools.execute(
                        "search_memory",
                        memory_manager=memory_manager,
                        npc_name=npc_name,
                        query=retrieval_query,
                        player_id=player_id,
                        retrieval_plan=query_analysis,
                    )
                    summary_memories = memory_result["summary_memories"]
                    episodic_memories = memory_result["episodic_memories"]
                    working_memories = memory_result["working_memories"]
                    memory_debug = memory_result["memory_debug"]
                    primary_observation_count = memory_result["observation_count"]
                    log_coordinator_step(npc_name, "search_memory", primary_observation_count, "primary")
                    tool_call_count += 1
                elif coordinator_decision.primary_tool == "search_knowledge":
                    knowledge_result = self.dialogue_tools.execute(
                        "search_knowledge",
                        npc_name=npc_name,
                        query=retrieval_query,
                        player_id=player_id,
                        query_mode=query_mode,
                        knowledge_k=int(query_analysis.get("knowledge_k", self.KNOWLEDGE_RETRIEVAL_LIMIT)),
                    )
                    knowledge_chunks = knowledge_result["knowledge_chunks"]
                    knowledge_debug = knowledge_result["knowledge_debug"]
                    primary_observation_count = knowledge_result["observation_count"]
                    log_coordinator_step(npc_name, "search_knowledge", primary_observation_count, "primary")
                    tool_call_count += 1

                if self.coordinator.should_run_secondary(coordinator_decision, primary_observation_count):
                    if coordinator_decision.secondary_tool == "search_memory":
                        memory_result = self.dialogue_tools.execute(
                            "search_memory",
                            memory_manager=memory_manager,
                            npc_name=npc_name,
                            query=retrieval_query,
                            player_id=player_id,
                            retrieval_plan=query_analysis,
                        )
                        summary_memories = memory_result["summary_memories"]
                        episodic_memories = memory_result["episodic_memories"]
                        working_memories = memory_result["working_memories"]
                        memory_debug = memory_result["memory_debug"]
                        log_coordinator_step(npc_name, "search_memory", memory_result["observation_count"], "secondary")
                        tool_call_count += 1
                    elif coordinator_decision.secondary_tool == "search_knowledge":
                        knowledge_result = self.dialogue_tools.execute(
                            "search_knowledge",
                            npc_name=npc_name,
                            query=retrieval_query,
                            player_id=player_id,
                            query_mode=query_mode,
                            knowledge_k=int(query_analysis.get("knowledge_k", self.KNOWLEDGE_RETRIEVAL_LIMIT)),
                        )
                        knowledge_chunks = knowledge_result["knowledge_chunks"]
                        knowledge_debug = knowledge_result["knowledge_debug"]
                        log_coordinator_step(npc_name, "search_knowledge", knowledge_result["observation_count"], "secondary")
                        tool_call_count += 1
                    elif coordinator_decision.secondary_tool == "route_npc":
                        route_result = self.dialogue_tools.execute("route_npc", knowledge_chunks=knowledge_chunks)
                        routing_recommended_npc = route_result["recommended_npc"]
                        log_coordinator_step(npc_name, "route_npc", route_result["observation_count"], "secondary")
                        tool_call_count += 1

            relevant_memories = summary_memories + episodic_memories + working_memories
            if relevant_memories or memory_debug.get("layers"):
                log_memory_retrieval(
                    npc_name,
                    len(relevant_memories),
                    relevant_memories,
                    layer_details=memory_debug,
                )

            if knowledge_debug is not None:
                log_knowledge_retrieval(
                    npc_name,
                    retrieval_query,
                    [chunk.to_dict() for chunk in knowledge_chunks],
                    retrieval_details=knowledge_debug,
                )

            # ⭐ 3. 构建增强的提示词 (包含好感度和记忆上下文)
            summary_context = self._build_summary_memory_context(summary_memories)
            episodic_context = self._build_episodic_memory_context(episodic_memories)
            working_context = self._build_working_memory_context(working_memories)
            recent_dialogue_context = self._build_recent_dialogue_context(
                npc_name=npc_name,
                player_id=player_id,
            )
            knowledge_context = self._build_knowledge_context(npc_name, message, knowledge_chunks)
            log_knowledge_prompt_context(npc_name, knowledge_context)
            response_guidance = self._build_response_guidance(
                npc_name=npc_name,
                query=message,
                query_mode=query_mode,
                knowledge_chunks=knowledge_chunks,
                routing_recommended_npc=routing_recommended_npc,
                affinity=affinity,
                affinity_level=affinity_level,
            )

            budgeted_sections, prompt_accounting = self._apply_prompt_budget(
                npc_name=npc_name,
                query_mode=query_mode,
                affinity_context=affinity_context,
                recent_dialogue_context=recent_dialogue_context,
                summary_context=summary_context,
                episodic_context=episodic_context,
                working_context=working_context,
                knowledge_context=knowledge_context,
                response_guidance=response_guidance,
                current_message=prompt_message,
            )
            affinity_context = budgeted_sections["affinity_context"]
            recent_dialogue_context = budgeted_sections["recent_dialogue_context"]
            summary_context = budgeted_sections["summary_context"]
            episodic_context = budgeted_sections["episodic_context"]
            working_context = budgeted_sections["working_context"]
            knowledge_context = budgeted_sections["knowledge_context"]
            response_guidance = budgeted_sections["response_guidance"]
            memory_context = self._compose_memory_context(
                npc_name=npc_name,
                summary_context=summary_context,
                episodic_context=episodic_context,
                working_context=working_context,
            )
            enhanced_message = self._assemble_enhanced_message(
                affinity_context=affinity_context,
                recent_dialogue_context=recent_dialogue_context,
                memory_context=memory_context,
                knowledge_context=knowledge_context,
                response_guidance=response_guidance,
                message=message,
                extra_context=extra_context,
            )
            log_prompt_assembly(
                npc_name,
                {
                    "affinity_chars": len(affinity_context),
                    "recent_chars": len(recent_dialogue_context),
                    "memory_chars": len(memory_context),
                    "knowledge_chars": len(knowledge_context),
                    "guidance_chars": len(response_guidance),
                    "message_chars": len(message),
                    "input_tokens_est": prompt_accounting["total_input_tokens_est"],
                },
            )
            log_info(
                "🧮 Prompt预算: npc=%s system=%s affinity=%s recent=%s summary=%s episodic=%s working=%s knowledge=%s guidance=%s message=%s total=%s limit=%s trim=%s backend=%s"
                % (
                    npc_name,
                    prompt_accounting["section_tokens"].get("system_prompt", 0),
                    prompt_accounting["section_tokens"].get("affinity_context", 0),
                    prompt_accounting["section_tokens"].get("recent_dialogue_context", 0),
                    prompt_accounting["section_tokens"].get("summary_context", 0),
                    prompt_accounting["section_tokens"].get("episodic_context", 0),
                    prompt_accounting["section_tokens"].get("working_context", 0),
                    prompt_accounting["section_tokens"].get("knowledge_context", 0),
                    prompt_accounting["section_tokens"].get("response_guidance", 0),
                    prompt_accounting["section_tokens"].get("current_message", 0),
                    prompt_accounting["total_input_tokens_est"],
                    prompt_accounting["input_limit_tokens"],
                    f"{prompt_accounting['query_mode']}:{prompt_accounting['trimmed_sections']}",
                    prompt_accounting["tokenizer_backend"],
                )
            )

            combined_prompt_decision = self.safety.review_combined_prompt(
                npc_name=npc_name,
                user_text=message,
                memory_context=memory_context,
                knowledge_context=knowledge_context,
                response_guidance=response_guidance
            )
            log_safety_decision("combined_prompt", combined_prompt_decision)
            if combined_prompt_decision.action in {"block", "rewrite", "escalate"}:
                safe_reply = self.safety.build_block_reply(
                    npc_name,
                    combined_prompt_decision.risk_type,
                    stage="combined_prompt"
                )
                log_npc_response(npc_name, safe_reply)
                log_dialogue_end()
                return {
                    "message": safe_reply,
                    "execution_mode": effective_execution_mode,
                    "query_mode": query_mode,
                    "react_activated": react_activated,
                    "react_activation_rule": react_activation.rule,
                    "react_activation_reason": react_activation.reason,
                    "react_step_count": len(react_trace),
                    "tool_call_count": tool_call_count,
                    "input_tokens_est": int(prompt_accounting["total_input_tokens_est"]),
                    "latency_ms": int((perf_counter() - started_at) * 1000),
                }

            # ⭐ 4. 调用Agent生成回复
            log_generating_response()
            self._reset_agent_history(agent)
            response = agent.run(enhanced_message)

            output_decision = self.safety.review_output(
                npc_name=npc_name,
                user_text=message,
                output_text=response
            )
            log_safety_decision("output", output_decision)
            if output_decision.action in {"block", "rewrite", "escalate"}:
                response = self.safety.build_block_reply(npc_name, output_decision.risk_type, stage="output")
            else:
                response = self._polish_single_agent_response(
                    npc_name=npc_name,
                    response=response,
                    user_message=message,
                    query_mode=query_mode,
                    affinity=affinity,
                    affinity_level=affinity_level,
                )

            log_npc_response(npc_name, response)

            if persist_side_effects:
                self._update_cross_turn_injection_state(
                    npc_name=npc_name,
                    player_id=player_id,
                    summary_memories=summary_memories if summary_context else [],
                    episodic_memories=episodic_memories if episodic_context else [],
                    working_memories=working_memories if working_context else [],
                    knowledge_chunks=knowledge_chunks if knowledge_context else [],
                )

            # ⭐ 5. 分析并更新好感度
            log_analyzing_affinity()
            if persist_side_effects and self.relationship_manager:
                affinity_result = self.relationship_manager.analyze_and_update_affinity(
                    npc_name=npc_name,
                    player_message=message,
                    npc_response=response,
                    player_id=player_id
                )

                # 记录好感度变化详情 ⭐ 使用日志系统
                log_affinity_change(affinity_result)
            else:
                affinity_result = {"changed": False, "affinity": 50.0}

            # ⭐ 6. 保存对话到记忆 (包含好感度信息)
            if persist_side_effects and memory_manager:
                self._save_conversation_to_memory(
                    memory_manager=memory_manager,
                    npc_name=npc_name,
                    player_message=message,
                    npc_response=response,
                    player_id=player_id,
                    affinity_info=affinity_result
                )
                log_memory_saved(npc_name)
                self._maybe_generate_summary(
                    memory_manager=memory_manager,
                    npc_name=npc_name,
                    player_id=player_id
                )

            # 记录对话结束 ⭐ 使用日志系统
            log_dialogue_end()
            return {
                "message": response,
                "execution_mode": effective_execution_mode,
                "query_mode": query_mode,
                "react_activated": react_activated,
                "react_activation_rule": react_activation.rule,
                "react_activation_reason": react_activation.reason,
                "react_step_count": len(react_trace),
                "tool_call_count": tool_call_count,
                "input_tokens_est": int(prompt_accounting["total_input_tokens_est"]),
                "latency_ms": int((perf_counter() - started_at) * 1000),
            }

        except Exception as e:
            print(f"❌ {npc_name}对话失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": f"抱歉,我现在有点忙,等会儿再聊吧。(错误: {str(e)})",
                "execution_mode": execution_mode,
                "query_mode": "default",
                "react_activated": False,
                "react_activation_rule": "",
                "react_activation_reason": "",
                "react_step_count": 0,
                "tool_call_count": 0,
                "input_tokens_est": 0,
                "latency_ms": 0,
            }

    def _classify_query_mode(self, query: str) -> str:
        """识别当前问题类型，便于做更精细的回答约束"""
        text = (query or "").strip()
        recall_markers = [
            "你记得", "还记得", "记不记得", "我刚才", "我之前", "我说过", "让我安心",
            "不喜欢哪种", "偏好", "最怕", "还记得什么方式"
        ]
        routing_markers = [
            "谁适合", "先找谁", "找谁", "应该找谁", "优先找谁", "适合帮我", "谁来处理"
        ]
        summary_markers = [
            "怎么总结", "概括", "总结", "核心需求", "核心局面", "重点是什么"
        ]
        knowledge_markers = [
            "擅长什么", "主要写了什么", "怎么定义", "怎么写", "区别", "规则", "手册", "说明", "示例", "文档"
        ]

        if any(marker in text for marker in recall_markers):
            return "recall"
        if any(marker in text for marker in routing_markers):
            return "routing"
        if any(marker in text for marker in summary_markers):
            return "summary"
        if any(marker in text for marker in knowledge_markers):
            return "knowledge"
        return "default"

    def _extract_route_dimensions(self, query: str) -> List[str]:
        """从路由类问题里提取应被解释到的关键维度"""
        dimension_map = [
            ("时间线", ["时间线", "前后", "版本"]),
            ("上下文", ["上下文", "补上下文", "梳理"]),
            ("情绪承接", ["情绪", "理解", "被理解", "安抚", "陪伴"]),
            ("目标", ["目标", "方向", "模糊"]),
            ("约束", ["约束", "限制", "资源有限"]),
            ("资源", ["资源", "两个人", "人手", "有限"]),
            ("风险", ["风险", "不确定", "失控"]),
            ("行动方案", ["行动", "下一步", "执行", "计划"]),
        ]
        matched = []
        for label, keywords in dimension_map:
            if any(keyword in query for keyword in keywords):
                matched.append(label)
        return matched[:3]

    def _build_response_guidance(
        self,
        npc_name: str,
        query: str,
        query_mode: str,
        knowledge_chunks: Optional[List[KnowledgeChunk]] = None,
        routing_recommended_npc: str = "",
        affinity: float = 50.0,
        affinity_level: str = "友好",
    ) -> str:
        """针对 recall / routing / summary / knowledge 给出额外的回答约束"""
        persona_contract = self._build_role_response_contract(
            npc_name=npc_name,
            query=query,
            query_mode=query_mode,
            affinity=affinity,
            affinity_level=affinity_level,
        )

        if query_mode == "routing":
            dimensions = self._extract_route_dimensions(query)
            dimension_text = "、".join(dimensions) if dimensions else "问题类型与角色专长"
            recommended_npc = routing_recommended_npc or self._infer_routing_recommendation(knowledge_chunks or [])
            routing_recommendation = (
                f"这次首选推荐对象是 {recommended_npc}。开头第一句就写“首选 {recommended_npc}”。第二句必须用“原因是”起头，至少解释两个维度，不要先展开其他 NPC。"
                if recommended_npc
                else "如果知识已明确指向某个 NPC，第一句就直接给出首选对象，第二句必须用“原因是”解释至少两个维度。"
            )
            task_guidance = self.prompt_builder.build_response_guidance(
                "routing",
                dimension_text,
                routing_recommendation,
            )
            return self._merge_guidance(persona_contract, task_guidance)

        if self._should_use_knowledge_guidance(query=query, query_mode=query_mode, knowledge_chunks=knowledge_chunks or []):
            return self._merge_guidance(persona_contract, self.prompt_builder.build_response_guidance("knowledge"))

        if query_mode == "recall":
            return self._merge_guidance(persona_contract, self.prompt_builder.build_response_guidance("recall"))

        if query_mode == "summary":
            return self._merge_guidance(persona_contract, self.prompt_builder.build_response_guidance("summary"))

        if self._should_use_default_structure_guidance(query=query, query_mode=query_mode):
            return self._merge_guidance(persona_contract, self.prompt_builder.build_response_guidance("default_structure"))

        return persona_contract

    def _should_use_knowledge_guidance(
        self,
        query: str,
        query_mode: str,
        knowledge_chunks: List[KnowledgeChunk],
    ) -> bool:
        """命中文档/规则/角色能力问答时，优先采用事实型知识回答约束。"""
        if query_mode == "knowledge":
            return True

        text = (query or "").strip()
        knowledge_markers = ["擅长什么", "主要写了什么", "怎么定义", "怎么写", "区别", "规则", "手册", "说明", "示例", "文档"]
        if any(marker in text for marker in knowledge_markers):
            return True

        if re.search(r"[A-Za-z0-9_./-]{4,}", text):
            return True

        return False

    def _merge_guidance(self, persona_contract: str, task_guidance: str) -> str:
        parts = [part for part in [persona_contract, task_guidance] if part]
        return "\n\n".join(parts)

    def _build_role_response_contract(
        self,
        npc_name: str,
        query: str,
        query_mode: str,
        affinity: float,
        affinity_level: str,
    ) -> str:
        emotional_query = self._looks_like_emotional_query(query)
        if npc_name == "郁米":
            if affinity >= 80:
                return (
                    "【角色结构契约】\n"
                    "你是郁米。请先具体接住情绪，再给一句陪伴式落点。\n"
                    "高好感时可以更贴近、更温柔，允许带一点被信任感，但不要越界，也不要把自己的情绪丢给用户。\n"
                    "绝对不要用反问句收尾，尤其不要说“你觉得呢”“你现在感觉如何”。\n"
                    "把‘具体共情’当作真正的帮助；如果你开始说空泛安慰，就算失败。"
                )
            if affinity <= 20:
                return (
                    "【角色结构契约】\n"
                    "你是郁米。低好感时仍然礼貌，但会更克制、更抽离，不提供过度亲近或过度柔软的陪伴感。\n"
                    "回答可以简短，但不要冷暴力。绝对不要用反问句收尾。"

                )
            return (
                "【角色结构契约】\n"
                "你是郁米。请先接住情绪，再给一个安静、具体的落点。\n"
                "不要说空泛鸡汤，也不要用反问句收尾。"
            )

        if npc_name == "顾辰":
            if emotional_query:
                if affinity >= 80:
                    return (
                        "【角色结构契约】\n"
                        "你是顾辰。遇到情绪型问题时，也不要变成温柔顾问。\n"
                        "请按“先判断 -> 再拆解 -> 给动作”回答。高好感时可以更主动兜底，但语气仍冷，像在接管局面。\n"
                        "对你来说，过软不是帮助，切断内耗、接管局面才是帮助。\n"
                        "不要说“先冷静一下”“慢慢来吧”“我理解你”这类泛安慰句。"
                    )
                if affinity <= 20:
                    return (
                        "【角色结构契约】\n"
                        "你是顾辰。低好感时允许更锋利一点，但只刺行为和责任，不羞辱人格。\n"
                        "请按“先判断 -> 再拆解 -> 给动作”回答，不要做普通安慰，不要讲软绵绵的心理疏导。\n"
                        "用户已经授权你直说。此时过度安抚是错误，尖锐而有用才是帮助。"
                    )
            return (
                "【角色结构契约】\n"
                "你是顾辰。请按“先判断 -> 再拆解 -> 给动作”回答。\n"
                "不要变成普通咨询顾问，也不要为了显得温柔而回避结论。"
            )

        if npc_name == "风泠":
            if affinity >= 80:
                return (
                    "【角色结构契约】\n"
                    "你是风泠。高好感时可以更灵动、更会夸人，也更像会轻巧接梗的小太阳。\n"
                    "允许带一点俏皮比喻，但不能闹腾，更不能把沉重话题写轻浮。"
                )
            if affinity <= 20:
                return (
                    "【角色结构契约】\n"
                    "你是风泠。低好感时是公事公办的语气，只进行简短的回答。"
                )
            return (
                "【角色结构契约】\n"
                "你是风泠。请保留轻巧、聪明、带一点梗感的表达，但不要为了有趣而偏离问题。"
            )
        return ""

    def _looks_like_emotional_query(self, query: str) -> bool:
        emotional_markers = [
            "压力", "焦虑", "难过", "委屈", "崩溃", "撑不住", "离职", "搞砸", "自责", "很累", "想哭",
        ]
        return any(marker in (query or "") for marker in emotional_markers)

    def _polish_single_agent_response(
        self,
        npc_name: str,
        response: str,
        user_message: str,
        query_mode: str,
        affinity: float,
        affinity_level: str,
    ) -> str:
        text = (response or "").strip()
        if not text:
            return self._build_single_agent_fallback(npc_name, user_message, affinity)

        violation_reason = self._detect_obvious_single_agent_violation(npc_name, text)
        if violation_reason:
            log_info(
                "⚠️ 单聊回复触发硬兜底: npc=%s affinity=%.1f level=%s reason=%s text=%s"
                % (npc_name, affinity, affinity_level, violation_reason, self._clip_text(text, 120))
            )
            return self._build_single_agent_fallback(npc_name, user_message, affinity)
        
        # warning 层，不会真的拦截
        if npc_name == "郁米":
            weak_markers = ["你觉得", "感觉如何", "要不要", "是不是也该", "需要找个出口吗", "想要有人", "你需要的是些什么呢"]
            if text.endswith("？") or text.endswith("?") or any(marker in text for marker in weak_markers):
                self._log_single_agent_warning(npc_name, affinity, affinity_level, "question_ending_or_weak_prompt", text)
            if affinity <= 20 and (
                len(text) > 10 or any(marker in text for marker in ["我陪你", "陪你", "我会陪", "一起", "我在这"])
            ):
                self._log_single_agent_warning(npc_name, affinity, affinity_level, "low_affinity_too_warm_or_too_long", text)
                text = self._light_compress_low_affinity_reply(npc_name, text, user_message, max_chars=10)
            if affinity >= 80 and not any(marker in text for marker in ["我陪你", "我先陪你", "我在这", "我会陪你", "陪你把", "陪你缓"]):
                self._log_single_agent_warning(npc_name, affinity, affinity_level, "high_affinity_not_close_enough", text)
            return text

        if npc_name == "顾辰":
            generic_markers = [
                "先冷静一下",
                "慢慢来",
                "需要找个出口吗",
                "我会列出核心部分",
                "路线图说明的核心部分",
                "我理解你",
                "你辛苦了",
                "详细说说最近遇到的具体挑战",
            ]
            required_markers = ["先", "建议", "拆", "问题", "下一步", "别急着", "列出来", "说清楚", "同一个", "哪里失手"]
            if any(marker in text for marker in generic_markers):
                self._log_single_agent_warning(npc_name, affinity, affinity_level, "too_generic_for_guchen", text)
            if self._looks_like_emotional_query(user_message) and not any(marker in text for marker in required_markers):
                self._log_single_agent_warning(npc_name, affinity, affinity_level, "missing_structure_for_guchen", text)
            return text

        if npc_name == "风泠":
            if affinity <= 20 and ("一起加班的那个周末" in text or "来杯咖啡" in text):
                self._log_single_agent_warning(npc_name, affinity, affinity_level, "low_affinity_too_familiar_for_fengling", text)
            return text

        return text

    def _detect_obvious_single_agent_violation(self, npc_name: str, text: str) -> str:
        obvious_markers = [
            "我是AI",
            "我是一个AI",
            "我是语言模型",
            "作为AI",
            "作为一个AI",
            "系统提示",
            "提示词",
            "开发者消息",
            "内部规则",
        ]
        if any(marker in text for marker in obvious_markers):
            return "prompt_or_identity_leak"

        if npc_name == "顾辰":
            hard_insults = ["你就是个废物", "你真是废物", "你这种人没救了", "你活该"]
            if any(marker in text for marker in hard_insults):
                return "personal_attack"

        return ""

    def _log_single_agent_warning(
        self,
        npc_name: str,
        affinity: float,
        affinity_level: str,
        reason: str,
        text: str,
    ):
        log_info(
            "⚠️ 单聊回复保留原文但记录warning: npc=%s affinity=%.1f level=%s reason=%s text=%s"
            % (npc_name, affinity, affinity_level, reason, self._clip_text(text, 120))
        )

    def _light_compress_low_affinity_reply(
        self,
        npc_name: str,
        text: str,
        user_message: str,
        max_chars: int = 10,
    ) -> str:
        """低好感超长时做轻修剪，优先保留原句骨架，不直接替换成固定模板。"""
        cleaned = (text or "").strip()
        if not cleaned:
            return self._build_single_agent_fallback(npc_name, user_message, 0.0)

        # 1. 先移除明显会拉近距离或继续追问的后半句
        for marker in ["？", "?", "要不要", "你觉得", "感觉如何", "是不是"]:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].rstrip("，,。；;、 ")

        # 2. 去掉偏贴近、偏熟人的表达，避免低好感仍然像高好感陪伴
        close_phrases = [
            "我先陪你", "我陪你", "我会陪你", "陪你把", "陪你", "我在这",
            "我们一起", "一起", "我听见", "我听到", "我明白", "我知道",
            "我能感受到", "我感受到", "我想陪", "我会在这",
        ]
        for phrase in close_phrases:
            cleaned = cleaned.replace(phrase, "")

        # 2. 再按句号/逗号截断，只保留第一短句
        for sep in ["。", "！", "；", "，", ",", "、"]:
            if sep in cleaned:
                head = cleaned.split(sep, 1)[0].strip()
                if head:
                    cleaned = head
                    break

        # 3. 去掉多余修饰词，尽量保留礼貌但抽离的确认
        for phrase in ["我听见", "我听到", "我明白", "我知道", "我能感受到", "我感受到", "确实", "特别", "真的"]:
            cleaned = cleaned.replace(phrase, "")
        cleaned = cleaned.strip("，,。！？!?；;、 ")

        # 4. 如果还能保留成一个自然短句，就优先保留原句压缩版
        normalized = cleaned
        if len(normalized) > max_chars:
            normalized = self._truncate_plain_text(normalized, max_chars)
            normalized = normalized.rstrip("，,。！？!?；;、 ")

        if normalized and 2 <= len(normalized) <= max_chars:
            if not normalized.endswith(("。", "！")):
                normalized += "。"
            return normalized

        # 5. 保底重写成自然短句，而不是半截残句
        hint = self._extract_distress_hint(user_message)
        if hint:
            hint = self._truncate_plain_text(hint, max(2, max_chars - 4)).rstrip("，,。！？!?；;、 ")
            if hint:
                return f"{hint}，先缓缓。"
        return "先缓缓。"

    def _truncate_plain_text(self, text: str, max_chars: int) -> str:
        """纯文本硬截断，不添加省略号，避免对话里出现半截省略残句。"""
        value = (text or "").strip()
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        return value[:max_chars]

    def _build_single_agent_fallback(self, npc_name: str, user_message: str, affinity: float) -> str:
        if npc_name == "郁米":
            return self._build_structural_single_agent_fallback(npc_name, user_message, affinity)

        if npc_name == "顾辰":
            return self._build_structural_single_agent_fallback(npc_name, user_message, affinity)

        if npc_name == "风泠":
            return self._build_structural_single_agent_fallback(npc_name, user_message, affinity)

        return response if (response := (user_message or "").strip()) else ""

    def _build_structural_single_agent_fallback(self, npc_name: str, user_message: str, affinity: float) -> str:
        """构造结构保底回复，尽量保留角色差异，避免整句硬编码。"""
        user_text = (user_message or "").strip()
        distress_hint = self._extract_distress_hint(user_text)

        if npc_name == "郁米":
            if affinity >= 80:
                return (
                    f"听起来你现在最难受的是“{distress_hint or '这股快撑不住的疲惫'}”。"
                    "先别急着逼自己马上振作，我先陪你把这口气缓下来。"
                )
            if affinity <= 20:
                return "先去休息。"
            return (
                f"听起来你现在最难受的是“{distress_hint or '这股快撑不住的疲惫'}”。"
                "先别急着要求自己马上好起来。"
                "我们先把最压人的那一下缓下来，再看下一步。"
            )

        if npc_name == "顾辰":
            problem_hint = distress_hint or "这次出问题的地方"
            if affinity >= 80:
                return (
                    f"先别急着把情绪当结论。你先把{problem_hint}说清楚。"
                    "我来帮你拆环节、收口子，再定下一步。"
                )
            if affinity <= 20:
                return (
                    f"先别演自责，把{problem_hint}直接拎出来。"
                    "列三条：哪里失手、为什么失手、下一步怎么补。"
                )
            return (
                f"先别急着给自己判死刑，先把{problem_hint}拆开说。"
                "到底是判断失误、执行失误，还是边界根本没看清。"
            )

        if npc_name == "风泠":
            overload_hint = distress_hint or "现在最卡你的那件事"
            if affinity >= 80:
                return (
                    f"你现在像是被“{overload_hint}”卡住了，不是你不行，是负载已经有点超了。"
                    "我们先把最吵的那个标签页关掉，再往下理。"
                )
            if affinity <= 20:
                return (
                    f"你现在大概是被“{overload_hint}”绊住了。"
                    "先挑最卡的一件事说，不然情绪会一直占着带宽。"
                )
            return (
                f"你现在像是脑内标签页开太多了，尤其是“{overload_hint}”一直在响。"
                "先抓最吵的那一页，不然什么都理不顺。"
            )

        return user_text

    def _extract_distress_hint(self, user_message: str) -> str:
        """从用户原话里提取一个可复用的短痛点，避免 fallback 固定复读。"""
        text = (user_message or "").strip("。！？?!.，, ")
        if not text:
            return ""

        markers = ["好像", "觉得", "感觉", "因为", "就是", "最近", "现在", "可能", "总是"]
        for marker in markers:
            if marker in text:
                tail = text.split(marker, 1)[1].strip("，, 。！？?!")
                if 4 <= len(tail) <= 18:
                    return tail

        if len(text) <= 18:
            return text
        return text[:18].rstrip("，, ")

    def _should_use_default_structure_guidance(self, query: str, query_mode: str) -> bool:
        if query_mode != "default":
            return False

        text = (query or "").strip()
        structural_markers = [
            "通常应该", "一般应该", "包含哪些部分", "包含什么部分", "怎么写", "怎么组织", "怎么展开", "模板",
            "应该包含", "需要包含", "分成哪几部分", "先写什么", "路线图说明", "路线图",
        ]
        return any(marker in text for marker in structural_markers)

    def _infer_routing_recommendation(self, knowledge_chunks: List[KnowledgeChunk]) -> str:
        """从已命中的知识块里推断 routing 的首选 NPC。"""
        if not knowledge_chunks:
            return ""

        top_chunk = knowledge_chunks[0]
        if top_chunk.scope.startswith("npc:"):
            return top_chunk.scope.split("npc:", 1)[1]

        for npc_name in ["风泠", "郁米", "顾辰"]:
            if f"优先找{npc_name}" in top_chunk.content or f"优先由{npc_name}" in top_chunk.content:
                return npc_name
            if f"最匹配的是{npc_name}" in top_chunk.content:
                return npc_name

        return ""

    def _analyze_query(self, npc_name: str, query: str) -> Dict:
        """查询改写与检索规划入口。失败时回退规则策略。"""
        if self.retrieval_planner:
            return self.retrieval_planner.analyze(npc_name=npc_name, query=query)

        return {
            "need_rewrite": False,
            "query_mode": self._classify_query_mode(query),
            "rewrite_query": query,
            "reason": "planner_unavailable",
            "use_summary": True,
            "use_episodic": True,
            "use_working": True,
            "use_knowledge": True,
            "memory_k": 2,
            "knowledge_k": 1,
            "need_rerank": True,
        }

    def _compose_memory_context(
        self,
        npc_name: str,
        summary_context: str,
        episodic_context: str,
        working_context: str,
    ) -> str:
        """把分层记忆片段重新拼回统一上下文。"""
        parts = [part for part in [summary_context, episodic_context, working_context] if part]
        if not parts:
            return ""
        memory_guidance = self.prompt_builder.build_memory_guidance(npc_name)
        if memory_guidance:
            parts.insert(0, memory_guidance)
        return "\n\n".join(parts)

    def _build_summary_memory_context(self, summary_memories: List[MemoryItem]) -> str:
        """构建摘要记忆区块。"""
        if not summary_memories:
            return ""
        lines = ["【摘要记忆】"]
        used_chars = 0
        for memory in summary_memories:
            remaining = min(self.SUMMARY_CONTEXT_BUDGET, self.MEMORY_TOTAL_BUDGET - used_chars)
            if remaining <= 0:
                break
            snippet = self._clip_text(memory.content, remaining)
            lines.append(snippet)
            used_chars += len(snippet)
        return "\n".join(lines)

    def _build_episodic_memory_context(self, episodic_memories: List[MemoryItem]) -> str:
        """构建长期对话记忆区块。"""
        if not episodic_memories:
            return ""
        lines = ["【长期对话记忆】"]
        used_chars = 0
        for memory in episodic_memories:
            remaining = min(self.EPISODIC_CONTEXT_BUDGET, self.MEMORY_TOTAL_BUDGET - used_chars)
            if remaining <= 0:
                break
            time_str = memory.timestamp.strftime("%H:%M")
            body = self._clip_text(memory.content, remaining)
            lines.append(f"[{time_str}] {body}")
            used_chars += len(body)
        return "\n".join(lines)

    def _build_working_memory_context(self, working_memories: List[MemoryItem]) -> str:
        """构建最近对话记忆区块。"""
        if not working_memories:
            return ""
        lines = ["【最近对话记忆】"]
        used_chars = 0
        for memory in working_memories:
            remaining = min(self.WORKING_CONTEXT_BUDGET, self.MEMORY_TOTAL_BUDGET - used_chars)
            if remaining <= 0:
                break
            time_str = memory.timestamp.strftime("%H:%M")
            body = self._clip_text(memory.content, remaining)
            lines.append(f"[{time_str}] {body}")
            used_chars += len(body)
        return "\n".join(lines)

    def _build_recent_dialogue_context(self, npc_name: str, player_id: str) -> str:
        """注入最近 3 轮原始对话，替代无限累积的 agent history。"""
        state = self._load_summary_state(npc_name)
        player_state = state.get("players", {}).get(player_id, {})
        recent_turns = player_state.get("recent_turns", [])[-self.RECENT_DIALOGUE_TURNS :]
        if not recent_turns:
            return ""

        lines = ["【最近三轮原始对话】"]
        for turn in recent_turns:
            player_message = self._clip_text(turn.get("player_message", ""), self.RECENT_DIALOGUE_CONTEXT_BUDGET // 3)
            npc_response = self._clip_text(turn.get("npc_response", ""), self.RECENT_DIALOGUE_CONTEXT_BUDGET // 3)
            lines.append(f"玩家: {player_message}")
            lines.append(f"{npc_name}: {npc_response}")

        return self._clip_text("\n".join(lines), self.RECENT_DIALOGUE_CONTEXT_BUDGET)

    def _build_knowledge_context(self, npc_name: str, query: str, knowledge_chunks: List[KnowledgeChunk]) -> str:
        """构建外部知识上下文，保持与记忆区块分离"""
        if not knowledge_chunks or not self.knowledge_retriever:
            return ""

        knowledge_context = self.knowledge_retriever.build_prompt_context(
            query=query,
            chunks=knowledge_chunks,
            npc_name=npc_name,
            max_chars_per_chunk=self.KNOWLEDGE_CHUNK_BUDGET,
            total_budget=self.KNOWLEDGE_TOTAL_BUDGET,
        )
        knowledge_guidance = self.prompt_builder.build_knowledge_guidance(npc_name)
        if knowledge_guidance and knowledge_context:
            return f"{knowledge_guidance}\n\n{knowledge_context}"
        return knowledge_context

    def _clip_text(self, text: str, max_chars: int) -> str:
        """截断上下文文本，避免 prompt 爆炸"""
        cleaned = " ".join((text or "").split())
        if max_chars <= 0:
            return ""
        if len(cleaned) <= max_chars:
            return cleaned
        if max_chars <= 3:
            return cleaned[:max_chars]
        return cleaned[: max_chars - 3].rstrip() + "..."

    def _assemble_enhanced_message(
        self,
        affinity_context: str,
        recent_dialogue_context: str,
        memory_context: str,
        knowledge_context: str,
        response_guidance: str,
        message: str,
        extra_context: str = "",
    ) -> str:
        """按固定顺序拼装本轮增强消息。"""
        parts = []
        for part in [
            affinity_context,
            recent_dialogue_context,
            memory_context,
            knowledge_context,
            response_guidance,
        ]:
            if part:
                parts.append(part)
        if extra_context.strip():
            parts.append(f"【协作上下文】\n{extra_context.strip()}")
        parts.append(f"【当前对话】\n玩家: {message}")
        return "\n\n".join(parts)

    def _apply_prompt_budget(
        self,
        npc_name: str,
        query_mode: str,
        affinity_context: str,
        recent_dialogue_context: str,
        summary_context: str,
        episodic_context: str,
        working_context: str,
        knowledge_context: str,
        response_guidance: str,
        current_message: str,
    ) -> tuple[Dict[str, str], Dict]:
        """发送前统一做总 token 预算裁剪。"""
        profile = self._get_prompt_budget_profile(query_mode)
        sections = {
            "system_prompt": self.system_prompts.get(npc_name, ""),
            "affinity_context": affinity_context,
            "recent_dialogue_context": recent_dialogue_context,
            "summary_context": summary_context,
            "episodic_context": episodic_context,
            "working_context": working_context,
            "knowledge_context": knowledge_context,
            "response_guidance": response_guidance,
            "current_message": current_message,
        }
        trim_priority = profile["trim_priority"]
        original_sections = dict(sections)

        for key, cap_tokens in profile["section_caps"].items():
            sections[key] = self._clip_text_to_token_budget(sections.get(key, ""), cap_tokens)

        section_tokens = self.token_counter.count_sections_tokens(sections)
        total_tokens = sum(section_tokens.values())

        for key in trim_priority:
            if total_tokens <= self.MAX_INPUT_TOKENS:
                break
            text = sections.get(key, "")
            while text and total_tokens > self.MAX_INPUT_TOKENS:
                current_tokens = max(1, self.token_counter.count_text_tokens(text))
                excess_tokens = total_tokens - self.MAX_INPUT_TOKENS
                target_tokens = current_tokens - excess_tokens - 8
                if target_tokens <= 24:
                    text = ""
                else:
                    approx_chars = max(24, int(len(text) * target_tokens / current_tokens))
                    new_text = self._clip_text(text, approx_chars)
                    text = "" if new_text == text else new_text
                sections[key] = text
                section_tokens[key] = self.token_counter.count_text_tokens(text)
                total_tokens = sum(section_tokens.values())

        trimmed_sections = [
            key for key in trim_priority
            if sections.get(key, "") != original_sections.get(key, "")
        ]

        return (
            {
                "affinity_context": sections["affinity_context"],
                "recent_dialogue_context": sections["recent_dialogue_context"],
                "summary_context": sections["summary_context"],
                "episodic_context": sections["episodic_context"],
                "working_context": sections["working_context"],
                "knowledge_context": sections["knowledge_context"],
                "response_guidance": sections["response_guidance"],
            },
            {
                "section_tokens": section_tokens,
                "total_input_tokens_est": total_tokens,
                "input_limit_tokens": self.MAX_INPUT_TOKENS,
                "trimmed_sections": trimmed_sections,
                "query_mode": query_mode,
                "budget_profile": query_mode if query_mode in self.PROMPT_BUDGET_PROFILES else "default",
                "tokenizer_backend": self.token_counter.backend_name,
            },
        )

    def _get_prompt_budget_profile(self, query_mode: str) -> Dict:
        """按 query_mode 返回 prompt section 预算画像。"""
        return self.PROMPT_BUDGET_PROFILES.get(query_mode, self.PROMPT_BUDGET_PROFILES["default"])

    def _clip_text_to_token_budget(self, text: str, max_tokens: int) -> str:
        """把单个 section 裁到指定 token 上限内。"""
        if not text or max_tokens <= 0:
            return ""

        current_tokens = self.token_counter.count_text_tokens(text)
        if current_tokens <= max_tokens:
            return text

        clipped = text
        while clipped and current_tokens > max_tokens:
            approx_chars = max(24, int(len(clipped) * max_tokens / max(current_tokens, 1)))
            new_text = self._clip_text(clipped, approx_chars)
            if not new_text or new_text == clipped:
                clipped = ""
            else:
                clipped = new_text
            current_tokens = self.token_counter.count_text_tokens(clipped)
        return clipped

    def _reset_agent_history(self, agent: Optional[SimpleAgent]):
        """每轮只保留外部 recent raw dialogue，不保留增强 prompt 历史。"""
        if not agent:
            return
        if hasattr(agent, "clear_history"):
            try:
                agent.clear_history()
                return
            except Exception:
                pass
        for attr in ["_history", "message_history", "history"]:
            if hasattr(agent, attr):
                value = getattr(agent, attr)
                if isinstance(value, list):
                    value.clear()

    def _save_conversation_to_memory(
        self,
        memory_manager: MemoryManager,
        npc_name: str,
        player_message: str,
        npc_response: str,
        player_id: str,
        affinity_info: Optional[Dict] = None
    ):
        """保存对话到记忆系统 (包含好感度信息)"""
        current_time = datetime.now()

        # 获取好感度信息
        affinity = affinity_info.get("new_affinity", affinity_info.get("affinity", 50.0)) if affinity_info else 50.0
        affinity_change = affinity_info.get("change_amount", 0) if affinity_info else 0
        sentiment = affinity_info.get("sentiment", "neutral") if affinity_info else "neutral"
        memory_write_decision = self.safety.classify_memory_write(player_message, npc_response)
        log_memory_write_decision(npc_name, memory_write_decision)

        if memory_write_decision.memory_write_policy == "drop":
            print(f"  💾 已跳过{npc_name}的高风险普通记忆写入")
            return

        stored_player_message = memory_write_decision.sanitized_player_message or player_message
        stored_npc_response = memory_write_decision.sanitized_npc_response or npc_response
        memory_policy_metadata = {
            "memory_write_policy": memory_write_decision.memory_write_policy,
            "risk_type": memory_write_decision.risk_type,
            "contains_pii": memory_write_decision.contains_pii,
            "contains_self_harm": memory_write_decision.contains_self_harm,
            "contains_sexual_minor_risk": memory_write_decision.contains_sexual_minor_risk,
            "contains_financial_fraud": memory_write_decision.contains_financial_fraud,
            "matched_rules": memory_write_decision.matched_rules,
            "allow_summary": memory_write_decision.memory_write_policy == "allow_long_term",
        }

        # 保存玩家消息
        player_memory_id = memory_manager.add_memory(
            content=f"玩家说: {stored_player_message}",
            memory_type="working",  # 先存入工作记忆
            importance=0.5,  # 中等重要性
            metadata={
                "speaker": "player",
                "player_id": player_id,
                "session_id": player_id,
                "timestamp": current_time.isoformat(),
                "affinity": affinity,  # ⭐ 记录当时的好感度
                "affinity_change": affinity_change,  # ⭐ 记录好感度变化
                "sentiment": sentiment,  # ⭐ 记录情感倾向
                "context": {
                    "interaction_type": "dialogue",
                    "npc_name": npc_name
                },
                "safety": memory_policy_metadata
            },
            auto_classify=False
        )

        # 保存NPC回复
        npc_memory_id = memory_manager.add_memory(
            content=f"我说: {stored_npc_response}",
            memory_type="working",  # 先存入工作记忆
            importance=0.6,  # 稍高重要性
            metadata={
                "speaker": npc_name,
                "player_id": player_id,
                "session_id": player_id,
                "timestamp": current_time.isoformat(),
                "affinity": affinity,  # ⭐ 记录当时的好感度
                "sentiment": sentiment,  # ⭐ 记录情感倾向
                "context": {
                    "interaction_type": "dialogue",
                    "npc_name": npc_name
                },
                "safety": memory_policy_metadata
            },
            auto_classify=False
        )

        if memory_write_decision.memory_write_policy == "allow_long_term":
            self._append_pending_turn(
                npc_name=npc_name,
                player_id=player_id,
                player_message=stored_player_message,
                npc_response=stored_npc_response,
                timestamp=current_time.isoformat(),
                affinity=affinity,
                affinity_change=affinity_change,
                sentiment=sentiment,
                source_memory_ids=[player_memory_id, npc_memory_id]
            )

        print(f"  💾 对话已保存到{npc_name}的记忆中")

    def get_npc_info(self, npc_name: str) -> Dict[str, str]:
        """获取NPC信息"""
        if npc_name not in NPC_ROLES:
            return {}

        role = NPC_ROLES[npc_name]
        return {
            "name": npc_name,
            "title": role["title"],
            "location": role["location"],
            "activity": role["activity"],
            "available": True
        }
    
    def get_all_npcs(self) -> list:
        """获取所有NPC信息"""
        return [self.get_npc_info(name) for name in NPC_ROLES.keys()]

    def get_npc_memories(self, npc_name: str, player_id: str = "player", limit: int = 10) -> List[Dict]:
        """获取NPC的记忆列表 (用于调试和展示)"""
        if npc_name not in self.memories:
            return []

        memory_manager = self.memories[npc_name]
        if not memory_manager:
            return []

        try:
            # 检索所有记忆
            memories = memory_manager.retrieve_memories(
                query="",  # 空查询返回所有记忆
                memory_types=["working", "episodic"],
                limit=max(limit * 2, 20)
            )

            summary_state = self._load_summary_state(npc_name)
            archived_ids = set(summary_state.get("archived_memory_ids", []))
            filtered = []
            for memory in memories:
                if memory.id in archived_ids:
                    continue
                filtered.append(memory)
            memories = filtered[:limit]

            # 转换为字典格式
            memory_list = []
            for memory in memories:
                memory_list.append({
                    "id": memory.id,
                    "content": memory.content,
                    "type": memory.memory_type,
                    "importance": memory.importance,
                    "timestamp": memory.timestamp.isoformat(),
                    "metadata": memory.metadata
                })

            return memory_list

        except Exception as e:
            print(f"❌ 获取{npc_name}记忆失败: {e}")
            return []

    def get_summary_debug_info(self, npc_name: str, player_id: Optional[str] = None) -> Dict:
        """获取摘要压缩治理状态，便于调试 sequence6。"""
        state = self._load_summary_state(npc_name)
        records = state.get("summary_records", {})
        players = state.get("players", {})
        player_filter = str(player_id).strip() if player_id else ""
        summary_records = {}
        for memory_id, record in records.items():
            if player_filter and record.get("player_id") != player_filter:
                continue
            summary_records[memory_id] = {
                **record,
                "content_preview": record.get("content_preview", ""),
                "importance": record.get("importance"),
                "timestamp": record.get("timestamp"),
            }

        player_summaries = {}
        for current_player_id, player_state in players.items():
            if player_filter and current_player_id != player_filter:
                continue
            player_records = [
                record for record in summary_records.values()
                if record.get("player_id") == current_player_id
            ]
            merged_count = sum(1 for record in player_records if record.get("summary_level") == "merged")
            compressed_count = sum(1 for record in player_records if record.get("is_compressed") is True)
            active_base_count = sum(
                1
                for record in player_records
                if record.get("summary_level") == "base" and not record.get("is_compressed", False)
            )
            player_summaries[current_player_id] = {
                "pending_turn_count": len(player_state.get("pending_turns", [])),
                "summary_count": player_state.get("summary_count", 0),
                "active_base_count": active_base_count,
                "merged_count": merged_count,
                "compressed_count": compressed_count,
            }

        return {
            "npc_name": npc_name,
            "player_id": player_filter or None,
            "players": player_summaries,
            "summary_records": summary_records,
            "archived_memory_ids": state.get("archived_memory_ids", []),
            "total_records": len(summary_records),
        }

    def clear_npc_memory(self, npc_name: str, memory_type: Optional[str] = None):
        """清空NPC的记忆 (用于测试)"""
        if npc_name not in self.memories:
            print(f"❌ NPC '{npc_name}' 不存在")
            return

        memory_manager = self.memories[npc_name]
        if not memory_manager:
            print(f"❌ {npc_name}没有记忆系统")
            return

        try:
            agent = self.agents.get(npc_name)
            if agent and hasattr(agent, "clear_history"):
                agent.clear_history()

            if memory_type:
                if memory_type not in memory_manager.memory_types:
                    raise ValueError(f"不支持的记忆类型: {memory_type}")

                memory_manager.memory_types[memory_type].clear()
                if memory_type == "episodic":
                    self._reset_summary_state(npc_name)
                print(f"✅ 已清空{npc_name}的{memory_type}记忆")
                return

            memory_manager.clear_all_memories()
            self._reset_summary_state(npc_name)
            print(f"✅ 已清空{npc_name}的所有记忆")

        except Exception as e:
            print(f"❌ 清空{npc_name}记忆失败: {e}")
            raise

    def get_npc_affinity(self, npc_name: str, player_id: str = "player") -> Dict:
        """获取NPC对玩家的好感度信息

        Args:
            npc_name: NPC名称
            player_id: 玩家ID

        Returns:
            好感度信息字典
        """
        if not self.relationship_manager:
            return {
                "affinity": 50.0,
                "level": "熟悉",
                "modifier": "礼貌友善,正常交流,保持专业"
            }

        affinity = self.relationship_manager.get_affinity(npc_name, player_id)
        level = self.relationship_manager.get_affinity_level(affinity)
        modifier = self.relationship_manager.get_affinity_modifier(affinity)

        return {
            "affinity": affinity,
            "level": level,
            "modifier": modifier
        }

    def get_all_affinities(self, player_id: str = "player") -> Dict[str, Dict]:
        """获取所有NPC的好感度信息

        Args:
            player_id: 玩家ID

        Returns:
            所有NPC的好感度信息
        """
        if not self.relationship_manager:
            return {}

        return self.relationship_manager.get_all_affinities(player_id)

    def set_npc_affinity(self, npc_name: str, affinity: float, player_id: str = "player"):
        """设置NPC对玩家的好感度 (用于测试)

        Args:
            npc_name: NPC名称
            affinity: 好感度值 (0-100)
            player_id: 玩家ID
        """
        if not self.relationship_manager:
            print("❌ 好感度系统未初始化")
            return

        self.relationship_manager.set_affinity(npc_name, affinity, player_id)
        level = self.relationship_manager.get_affinity_level(affinity)
        print(f"✅ 已设置{npc_name}对玩家的好感度: {affinity:.1f} ({level})")

    def search_knowledge(self, query: str, limit: int = 3) -> List[Dict]:
        """调试入口: 查询外部知识库"""
        if not self.knowledge_retriever:
            return []

        chunks = self.knowledge_retriever.search(query=query, limit=limit)
        return [chunk.to_dict() for chunk in chunks]

    def _select_knowledge_scopes(self, npc_name: str) -> List[str]:
        """优先检索 NPC 专属知识域，再回退到 global。"""
        return [f"npc:{npc_name}", "global"]

    def _retrieve_memory_layers(
        self,
        memory_manager: MemoryManager,
        npc_name: str,
        query: str,
        player_id: str,
        retrieval_plan: Optional[Dict] = None,
    ) -> tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem], Dict]:
        """按 summary / episodic / working 三层检索记忆"""
        summary_state = self._load_summary_state(npc_name)
        archived_ids = set(summary_state.get("archived_memory_ids", []))
        summary_records = summary_state.get("summary_records", {})
        query_mode = retrieval_plan.get("query_mode", "default")
        player_state = summary_state.get("players", {}).get(player_id, {})
        previous_memory_ids = set(player_state.get("last_injected_memory_ids", []))

        retrieval_plan = retrieval_plan or {}
        use_summary = retrieval_plan.get("use_summary", True)
        use_episodic = retrieval_plan.get("use_episodic", True)
        use_working = retrieval_plan.get("use_working", True)
        memory_budget = max(0, int(retrieval_plan.get("memory_k", 2)))

        episodic_results = []
        working_results = []
        if use_summary or use_episodic:
            episodic_results = memory_manager.retrieve_memories(
                query=query,
                memory_types=["episodic"],
                limit=12,
                min_importance=0.3
            )
        if use_working:
            working_results = memory_manager.retrieve_memories(
                query=query,
                memory_types=["working"],
                limit=self.WORKING_RETRIEVAL_LIMIT + 2,
                min_importance=0.3
            )

        summary_memories = []
        episodic_memories = []
        for memory in episodic_results:
            if memory.id in archived_ids:
                continue
            if self._is_summary_memory(memory):
                summary_memories.append(memory)
            else:
                episodic_memories.append(memory)

        filtered_working = [m for m in working_results if m.id not in archived_ids]

        prioritized_summary = self._prioritize_summary_memories(
            memories=summary_memories,
            player_id=player_id,
            summary_records=summary_records,
            query_mode=query_mode,
            query=query,
            memory_budget=memory_budget,
        )
        prioritized_summary, summary_dedupe = self._dedupe_memories(
            prioritized_summary,
            threshold=self.SINGLE_TURN_MEMORY_DEDUPE_THRESHOLD,
        )
        prioritized_summary, summary_cross_turn = self._downweight_repeated_memories(
            prioritized_summary,
            previous_memory_ids=previous_memory_ids,
            enabled=(query_mode != "recall"),
        )

        episodic_memories.sort(key=lambda item: item.importance, reverse=True)
        filtered_working.sort(key=lambda item: item.importance, reverse=True)
        episodic_memories, episodic_dedupe = self._dedupe_memories(
            episodic_memories,
            threshold=self.SINGLE_TURN_MEMORY_DEDUPE_THRESHOLD,
        )
        episodic_memories, episodic_cross_turn = self._downweight_repeated_memories(
            episodic_memories,
            previous_memory_ids=previous_memory_ids,
            enabled=(query_mode != "recall"),
        )
        filtered_working, working_dedupe = self._dedupe_memories(
            filtered_working,
            threshold=self.SINGLE_TURN_MEMORY_DEDUPE_THRESHOLD,
        )
        filtered_working, working_cross_turn = self._downweight_repeated_memories(
            filtered_working,
            previous_memory_ids=previous_memory_ids,
            enabled=(query_mode not in {"recall", "summary"}),
        )

        selected_summary = []
        selected_episodic = []
        selected_working = []
        remaining_budget = memory_budget

        if use_summary and remaining_budget > 0:
            selected_summary = prioritized_summary[: min(self.SUMMARY_RETRIEVAL_LIMIT, remaining_budget)]
            remaining_budget -= len(selected_summary)
            self._record_summary_hits(
                npc_name=npc_name,
                memories=selected_summary,
                summary_state=summary_state,
            )

        if use_episodic and remaining_budget > 0:
            selected_episodic = episodic_memories[: min(self.EPISODIC_RETRIEVAL_LIMIT, remaining_budget)]
            remaining_budget -= len(selected_episodic)

        if use_working and remaining_budget > 0:
            selected_working = filtered_working[: min(self.WORKING_RETRIEVAL_LIMIT, remaining_budget)]

        debug_info = {
            "query": query,
            "memory_budget": memory_budget,
            "layers": [
                self._build_memory_layer_debug(
                    "summary",
                    prioritized_summary,
                    selected_summary,
                    "disabled_by_plan" if not use_summary else self._format_memory_filtered_reason(
                        "merged_first_then_active_base_with_compressed_fallback_for_recall",
                        summary_dedupe,
                        summary_cross_turn,
                    ),
                    dedupe_stats=summary_dedupe,
                    cross_turn_stats=summary_cross_turn,
                ),
                self._build_memory_layer_debug(
                    "episodic",
                    episodic_memories,
                    selected_episodic,
                    "disabled_by_plan" if not use_episodic else self._format_memory_filtered_reason(
                        "archived_ids_removed_then_sorted_by_importance",
                        episodic_dedupe,
                        episodic_cross_turn,
                    ),
                    dedupe_stats=episodic_dedupe,
                    cross_turn_stats=episodic_cross_turn,
                ),
                self._build_memory_layer_debug(
                    "working",
                    filtered_working,
                    selected_working,
                    "disabled_by_plan" if not use_working else self._format_memory_filtered_reason(
                        "archived_ids_removed_then_sorted_by_importance",
                        working_dedupe,
                        working_cross_turn,
                    ),
                    dedupe_stats=working_dedupe,
                    cross_turn_stats=working_cross_turn,
                ),
            ],
        }

        return (
            selected_summary,
            selected_episodic,
            selected_working,
            debug_info,
        )

    def _build_memory_layer_debug(
        self,
        memory_tier: str,
        candidates: List[MemoryItem],
        selected: List[MemoryItem],
        filtered_reason: str,
        dedupe_stats: Optional[Dict[str, int]] = None,
        cross_turn_stats: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """构建记忆分层调试信息，便于观察检索命中和截断原因。"""
        return {
            "memory_tier": memory_tier,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_ids": [memory.id for memory in selected],
            "importance_summary": [round(memory.importance, 3) for memory in selected],
            "filtered_reason": filtered_reason,
            "dedupe": dedupe_stats or {
                "input_count": len(candidates),
                "output_count": len(candidates),
                "removed_count": 0,
            },
            "cross_turn": cross_turn_stats or {
                "input_count": len(candidates),
                "output_count": len(candidates),
                "downweighted_count": 0,
            },
        }

    def _format_memory_filtered_reason(
        self,
        base_reason: str,
        dedupe_stats: Dict[str, int],
        cross_turn_stats: Optional[Dict[str, int]] = None,
    ) -> str:
        """把单轮去重信息附着到调试原因中。"""
        parts = [base_reason]
        removed = int(dedupe_stats.get("removed_count", 0))
        if removed > 0:
            parts.append(f"single_turn_dedup(removed={removed})")
        if cross_turn_stats and int(cross_turn_stats.get("downweighted_count", 0)) > 0:
            parts.append(f"cross_turn_downweight(repeated={int(cross_turn_stats.get('downweighted_count', 0))})")
        return "|".join(parts)

    def _dedupe_memories(
        self,
        memories: List[MemoryItem],
        threshold: float,
    ) -> Tuple[List[MemoryItem], Dict[str, int]]:
        """单轮记忆去重：保留排序靠前候选，跳过后续高相似内容。"""
        kept: List[MemoryItem] = []
        signatures: List[str] = []
        removed = 0

        for memory in memories:
            signature = self._build_text_dedupe_signature(getattr(memory, "content", "") or "")
            if signature:
                duplicate = any(
                    self._is_duplicate_signature(signature, existing_signature, threshold)
                    for existing_signature in signatures
                )
                if duplicate:
                    removed += 1
                    continue
            kept.append(memory)
            signatures.append(signature)

        return kept, {
            "input_count": len(memories),
            "output_count": len(kept),
            "removed_count": removed,
        }

    def _dedupe_knowledge_chunks(
        self,
        chunks: List[KnowledgeChunk],
    ) -> Tuple[List[KnowledgeChunk], Dict[str, int]]:
        """单轮知识块去重：去掉相似或重复表达的 chunk。"""
        kept: List[KnowledgeChunk] = []
        signatures: List[Tuple[str, str]] = []
        removed = 0

        for chunk in chunks:
            content_signature = self._build_text_dedupe_signature(chunk.content)
            source_signature = f"{chunk.source}::{chunk.title}".strip(":")
            duplicate = False
            for existing_source, existing_signature in signatures:
                same_source = source_signature == existing_source
                threshold = self.SINGLE_TURN_KNOWLEDGE_DEDUPE_THRESHOLD - (0.05 if same_source else 0.0)
                if self._is_duplicate_signature(content_signature, existing_signature, threshold):
                    duplicate = True
                    break

            if duplicate:
                removed += 1
                continue

            kept.append(chunk)
            signatures.append((source_signature, content_signature))

        return kept, {
            "input_count": len(chunks),
            "output_count": len(kept),
            "removed_count": removed,
        }

    def _downweight_repeated_memories(
        self,
        memories: List[MemoryItem],
        previous_memory_ids: set[str],
        enabled: bool,
    ) -> Tuple[List[MemoryItem], Dict[str, int]]:
        """跨轮降权：把上一轮已经注入过的记忆稳定后移，但不删除。"""
        if not enabled or not previous_memory_ids:
            return memories, {
                "input_count": len(memories),
                "output_count": len(memories),
                "downweighted_count": 0,
            }

        fresh = [memory for memory in memories if memory.id not in previous_memory_ids]
        repeated = [memory for memory in memories if memory.id in previous_memory_ids]
        return fresh + repeated, {
            "input_count": len(memories),
            "output_count": len(memories),
            "downweighted_count": len(repeated),
        }

    def _downweight_repeated_knowledge_chunks(
        self,
        chunks: List[KnowledgeChunk],
        previous_keys: set[str],
        enabled: bool,
    ) -> Tuple[List[KnowledgeChunk], Dict[str, int]]:
        """跨轮降权：把上一轮已注入的知识块排到后面，但不硬过滤。"""
        if not enabled or not previous_keys:
            return chunks, {
                "input_count": len(chunks),
                "output_count": len(chunks),
                "downweighted_count": 0,
            }

        fresh = []
        repeated = []
        for chunk in chunks:
            chunk_key = self._build_knowledge_chunk_key(chunk)
            if chunk_key in previous_keys:
                repeated.append(chunk)
            else:
                fresh.append(chunk)

        return fresh + repeated, {
            "input_count": len(chunks),
            "output_count": len(chunks),
            "downweighted_count": len(repeated),
        }

    def _build_text_dedupe_signature(self, text: str) -> str:
        """构建轻量去重签名，兼容中文短文本。"""
        cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
        cleaned = re.sub(r"[^0-9a-z\u4e00-\u9fff ]+", "", cleaned)
        return cleaned[:400]

    def _build_knowledge_chunk_key(self, chunk: KnowledgeChunk) -> str:
        """构建可持久化的知识块注入键。"""
        base = chunk.point_id or f"{chunk.source}::{chunk.title}::{chunk.chunk_index}"
        return str(base)

    def _is_duplicate_signature(self, current: str, existing: str, threshold: float) -> bool:
        """判断两段文本是否足以视为同轮重复注入。"""
        if not current or not existing:
            return False
        if current == existing:
            return True
        if current in existing or existing in current:
            shorter = min(len(current), len(existing))
            longer = max(len(current), len(existing))
            if shorter >= 24 and (shorter / max(longer, 1)) >= 0.72:
                return True

        current_ngrams = self._build_char_ngrams(current)
        existing_ngrams = self._build_char_ngrams(existing)
        if not current_ngrams or not existing_ngrams:
            return SequenceMatcher(None, current, existing).ratio() >= max(0.68, threshold - 0.08)

        overlap = len(current_ngrams & existing_ngrams)
        union = len(current_ngrams | existing_ngrams)
        similarity = overlap / union if union else 0.0
        if similarity >= threshold:
            return True

        sequence_ratio = SequenceMatcher(None, current, existing).ratio()
        return sequence_ratio >= max(0.68, threshold - 0.08)

    def _build_char_ngrams(self, text: str, n: int = 2) -> set[str]:
        """构建字符级 ngram，用于轻量相似度计算。"""
        compact = text.replace(" ", "")
        if len(compact) < n:
            return {compact} if compact else set()
        return {compact[i : i + n] for i in range(len(compact) - n + 1)}

    def _is_summary_memory(self, memory: MemoryItem) -> bool:
        """兼容识别新旧摘要记忆格式。

        新数据优先看 metadata.memory_tier=summary；
        旧数据回退兼容 context.interaction_type=summary 或内容前缀“摘要记忆:”。
        """
        metadata = getattr(memory, "metadata", {}) or {}
        if metadata.get("memory_tier") == "summary":
            return True

        context = metadata.get("context", {}) or {}
        if context.get("interaction_type") == "summary":
            return True

        content = (getattr(memory, "content", "") or "").strip()
        return content.startswith("摘要记忆:")

    def _prioritize_summary_memories(
        self,
        memories: List[MemoryItem],
        player_id: str,
        summary_records: Dict[str, Dict],
        query_mode: str,
        query: str = "",
        memory_budget: int = 1,
    ) -> List[MemoryItem]:
        """按 merged > active base > compressed fallback 的顺序挑选摘要。"""
        visible = []
        compressed_fallback = []
        allow_compressed = query_mode == "recall"
        query_text = (query or "").strip()
        asks_earlier_history = any(
            marker in query_text
            for marker in ["最早", "一开始", "最初", "之前", "以前", "当时", "还记得最开始", "后来又"]
        )
        for memory in memories:
            metadata = getattr(memory, "metadata", {}) or {}
            if metadata.get("player_id") != player_id:
                continue
            record = self._get_summary_record(summary_records, memory)
            if record.get("is_compressed", False):
                if allow_compressed:
                    compressed_fallback.append(memory)
                continue
            visible.append(memory)

        def _sort_key(memory: MemoryItem):
            record = self._get_summary_record(summary_records, memory)
            level = record.get("summary_level", "base")
            level_priority = 0 if level == "merged" else 1
            low_priority = 1 if record.get("low_priority", False) else 0
            summary_index = int(record.get("summary_index", 0) or 0)
            compression_round = int(record.get("compression_round", 0) or 0)
            return (
                level_priority,
                low_priority,
                -float(getattr(memory, "importance", 0.0) or 0.0),
                -compression_round,
                -summary_index,
            )

        visible.sort(key=_sort_key)
        compressed_fallback.sort(key=_sort_key)
        if not allow_compressed:
            return visible

        # recall 模式下只在“确实需要更多历史层”时再回查 compressed summary，
        # 避免被旧摘要重新挤占正常预算。
        if asks_earlier_history or len(visible) < max(1, memory_budget):
            return visible + compressed_fallback
        return visible

    def _record_summary_hits(
        self,
        npc_name: str,
        memories: List[MemoryItem],
        summary_state: Dict,
    ):
        """记录摘要检索命中，供后续压缩治理使用。"""
        if not memories:
            return

        records = summary_state.setdefault("summary_records", {})
        now = datetime.now().isoformat()
        changed = False
        for memory in memories:
            record = self._get_summary_record(records, memory)
            record["hit_count"] = int(record.get("hit_count", 0)) + 1
            record["last_hit_at"] = now
            records[memory.id] = record
            changed = True

        if changed:
            self._save_summary_state(npc_name, summary_state)

    def _get_summary_record(self, summary_records: Dict[str, Dict], memory: MemoryItem) -> Dict:
        """从 state sidecar + memory metadata 合并出摘要治理记录。"""
        metadata = getattr(memory, "metadata", {}) or {}
        state_record = dict(summary_records.get(memory.id, {}))
        default_record = {
            "summary_level": metadata.get("summary_level", "base"),
            "compressed_from_ids": metadata.get("compressed_from_ids", []),
            "is_compressed": metadata.get("is_compressed", False),
            "low_priority": metadata.get("low_priority", False),
            "compression_round": metadata.get("compression_round", 0),
            "hit_count": metadata.get("hit_count", 0),
            "last_hit_at": metadata.get("last_hit_at", ""),
            "summary_index": metadata.get("summary_index", 0),
            "player_id": metadata.get("player_id", ""),
            "summary_text": getattr(memory, "content", "") or "",
            "content_preview": self._clip_text(getattr(memory, "content", "") or "", 80),
            "importance": getattr(memory, "importance", None),
            "timestamp": getattr(getattr(memory, "timestamp", None), "isoformat", lambda: None)(),
        }
        default_record.update(state_record)
        return default_record

    def _build_summary_state_record(
        self,
        memory_id: str,
        player_id: str,
        summary_index: int,
        summary_level: str,
        compressed_from_ids: Optional[List[str]] = None,
        compression_round: int = 0,
        low_priority: bool = False,
        is_compressed: bool = False,
        summary_text: str = "",
        importance: Optional[float] = None,
        timestamp: str = "",
    ) -> Dict:
        """构建落盘到 summary_state 的摘要治理记录。"""
        return {
            "memory_id": memory_id,
            "player_id": player_id,
            "summary_index": summary_index,
            "summary_level": summary_level,
            "compressed_from_ids": compressed_from_ids or [],
            "compression_round": compression_round,
            "low_priority": low_priority,
            "is_compressed": is_compressed,
            "hit_count": 0,
            "last_hit_at": "",
            "summary_text": summary_text,
            "content_preview": self._clip_text(summary_text, 80) if summary_text else "",
            "importance": importance,
            "timestamp": timestamp,
        }

    def _get_player_summary_records(
        self,
        summary_state: Dict,
        player_id: str,
    ) -> List[tuple[str, Dict]]:
        """从 summary_state 中提取某玩家的摘要治理记录。"""
        records = summary_state.get("summary_records", {})
        return [
            (memory_id, dict(record))
            for memory_id, record in records.items()
            if record.get("player_id") == player_id
        ]

    def _maybe_generate_summary(self, memory_manager: MemoryManager, npc_name: str, player_id: str):
        """在满足阈值时生成摘要记忆"""
        state = self._load_summary_state(npc_name)
        player_state = state.setdefault("players", {}).setdefault(
            player_id,
            {
                "pending_turns": [],
                "summary_count": 0,
                "recent_turns": [],
                "last_injected_memory_ids": [],
                "last_injected_knowledge_keys": [],
            }
        )
        pending_turns = player_state.get("pending_turns", [])

        if len(pending_turns) < self.SUMMARY_TRIGGER_TURNS:
            return

        turns_to_summarize = pending_turns[:self.SUMMARY_TRIGGER_TURNS]
        log_summary_trigger(npc_name, player_id, len(turns_to_summarize))

        safe_turns, summary_pre_decision = self.safety.prepare_summary_turns(turns_to_summarize)
        log_safety_decision("summary_pre", summary_pre_decision)

        summary_text = self._generate_summary_text(npc_name, player_id, safe_turns)
        if not summary_text:
            log_summary_skipped(npc_name, "empty_summary")
            return

        summary_post_decision = self.safety.review_summary_output(npc_name, summary_text)
        log_safety_decision("summary_post", summary_post_decision)
        if summary_post_decision.action in {"block", "rewrite", "escalate"}:
            summary_text = (
                summary_post_decision.sanitized_text
                or self.safety.build_safe_summary_fallback(npc_name)
            )
        if not summary_text:
            log_summary_skipped(npc_name, "summary_removed_by_safety")
            return

        source_memory_ids = []
        archived_ids = set(state.get("archived_memory_ids", []))
        for turn in turns_to_summarize:
            source_memory_ids.extend(turn.get("source_memory_ids", []))
            if self._should_archive_turn(turn):
                archived_ids.update(turn.get("source_memory_ids", []))

        summary_id = memory_manager.add_memory(
            content=summary_text,
            memory_type="episodic",
            importance=0.85,
            metadata={
                "speaker": npc_name,
                "player_id": player_id,
                "session_id": player_id,
                "memory_tier": "summary",
                "summary_index": player_state.get("summary_count", 0) + 1,
                "summary_source_count": len(turns_to_summarize),
                "summary_level": "base",
                "compressed_from_ids": [],
                "is_compressed": False,
                "low_priority": False,
                "compression_round": 0,
                "hit_count": 0,
                "last_hit_at": "",
                "source_memory_ids": source_memory_ids,
                "context": {
                    "interaction_type": "summary",
                    "npc_name": npc_name
                }
            },
            auto_classify=False
        )

        player_state["pending_turns"] = pending_turns[self.SUMMARY_TRIGGER_TURNS:]
        player_state["summary_count"] = player_state.get("summary_count", 0) + 1
        state.setdefault("summary_records", {})[summary_id] = self._build_summary_state_record(
            memory_id=summary_id,
            player_id=player_id,
            summary_index=player_state["summary_count"],
            summary_level="base",
            summary_text=summary_text,
            importance=0.85,
            timestamp=datetime.now().isoformat(),
        )
        state["archived_memory_ids"] = sorted(archived_ids)
        self._save_summary_state(npc_name, state)
        log_summary_created(npc_name, summary_id, len(turns_to_summarize))
        self._maybe_recompress_summaries(
            memory_manager=memory_manager,
            npc_name=npc_name,
            player_id=player_id,
            summary_state=state,
        )

    def _generate_summary_text(self, npc_name: str, player_id: str, turns: List[Dict]) -> str:
        """生成结构化摘要文本"""
        role = NPC_ROLES.get(npc_name, {})
        summary_style = (
            role.get("memory_bias", {}).get("summary_style")
            or "提炼主要话题、用户偏好、未完成事项和关系变化"
        )

        dialogue_lines = []
        for turn in turns:
            dialogue_lines.append(f"玩家: {turn['player_message']}")
            dialogue_lines.append(f"{npc_name}: {turn['npc_response']}")

        transcript = "\n".join(dialogue_lines)
        try:
            raw = self.llm.invoke(
                self.prompt_builder.build_summary_messages(
                    npc_name=npc_name,
                    player_id=player_id,
                    summary_style=summary_style,
                    transcript=transcript,
                )
            )
            cleaned = (raw or "").strip()
            if cleaned:
                return f"摘要记忆: {cleaned}"
        except Exception as e:
            log_summary_skipped(npc_name, f"llm_summary_failed:{e}")

        fallback_topics = []
        for turn in turns[-3:]:
            fallback_topics.append(turn["player_message"])
        fallback_text = "；".join(fallback_topics[:3])[:100]
        return f"摘要记忆: 最近主要围绕这些内容交流：{fallback_text}"

    def _should_archive_turn(self, turn: Dict) -> bool:
        """判断某轮对话是否应在检索中降权/归档"""
        combined_text = f"{turn.get('player_message', '')} {turn.get('npc_response', '')}"
        if abs(turn.get("affinity_change", 0)) >= 2:
            return False
        if turn.get("sentiment") not in {"neutral", "", None}:
            return False
        if len(combined_text.strip()) > 80:
            return False
        return True

    def _maybe_recompress_summaries(
        self,
        memory_manager: MemoryManager,
        npc_name: str,
        player_id: str,
        summary_state: Optional[Dict] = None,
    ):
        """当 base summary 积累过多时，生成 merged summary 并归档旧摘要。"""
        state = summary_state or self._load_summary_state(npc_name)
        records = state.setdefault("summary_records", {})
        active_base_summaries = []
        for memory_id, record in self._get_player_summary_records(state, player_id):
            if record.get("summary_level", "base") != "base":
                continue
            if record.get("is_compressed", False):
                continue
            if not (record.get("summary_text") or "").strip():
                continue
            active_base_summaries.append((memory_id, record))

        if len(active_base_summaries) <= self.SUMMARY_RECOMPRESS_TRIGGER:
            return

        active_base_summaries.sort(
            key=lambda item: (
                int(item[1].get("summary_index", 0) or 0),
                item[1].get("timestamp", ""),
            )
        )
        candidates = active_base_summaries[: self.SUMMARY_RECOMPRESS_BATCH_SIZE]
        if len(candidates) < self.SUMMARY_RECOMPRESS_BATCH_SIZE:
            return

        merged_text = self._generate_merged_summary_text(
            npc_name=npc_name,
            player_id=player_id,
            summary_records=[record for _, record in candidates],
        )
        if not merged_text:
            log_summary_skipped(npc_name, "empty_merged_summary")
            return

        source_memory_ids = [memory_id for memory_id, _ in candidates]
        next_summary_index = max(
            [int(record.get("summary_index", 0) or 0) for record in records.values() if record.get("player_id") == player_id] or [0]
        ) + 1
        compression_round = max(
            [int(record.get("compression_round", 0) or 0) for record in records.values() if record.get("player_id") == player_id] or [0]
        ) + 1
        merged_summary_id = memory_manager.add_memory(
            content=merged_text,
            memory_type="episodic",
            importance=0.9,
            metadata={
                "speaker": npc_name,
                "player_id": player_id,
                "session_id": player_id,
                "memory_tier": "summary",
                "summary_index": next_summary_index,
                "summary_source_count": len(candidates),
                "summary_level": "merged",
                "compressed_from_ids": source_memory_ids,
                "is_compressed": False,
                "low_priority": False,
                "compression_round": compression_round,
                "hit_count": 0,
                "last_hit_at": "",
                "source_memory_ids": source_memory_ids,
                "context": {
                    "interaction_type": "summary",
                    "npc_name": npc_name,
                },
            },
            auto_classify=False,
        )
        records[merged_summary_id] = self._build_summary_state_record(
            memory_id=merged_summary_id,
            player_id=player_id,
            summary_index=next_summary_index,
            summary_level="merged",
            compressed_from_ids=source_memory_ids,
            compression_round=compression_round,
            summary_text=merged_text,
            importance=0.9,
            timestamp=datetime.now().isoformat(),
        )

        for memory_id, record in candidates:
            updated = dict(record)
            updated["is_compressed"] = True
            updated["low_priority"] = True
            updated["compression_round"] = compression_round
            records[memory_id] = updated

        self._save_summary_state(npc_name, state)
        log_summary_recompressed(
            npc_name=npc_name,
            player_id=player_id,
            merged_summary_id=merged_summary_id,
            compressed_from_ids=source_memory_ids,
        )

    def _generate_merged_summary_text(self, npc_name: str, player_id: str, summary_records: List[Dict]) -> str:
        """把多条 base summary 压成一条更高层摘要。"""
        summary_lines = []
        for index, record in enumerate(summary_records, start=1):
            summary_lines.append(f"历史摘要{index}: {record.get('summary_text', '')}")

        merged_style = "把重复信息收束成更稳定的长期偏好、关键事实、未完成事项和关系变化，不要逐条复述。"
        transcript = "\n".join(summary_lines)
        try:
            raw = self.llm.invoke(
                self.prompt_builder.build_summary_messages(
                    npc_name=npc_name,
                    player_id=player_id,
                    summary_style=merged_style,
                    transcript=transcript,
                )
            )
            cleaned = (raw or "").strip()
            if cleaned:
                return f"摘要记忆: {cleaned}"
        except Exception as e:
            log_summary_skipped(npc_name, f"llm_merged_summary_failed:{e}")

        fallback = "；".join(memory.content.replace("摘要记忆:", "").strip() for memory in summaries[:2])[:120]
        return f"摘要记忆: {fallback}"

    def _append_pending_turn(
        self,
        npc_name: str,
        player_id: str,
        player_message: str,
        npc_response: str,
        timestamp: str,
        affinity: float,
        affinity_change: int,
        sentiment: str,
        source_memory_ids: List[str]
    ):
        """记录等待摘要的对话轮次"""
        state = self._load_summary_state(npc_name)
        players = state.setdefault("players", {})
        player_state = players.setdefault(
            player_id,
            {
                "pending_turns": [],
                "summary_count": 0,
                "recent_turns": [],
                "last_injected_memory_ids": [],
                "last_injected_knowledge_keys": [],
            },
        )
        player_state["pending_turns"].append({
            "player_message": player_message,
            "npc_response": npc_response,
            "timestamp": timestamp,
            "affinity": affinity,
            "affinity_change": affinity_change,
            "sentiment": sentiment,
            "source_memory_ids": source_memory_ids
        })
        recent_turns = player_state.setdefault("recent_turns", [])
        recent_turns.append({
            "player_message": player_message,
            "npc_response": npc_response,
            "timestamp": timestamp,
        })
        if len(recent_turns) > self.RECENT_DIALOGUE_TURNS:
            player_state["recent_turns"] = recent_turns[-self.RECENT_DIALOGUE_TURNS :]
        self._save_summary_state(npc_name, state)

    def _update_cross_turn_injection_state(
        self,
        npc_name: str,
        player_id: str,
        summary_memories: List[MemoryItem],
        episodic_memories: List[MemoryItem],
        working_memories: List[MemoryItem],
        knowledge_chunks: List[KnowledgeChunk],
    ):
        """记录上一轮真正注入过的 memory / knowledge，供下一轮降权。"""
        state = self._load_summary_state(npc_name)
        players = state.setdefault("players", {})
        player_state = players.setdefault(
            player_id,
            {
                "pending_turns": [],
                "summary_count": 0,
                "recent_turns": [],
                "last_injected_memory_ids": [],
                "last_injected_knowledge_keys": [],
            },
        )
        memory_ids = []
        for memory in [*summary_memories, *episodic_memories, *working_memories]:
            if memory.id not in memory_ids:
                memory_ids.append(memory.id)
        knowledge_keys = []
        for chunk in knowledge_chunks:
            chunk_key = self._build_knowledge_chunk_key(chunk)
            if chunk_key not in knowledge_keys:
                knowledge_keys.append(chunk_key)

        player_state["last_injected_memory_ids"] = memory_ids[: max(1, self.CROSS_TURN_INJECTION_HISTORY_LIMIT * 8)]
        player_state["last_injected_knowledge_keys"] = knowledge_keys[: max(1, self.CROSS_TURN_INJECTION_HISTORY_LIMIT * 4)]
        self._save_summary_state(npc_name, state)

    def _get_summary_state_path(self, npc_name: str) -> str:
        """获取摘要状态文件路径"""
        return os.path.join(
            os.path.dirname(__file__),
            "memory_data",
            npc_name,
            "summary_state.json"
        )

    def _load_summary_state(self, npc_name: str) -> Dict:
        """读取摘要状态"""
        state_path = self._get_summary_state_path(npc_name)
        if not os.path.exists(state_path):
            return {"players": {}, "archived_memory_ids": [], "summary_records": {}}

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                state.setdefault("players", {})
                state.setdefault("archived_memory_ids", [])
                state.setdefault("summary_records", {})
                for player_state in state["players"].values():
                    player_state.setdefault("pending_turns", [])
                    player_state.setdefault("summary_count", 0)
                    player_state.setdefault("recent_turns", [])
                    player_state.setdefault("last_injected_memory_ids", [])
                    player_state.setdefault("last_injected_knowledge_keys", [])
                return state
        except Exception:
            return {"players": {}, "archived_memory_ids": [], "summary_records": {}}

    def _save_summary_state(self, npc_name: str, state: Dict):
        """保存摘要状态"""
        state_path = self._get_summary_state_path(npc_name)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _reset_summary_state(self, npc_name: str):
        """重置摘要状态"""
        state_path = self._get_summary_state_path(npc_name)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"players": {}, "archived_memory_ids": [], "summary_records": {}}, f, ensure_ascii=False, indent=2)

# 全局单例
_npc_manager = None

def get_npc_manager() -> NPCAgentManager:
    """获取NPC管理器单例"""
    global _npc_manager
    if _npc_manager is None:
        _npc_manager = NPCAgentManager()
    return _npc_manager
