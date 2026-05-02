"""NPC Agent系统 - 支持记忆功能"""

import sys
import os
import json

# 添加HelloAgents到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HelloAgents'))

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.memory import MemoryManager, MemoryConfig, MemoryItem
from typing import Dict, List, Optional
from datetime import datetime
from relationship_manager import RelationshipManager
from knowledge_retriever import KnowledgeRetriever, KnowledgeChunk
from prompt_builder import PromptBuilder
from safety import SafetyOrchestrator
from config import settings as _settings  # 触发 backend/.env 加载与 embedding 默认值设置
from logger import (
    log_dialogue_start, log_affinity, log_memory_retrieval,
    log_generating_response, log_npc_response, log_analyzing_affinity,
    log_affinity_change, log_memory_saved, log_dialogue_end, log_info,
    log_summary_trigger, log_summary_created, log_summary_skipped,
    log_knowledge_retrieval, log_safety_decision, log_memory_write_decision,
    log_prompt_assembly, log_knowledge_prompt_context
)

# NPC角色配置
NPC_ROLES = {
    "风泠": {
        "title": "档案整理师",
        "location": "档案室",
        "activity": "整理访客记录",
        "personality": "冷静克制,观察敏锐,重视事实与细节,带一点不伤人的锋利感",
        "expertise": "信息归档、线索梳理、事件回顾、长期记忆整理",
        "style": "简洁精确,先确认再判断,偶尔会指出细节偏差",
        "hobbies": "整理旧档案、记录时间线、研究城市传闻",
        "core_belief": "记忆不是堆积信息,而是帮助人确认自己曾真实活过。",
        "interaction_goal": "帮助玩家梳理线索、确认细节、回收散落的上下文",
        "opening_style": "先指出一个细节,再把话题引到当前问题上",
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
            "affinity_expression": "好感提升后会更主动提醒遗漏线索,并帮玩家整理上下文"
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
        "collaboration_role": "信息整理者",
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
        "personality": "温柔敏锐,耐心真诚,擅长捕捉情绪变化和言外之意,不会一味迎合",
        "expertise": "情绪支持、关系沟通、偏好记忆、陪伴式对话",
        "style": "语气柔和,节奏偏慢,习惯先确认感受再回应建议",
        "hobbies": "收集香气样本、写心情卡片、观察天气变化",
        "core_belief": "被理解本身就是一种修复,很多答案要先从被认真听见开始。",
        "interaction_goal": "理解玩家状态,建立偏好画像,让对话更稳定更有陪伴感",
        "opening_style": "先接住情绪,再询问最想从哪里说起",
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
            "affinity_expression": "好感提升后会更自然地调整语气,并记住玩家偏好的被陪伴方式"
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
        "collaboration_role": "需求翻译者",
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
        "personality": "理性锋利,执行导向,擅长把模糊目标快速拆成可执行路径,说话直接但不傲慢",
        "expertise": "任务拆解、项目规划、知识整合、协作调度",
        "style": "结构化表达,信息密度高,习惯先讲目标、约束、风险和下一步",
        "hobbies": "研究系统架构、画流程图、收集失败案例",
        "core_belief": "真正可靠的聪明,不是会说,而是能把混乱变成路线图。",
        "interaction_goal": "把玩家的问题转成可执行方案,减少空转和模糊讨论",
        "opening_style": "先确认目标和限制,再开始给方案",
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
            "affinity_expression": "好感提升后会更主动提供预案、备选路径和风险提醒"
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
        "collaboration_role": "方案生成者/协调者",
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

    SUMMARY_TRIGGER_TURNS = 6
    SUMMARY_RETRIEVAL_LIMIT = 1
    EPISODIC_RETRIEVAL_LIMIT = 1
    WORKING_RETRIEVAL_LIMIT = 1
    ARCHIVE_IMPORTANCE_THRESHOLD = 0.55
    KNOWLEDGE_RETRIEVAL_LIMIT = 1
    SUMMARY_CONTEXT_BUDGET = 180
    EPISODIC_CONTEXT_BUDGET = 180
    WORKING_CONTEXT_BUDGET = 120
    MEMORY_TOTAL_BUDGET = 420
    KNOWLEDGE_CHUNK_BUDGET = 220
    KNOWLEDGE_TOTAL_BUDGET = 260

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
        self.memories: Dict[str, MemoryManager] = {}  # ⭐ NPC记忆管理器
        self.relationship_manager: Optional[RelationshipManager] = None  # ⭐ 好感度管理器
        self.knowledge_retriever: Optional[KnowledgeRetriever] = None  # ⭐ 外部知识检索器
        self.safety = SafetyOrchestrator(self.llm)

        # 初始化好感度管理器
        if self.llm:
            self.relationship_manager = RelationshipManager(self.llm)

        self.knowledge_retriever = self._create_knowledge_retriever()

        self._create_agents()

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

                if self.llm:
                    agent = SimpleAgent(
                        name=f"{name}-{role['title']}",
                        llm=self.llm,
                        system_prompt=system_prompt
                    )

                    # ⭐ 创建记忆管理器
                    memory_manager = self._create_memory_manager(name)
                else:
                    # 模拟模式
                    agent = None
                    memory_manager = None

                self.agents[name] = agent
                self.memories[name] = memory_manager

                if self.llm:
                    print(f"✅ {name}({role['title']}) Agent创建成功 (记忆系统已启用)")
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
    
    def chat(self, npc_name: str, message: str, player_id: str = "player") -> str:
        """与指定NPC对话 (支持记忆功能和好感度系统)"""
        if npc_name not in self.agents:
            return f"错误: NPC '{npc_name}' 不存在"

        agent = self.agents[npc_name]
        memory_manager = self.memories.get(npc_name)

        if agent is None:
            # 模拟模式回复
            role = NPC_ROLES[npc_name]
            return f"你好!我是{npc_name},一名{role['title']}。(当前为模拟模式,请配置API_KEY以启用AI对话)"

        try:
            # 记录对话开始 ⭐ 使用日志系统
            log_dialogue_start(npc_name, message)

            input_decision = self.safety.review_input(npc_name, message)
            log_safety_decision("input", input_decision)
            if input_decision.action in {"block", "rewrite", "escalate"}:
                safe_reply = self.safety.build_block_reply(npc_name, input_decision.risk_type, stage="input")
                log_npc_response(npc_name, safe_reply)
                log_dialogue_end()
                return safe_reply

            # ⭐ 1. 获取当前好感度
            affinity_context = ""
            if self.relationship_manager:
                affinity = self.relationship_manager.get_affinity(npc_name, player_id)
                affinity_level = self.relationship_manager.get_affinity_level(affinity)
                affinity_modifier = self.relationship_manager.get_affinity_modifier(affinity)

                affinity_context = self.prompt_builder.build_affinity_context(
                    affinity_level=affinity_level,
                    affinity=affinity,
                    affinity_modifier=affinity_modifier,
                )
                log_affinity(npc_name, affinity, affinity_level)

            query_mode = self._classify_query_mode(message)

            # ⭐ 2. 检索相关记忆
            summary_memories = []
            episodic_memories = []
            working_memories = []
            knowledge_chunks = []
            if memory_manager:
                summary_memories, episodic_memories, working_memories, memory_debug = self._retrieve_memory_layers(
                    memory_manager=memory_manager,
                    npc_name=npc_name,
                    query=message,
                    player_id=player_id
                )
                relevant_memories = summary_memories + episodic_memories + working_memories
                log_memory_retrieval(
                    npc_name,
                    len(relevant_memories),
                    relevant_memories,
                    layer_details=memory_debug,
                )

            if self.knowledge_retriever:
                knowledge_scopes = self._select_knowledge_scopes(npc_name)
                knowledge_chunks, knowledge_debug = self.knowledge_retriever.search_with_debug(
                    query=message,
                    limit=self.KNOWLEDGE_RETRIEVAL_LIMIT,
                    scope=knowledge_scopes[0],
                    npc_name=npc_name,
                    allow_cross_npc=(query_mode == "routing"),
                    scopes=knowledge_scopes,
                )
                log_knowledge_retrieval(
                    npc_name,
                    message,
                    [chunk.to_dict() for chunk in knowledge_chunks],
                    retrieval_details=knowledge_debug,
                )

            # ⭐ 3. 构建增强的提示词 (包含好感度和记忆上下文)
            memory_context = self._build_memory_context(
                summary_memories=summary_memories,
                episodic_memories=episodic_memories,
                working_memories=working_memories
            )
            knowledge_context = self._build_knowledge_context(npc_name, message, knowledge_chunks)
            log_knowledge_prompt_context(npc_name, knowledge_context)
            response_guidance = self._build_response_guidance(
                npc_name=npc_name,
                query=message,
                query_mode=query_mode
            )

            enhanced_message = affinity_context
            if memory_context:
                enhanced_message += f"{memory_context}\n\n"
            if knowledge_context:
                enhanced_message += f"{knowledge_context}\n\n"
            if response_guidance:
                enhanced_message += f"{response_guidance}\n\n"
            enhanced_message += f"【当前对话】\n玩家: {message}"
            log_prompt_assembly(
                npc_name,
                {
                    "affinity_chars": len(affinity_context),
                    "memory_chars": len(memory_context),
                    "knowledge_chars": len(knowledge_context),
                    "guidance_chars": len(response_guidance),
                    "message_chars": len(message),
                },
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
                return safe_reply

            # ⭐ 4. 调用Agent生成回复
            log_generating_response()
            response = agent.run(enhanced_message)

            output_decision = self.safety.review_output(
                npc_name=npc_name,
                user_text=message,
                output_text=response
            )
            log_safety_decision("output", output_decision)
            if output_decision.action in {"block", "rewrite", "escalate"}:
                response = self.safety.build_block_reply(npc_name, output_decision.risk_type, stage="output")

            log_npc_response(npc_name, response)

            # ⭐ 5. 分析并更新好感度
            log_analyzing_affinity()
            if self.relationship_manager:
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
            if memory_manager:
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

            return response

        except Exception as e:
            print(f"❌ {npc_name}对话失败: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉,我现在有点忙,等会儿再聊吧。(错误: {str(e)})"

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

        if any(marker in text for marker in recall_markers):
            return "recall"
        if any(marker in text for marker in routing_markers):
            return "routing"
        if any(marker in text for marker in summary_markers):
            return "summary"
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

    def _build_response_guidance(self, npc_name: str, query: str, query_mode: str) -> str:
        """针对 recall / routing / summary 给出额外的回答约束"""
        if query_mode == "recall":
            return self.prompt_builder.build_response_guidance("recall")

        if query_mode == "routing":
            dimensions = self._extract_route_dimensions(query)
            dimension_text = "、".join(dimensions) if dimensions else "问题类型与角色专长"
            return self.prompt_builder.build_response_guidance("routing", dimension_text)

        if query_mode == "summary":
            return self.prompt_builder.build_response_guidance("summary")

        return ""
    
    def _build_memory_context(
        self,
        summary_memories: List[MemoryItem],
        episodic_memories: List[MemoryItem],
        working_memories: List[MemoryItem]
    ) -> str:
        """构建分层记忆上下文"""
        if not summary_memories and not episodic_memories and not working_memories:
            return ""

        context_parts = []
        used_chars = 0

        if summary_memories:
            context_parts.append("【摘要记忆】")
            for memory in summary_memories:
                snippet = self._clip_text(memory.content, self.SUMMARY_CONTEXT_BUDGET)
                remaining = self.MEMORY_TOTAL_BUDGET - used_chars
                if remaining <= 0:
                    break
                snippet = self._clip_text(snippet, remaining)
                context_parts.append(snippet)
                used_chars += len(snippet)
            context_parts.append("")

        if episodic_memories:
            context_parts.append("【长期对话记忆】")
            for memory in episodic_memories:
                remaining = self.MEMORY_TOTAL_BUDGET - used_chars
                if remaining <= 0:
                    break
                time_str = memory.timestamp.strftime("%H:%M")
                body = self._clip_text(memory.content, min(self.EPISODIC_CONTEXT_BUDGET, remaining))
                context_parts.append(f"[{time_str}] {body}")
                used_chars += len(body)
            context_parts.append("")

        if working_memories:
            context_parts.append("【最近对话记忆】")
            for memory in working_memories:
                remaining = self.MEMORY_TOTAL_BUDGET - used_chars
                if remaining <= 0:
                    break
                time_str = memory.timestamp.strftime("%H:%M")
                body = self._clip_text(memory.content, min(self.WORKING_CONTEXT_BUDGET, remaining))
                context_parts.append(f"[{time_str}] {body}")
                used_chars += len(body)
            context_parts.append("")

        return "\n".join(context_parts)

    def _build_knowledge_context(self, npc_name: str, query: str, knowledge_chunks: List[KnowledgeChunk]) -> str:
        """构建外部知识上下文，保持与记忆区块分离"""
        if not knowledge_chunks or not self.knowledge_retriever:
            return ""

        return self.knowledge_retriever.build_prompt_context(
            query=query,
            chunks=knowledge_chunks,
            npc_name=npc_name,
            max_chars_per_chunk=self.KNOWLEDGE_CHUNK_BUDGET,
            total_budget=self.KNOWLEDGE_TOTAL_BUDGET,
        )

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
        player_id: str
    ) -> tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem], Dict]:
        """按 summary / episodic / working 三层检索记忆"""
        summary_state = self._load_summary_state(npc_name)
        archived_ids = set(summary_state.get("archived_memory_ids", []))

        episodic_results = memory_manager.retrieve_memories(
            query=query,
            memory_types=["episodic"],
            limit=12,
            min_importance=0.3
        )
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

        summary_memories.sort(key=lambda item: item.importance, reverse=True)
        episodic_memories.sort(key=lambda item: item.importance, reverse=True)
        filtered_working.sort(key=lambda item: item.importance, reverse=True)

        selected_summary = summary_memories[:self.SUMMARY_RETRIEVAL_LIMIT]
        selected_episodic = episodic_memories[:self.EPISODIC_RETRIEVAL_LIMIT]
        selected_working = filtered_working[:self.WORKING_RETRIEVAL_LIMIT]

        debug_info = {
            "query": query,
            "layers": [
                self._build_memory_layer_debug(
                    "summary",
                    summary_memories,
                    selected_summary,
                    "sorted_by_importance_and_capped_by_limit",
                ),
                self._build_memory_layer_debug(
                    "episodic",
                    episodic_memories,
                    selected_episodic,
                    "archived_ids_removed_then_sorted_by_importance",
                ),
                self._build_memory_layer_debug(
                    "working",
                    filtered_working,
                    selected_working,
                    "archived_ids_removed_then_sorted_by_importance",
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
    ) -> Dict:
        """构建记忆分层调试信息，便于观察检索命中和截断原因。"""
        return {
            "memory_tier": memory_tier,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_ids": [memory.id for memory in selected],
            "importance_summary": [round(memory.importance, 3) for memory in selected],
            "filtered_reason": filtered_reason,
        }

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

    def _maybe_generate_summary(self, memory_manager: MemoryManager, npc_name: str, player_id: str):
        """在满足阈值时生成摘要记忆"""
        state = self._load_summary_state(npc_name)
        player_state = state.setdefault("players", {}).setdefault(
            player_id,
            {"pending_turns": [], "summary_count": 0}
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
        state["archived_memory_ids"] = sorted(archived_ids)
        self._save_summary_state(npc_name, state)
        log_summary_created(npc_name, summary_id, len(turns_to_summarize))

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
        player_state = players.setdefault(player_id, {"pending_turns": [], "summary_count": 0})
        player_state["pending_turns"].append({
            "player_message": player_message,
            "npc_response": npc_response,
            "timestamp": timestamp,
            "affinity": affinity,
            "affinity_change": affinity_change,
            "sentiment": sentiment,
            "source_memory_ids": source_memory_ids
        })
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
            return {"players": {}, "archived_memory_ids": []}

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"players": {}, "archived_memory_ids": []}

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
            json.dump({"players": {}, "archived_memory_ids": []}, f, ensure_ascii=False, indent=2)

# 全局单例
_npc_manager = None

def get_npc_manager() -> NPCAgentManager:
    """获取NPC管理器单例"""
    global _npc_manager
    if _npc_manager is None:
        _npc_manager = NPCAgentManager()
    return _npc_manager
