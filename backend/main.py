"""赛博小镇 FastAPI 后端主程序"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from config import settings
from api.models import (
    ChatRequest, ChatResponse, 
    NPCStatusResponse, NPCListResponse, NPCInfo,
    DelegatePreviewRequest, DelegatePreviewResponse,
    MultiChatRequest, MultiChatResponse, MultiChatAgentOutput,
)
from agents import get_npc_manager
from runtime.state_manager import get_state_manager

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("\n" + "="*60)
    print("🎮 赛博小镇后端服务启动中...")
    print("="*60)
    
    # 验证配置
    settings.validate()
    
    # 初始化NPC管理器
    npc_manager = get_npc_manager()
    
    # 初始化并启动状态管理器
    state_manager = get_state_manager(settings.NPC_UPDATE_INTERVAL)
    await state_manager.start()
    
    print("\n✅ 所有服务已启动!")
    print(f"📡 API地址: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📚 API文档: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("="*60 + "\n")
    
    yield
    
    # 关闭时
    print("\n🛑 正在关闭服务...")
    await state_manager.stop()
    print("✅ 服务已关闭\n")

# 创建FastAPI应用
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="赛博小镇 - 基于HelloAgents的AI NPC对话系统",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取全局实例
npc_manager = None
state_manager = None

def get_managers():
    """获取管理器实例"""
    global npc_manager, state_manager
    if npc_manager is None:
        npc_manager = get_npc_manager()
    if state_manager is None:
        state_manager = get_state_manager()
    return npc_manager, state_manager

# ==================== API路由 ====================

@app.get("/")
async def root():
    """根路径 - API信息"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "features": ["AI对话", "NPC记忆系统", "好感度系统", "批量状态更新", "外部知识检索"],
        "endpoints": {
            "docs": "/docs",
            "chat": "/chat",
            "multi_chat": "/multi_chat",
            "delegate_preview": "/orchestrate/delegate-preview",
            "npcs": "/npcs",
            "npcs_status": "/npcs/status",
            "npc_memories": "/npcs/{npc_name}/memories",
            "npc_summary_debug": "/npcs/{npc_name}/summary-debug",
            "npc_affinity": "/npcs/{npc_name}/affinity",
            "all_affinities": "/affinities",
            "knowledge_search": "/knowledge/search"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": "now"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_npc(request: ChatRequest):
    """与NPC对话接口
    
    玩家与指定NPC进行实时对话,使用独立的Agent处理
    """
    npc_mgr, _ = get_managers()
    
    # 验证NPC是否存在
    npc_info = npc_mgr.get_npc_info(request.npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{request.npc_name}' 不存在"
        )
    
    try:
        # 调用NPC Agent处理对话
        response = npc_mgr.chat_with_debug(
            request.npc_name,
            request.message,
            execution_mode=request.execution_mode,
        )
        
        return ChatResponse(
            npc_name=request.npc_name,
            npc_title=npc_info["title"],
            message=str(response.get("message", "")),
            success=True,
            execution_mode=str(response.get("execution_mode", request.execution_mode)),
            query_mode=str(response.get("query_mode", "default")),
            react_activated=bool(response.get("react_activated", False)),
            react_activation_rule=str(response.get("react_activation_rule", "")),
            react_activation_reason=str(response.get("react_activation_reason", "")),
            react_step_count=int(response.get("react_step_count", 0)),
            tool_call_count=int(response.get("tool_call_count", 0)),
            input_tokens_est=int(response.get("input_tokens_est", 0)),
            latency_ms=int(response.get("latency_ms", 0)),
            error_type=str(response.get("error_type", "")),
            prompt_budget_debug=dict(response.get("prompt_budget_debug", {}) or {}),
            retrieval_metrics=dict(response.get("retrieval_metrics", {}) or {}),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"对话处理失败: {str(e)}"
        )


@app.post("/orchestrate/delegate-preview", response_model=DelegatePreviewResponse)
async def delegate_preview(request: DelegatePreviewRequest):
    """LangGraph delegate 最小骨架调试入口

    只返回图状态与节点轨迹，不走正式 /chat 主链，也不保存记忆。
    """
    npc_mgr, _ = get_managers()
    npc_info = npc_mgr.get_npc_info(request.active_speaker)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{request.active_speaker}' 不存在"
        )

    try:
        state = npc_mgr.langgraph_delegate.invoke(
            user_message=request.message,
            active_speaker=request.active_speaker,
            player_id=request.player_id,
        )
        return DelegatePreviewResponse(
            active_speaker=request.active_speaker,
            message=request.message,
            orchestration_mode=str(state.get("orchestration_mode", "direct")),
            intent_type=str(state.get("intent_type", "")),
            query_mode=str(state.get("query_mode", "default")),
            needs_delegate=bool(state.get("needs_delegate", False)),
            delegate_to=str(state.get("delegate_to", "")),
            delegate_task=str(state.get("delegate_task", "")),
            final_style_owner=str(state.get("final_style_owner", "")),
            observation_cards=list(state.get("observation_cards", [])),
            node_trace=list(state.get("node_trace", [])),
            final_answer=str(state.get("final_answer", "")),
            langgraph_available=bool(npc_mgr.langgraph_delegate.available),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"delegate preview 失败: {str(e)}"
        )


@app.post("/multi_chat", response_model=MultiChatResponse)
async def multi_chat(request: MultiChatRequest):
    """多角色协作对话接口

    保留现有 /chat 主链不变，新增 LangGraph orchestration beta。
    """
    npc_mgr, _ = get_managers()

    try:
        state = npc_mgr.langgraph_multi.invoke(
            user_message=request.message,
            player_id=request.player_id,
            mode=request.mode,
            selected_agents=request.selected_agents,
            return_intermediate=request.return_intermediate,
        )
        execution_order = list(state.get("execution_order", []))
        order_map = {name: index for index, name in enumerate(execution_order)}
        intermediate_outputs = []
        if request.return_intermediate:
            raw_outputs = list(state.get("agent_outputs", []))
            raw_outputs.sort(key=lambda item: (order_map.get(str(item.get("npc_name", "")), 999), str(item.get("stage", ""))))
            intermediate_outputs = [
                MultiChatAgentOutput(**item)
                for item in raw_outputs
            ]
        return MultiChatResponse(
            mode=str(state.get("script_id", request.mode)),
            selected_agents=list(state.get("selected_agents", [])),
            execution_order=execution_order,
            aggregation_strategy=str(state.get("aggregation_strategy", "")),
            final_answer=str(state.get("final_answer", "")),
            intermediate_outputs=intermediate_outputs,
            node_trace=list(state.get("node_trace", [])),
            success=True,
            langgraph_available=bool(state.get("langgraph_runtime", npc_mgr.langgraph_multi.available)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"multi_chat 失败: {str(e)}"
        )

@app.get("/npcs", response_model=NPCListResponse)
async def list_npcs():
    """获取所有NPC列表"""
    npc_mgr, _ = get_managers()
    
    npcs_data = npc_mgr.get_all_npcs()
    npcs = [NPCInfo(**npc) for npc in npcs_data]
    
    return NPCListResponse(
        npcs=npcs,
        total=len(npcs)
    )

@app.get("/npcs/status", response_model=NPCStatusResponse)
async def get_npcs_status():
    """获取所有NPC的当前状态
    
    返回批量生成的NPC对话内容,用于显示NPC的自主行为
    """
    _, state_mgr = get_managers()
    
    state = state_mgr.get_current_state()
    
    return NPCStatusResponse(
        dialogues=state["dialogues"],
        last_update=state["last_update"],
        next_update_in=state["next_update_in"]
    )

@app.post("/npcs/status/refresh")
async def refresh_npcs_status():
    """强制刷新NPC状态
    
    立即触发一次批量对话生成
    """
    _, state_mgr = get_managers()
    
    await state_mgr.force_update()
    state = state_mgr.get_current_state()
    
    return {
        "message": "NPC状态已刷新",
        "dialogues": state["dialogues"]
    }

@app.get("/npcs/{npc_name}")
async def get_npc_info(npc_name: str):
    """获取指定NPC的详细信息"""
    npc_mgr, state_mgr = get_managers()

    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    # 添加当前对话
    current_dialogue = state_mgr.get_npc_dialogue(npc_name)
    npc_info["current_dialogue"] = current_dialogue

    return npc_info

@app.get("/npcs/{npc_name}/memories")
async def get_npc_memories(npc_name: str, limit: int = 10):
    """获取NPC的记忆列表

    Args:
        npc_name: NPC名称
        limit: 返回的记忆数量限制 (默认10条)

    Returns:
        NPC的记忆列表
    """
    npc_mgr, _ = get_managers()

    # 验证NPC是否存在
    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    try:
        memories = npc_mgr.get_npc_memories(npc_name, limit=limit)

        return {
            "npc_name": npc_name,
            "memories": memories,
            "total": len(memories)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取记忆失败: {str(e)}"
        )

@app.get("/npcs/{npc_name}/summary-debug")
async def get_npc_summary_debug(npc_name: str, player_id: str | None = None):
    """获取 sequence6 摘要压缩治理状态。"""
    npc_mgr, _ = get_managers()

    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    try:
        debug_info = npc_mgr.get_summary_debug_info(npc_name, player_id=player_id)
        return debug_info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取摘要调试信息失败: {str(e)}"
        )

@app.delete("/npcs/{npc_name}/memories")
async def clear_npc_memories(npc_name: str, memory_type: str = None):
    """清空NPC的记忆 (用于测试)

    Args:
        npc_name: NPC名称
        memory_type: 记忆类型 (working/episodic), 不指定则清空所有

    Returns:
        操作结果
    """
    npc_mgr, _ = get_managers()

    # 验证NPC是否存在
    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    try:
        npc_mgr.clear_npc_memory(npc_name, memory_type)

        return {
            "message": f"已清空{npc_name}的记忆",
            "npc_name": npc_name,
            "memory_type": memory_type or "all"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清空记忆失败: {str(e)}"
        )

@app.get("/npcs/{npc_name}/affinity")
async def get_npc_affinity(npc_name: str, player_id: str = "player"):
    """获取NPC对玩家的好感度

    Args:
        npc_name: NPC名称
        player_id: 玩家ID (默认为"player")

    Returns:
        好感度信息
    """
    npc_mgr, _ = get_managers()

    # 验证NPC是否存在
    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    try:
        affinity_info = npc_mgr.get_npc_affinity(npc_name, player_id)

        return {
            "npc_name": npc_name,
            "player_id": player_id,
            **affinity_info
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取好感度失败: {str(e)}"
        )

@app.get("/affinities")
async def get_all_affinities(player_id: str = "player"):
    """获取所有NPC对玩家的好感度

    Args:
        player_id: 玩家ID (默认为"player")

    Returns:
        所有NPC的好感度信息
    """
    npc_mgr, _ = get_managers()

    try:
        affinities = npc_mgr.get_all_affinities(player_id)

        return {
            "player_id": player_id,
            "affinities": affinities
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取好感度失败: {str(e)}"
        )

@app.put("/npcs/{npc_name}/affinity")
async def set_npc_affinity(npc_name: str, affinity: float, player_id: str = "player"):
    """设置NPC对玩家的好感度 (用于测试)

    Args:
        npc_name: NPC名称
        affinity: 好感度值 (0-100)
        player_id: 玩家ID (默认为"player")

    Returns:
        操作结果
    """
    npc_mgr, _ = get_managers()

    # 验证NPC是否存在
    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    # 验证好感度范围
    if affinity < 0 or affinity > 100:
        raise HTTPException(
            status_code=400,
            detail="好感度必须在0-100之间"
        )

    try:
        npc_mgr.set_npc_affinity(npc_name, affinity, player_id)
        affinity_info = npc_mgr.get_npc_affinity(npc_name, player_id)

        return {
            "message": f"已设置{npc_name}对玩家的好感度",
            "npc_name": npc_name,
            "player_id": player_id,
            **affinity_info
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"设置好感度失败: {str(e)}"
        )

@app.get("/knowledge/search")
async def search_knowledge(q: str, limit: int = 3):
    """调试外部知识检索结果

    保持为只读接口，不影响现有 chat 协议。
    """
    npc_mgr, _ = get_managers()

    if not q.strip():
        raise HTTPException(
            status_code=400,
            detail="查询词不能为空"
        )

    try:
        hits = npc_mgr.search_knowledge(q, limit=limit)
        return {
            "query": q,
            "hits": hits,
            "total": len(hits)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"知识检索失败: {str(e)}"
        )

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print("\n🚀 启动赛博小镇后端服务...")
    print(f"📍 监听地址: {settings.API_HOST}:{settings.API_PORT}")
    print(f"📖 访问文档: http://localhost:{settings.API_PORT}/docs\n")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # 开发模式自动重载
        log_level="info"
    )
