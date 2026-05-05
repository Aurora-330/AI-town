"""配置文件"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# 显式加载 backend/.env，避免直接运行 main.py 时依赖外部 shell 手动 export。
BACKEND_DIR = Path(__file__).resolve().parent
ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

# chapter15 当前只需要本地 embedding；如果用户未显式配置，则默认强制走本地链路。
os.environ.setdefault("EMBED_MODEL_TYPE", "local")
os.environ.setdefault("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

class Settings:
    """应用配置"""
    
    # API配置
    API_TITLE = "赛博小镇 API"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # NPC配置
    NPC_UPDATE_INTERVAL = 30  # NPC状态更新间隔(秒)
    
    # LLM配置 (从环境变量读取)
    # HelloAgents框架使用自定义LLM配置,不需要OPENAI_API_KEY
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))
    TOKENIZER_MODEL_ID: str = os.getenv("TOKENIZER_MODEL_ID", LLM_MODEL_ID)
    TOKENIZER_TRUST_REMOTE_CODE: bool = os.getenv("TOKENIZER_TRUST_REMOTE_CODE", "true").lower() == "true"

    # Embedding配置
    EMBED_MODEL_TYPE: str = os.getenv("EMBED_MODEL_TYPE", "local")
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # Qdrant配置
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT", "30"))
    QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))

    # 外部知识库RAG配置
    KNOWLEDGE_ENABLED: bool = os.getenv("KNOWLEDGE_ENABLED", "true").lower() == "true"
    KNOWLEDGE_COLLECTION: str = os.getenv("KNOWLEDGE_COLLECTION", "hello_agents_knowledge")
    KNOWLEDGE_TOP_K: int = int(os.getenv("KNOWLEDGE_TOP_K", "3"))
    KNOWLEDGE_CHUNK_SIZE: int = int(os.getenv("KNOWLEDGE_CHUNK_SIZE", "420"))
    KNOWLEDGE_CHUNK_OVERLAP: int = int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "60"))
    KNOWLEDGE_BASE_DIR: str = os.getenv(
        "KNOWLEDGE_BASE_DIR",
        str(BACKEND_DIR.parent / "knowledge_base")
    )

    # CORS配置
    CORS_ORIGINS = ["*"]  # 生产环境应限制具体域名

    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.LLM_API_KEY:
            print("⚠️  警告: 未设置LLM_API_KEY环境变量")
            print("   请在.env文件中配置LLM_API_KEY")
            print("   示例: LLM_API_KEY=\"your-api-key\"")
            return False

        print(f"✅ LLM配置:")
        print(f"   模型: {cls.LLM_MODEL_ID}")
        print(f"   服务地址: {cls.LLM_BASE_URL}")
        print("✅ Tokenizer配置:")
        print(f"   模型: {cls.TOKENIZER_MODEL_ID}")
        print("✅ Embedding配置:")
        print(f"   类型: {cls.EMBED_MODEL_TYPE}")
        print(f"   模型: {cls.EMBED_MODEL_NAME}")
        return True

settings = Settings()
