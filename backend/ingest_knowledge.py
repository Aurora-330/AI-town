"""外部知识库入库脚本"""

import argparse

from knowledge_retriever import KnowledgeRetriever


def main():
    """执行知识库入库"""
    parser = argparse.ArgumentParser(description="Ingest knowledge_base documents into Qdrant")
    parser.parse_args()

    retriever = KnowledgeRetriever()
    if not retriever.available():
        raise SystemExit("知识检索器不可用，请先检查Qdrant和Embedding配置。")

    result = retriever.ingest()
    print(
        f"✅ 知识库入库完成: documents={result['documents']}, chunks={result['chunks']}, "
        f"collection={retriever.collection_name}"
    )


if __name__ == "__main__":
    main()
