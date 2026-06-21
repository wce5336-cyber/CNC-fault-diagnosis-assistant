RAG 知识库使用说明
==================
分块文件: chunks.jsonl (1430 块)
向量库: ../rag_chroma (ChromaDB，需先运行 build_rag_index.py)
推荐检索字段: text
metadata.type: alarm_reference | maintenance_manual | diagnostic_qa

微调数据: ../cnc_diagnosis_sft.json
联合推理:
  1. python -m src.build_rag_index --reset
  2. python -m src.rag_chat --interactive
