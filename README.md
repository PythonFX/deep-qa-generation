# rag-qa-builder

根据 Markdown / TXT 文档构建可追溯的知识层，并进一步生成 RAG QA benchmark 数据集。

## 使用

```bash
python3 main.py generate \
  --input ./docs \
  --output ./output \
  --config ./configs/default.yaml
```

支持分阶段命令：

```bash
python3 main.py build-structure --input ./docs --output ./output
python3 main.py extract-concepts --input ./docs --output ./output
python3 main.py extract-facts --input ./docs --output ./output
python3 main.py build-graph --input ./docs --output ./output
python3 main.py analyze-combinations --input ./docs --output ./output
python3 main.py generate-qa --input ./docs --output ./output
python3 main.py validate-qa --input ./docs --output ./output
python3 main.py export-final --input ./docs --output ./output
```

## 设计说明

- Reader: 递归读取 `.md` / `.markdown` / `.txt`
- Structure Mapper: Markdown 标题映射与 TXT 标题启发式识别
- Concept / Fact: 规则优先，LLM 可选增强
- LLM: 复用已安装的 `llm_client` 包，并带缓存
- Validator: 基础 answerability / faithfulness / ambiguity / duplication 检查

## 输出

生成：

- `documents.json`
- `document_structure.json`
- `concepts.raw.json`
- `concepts.json`
- `facts.raw.json`
- `facts.json`
- `evidence.raw.json`
- `evidence.json`
- `concept_fact_graph.json`
- `fact_combinations.json`
- `question_blueprints.json`
- `qa_candidates.jsonl`
- `qa_validated.jsonl`
- `qa_rejected.jsonl`
- `dataset.final.jsonl`
- `errors.jsonl`
