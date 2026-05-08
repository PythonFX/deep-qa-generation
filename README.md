# rag-qa-builder

从 Markdown / TXT 文档生成高质量 RAG QA benchmark 数据集。

当前推荐使用新的 `generate-deep` 流程。它不再以 concept / fact / relation 图作为主干，而是先把文档切成语义 section，再抽取可出题的 evidence cards，由 LLM 生成自然、可检索、可验证的 QA。

## 推荐用法

```bash
.venv/bin/python main.py generate-deep \
  --input ./docs \
  --output ./output \
  --config ./configs/default.yaml \
  --target-size 200 \
  --force
```

参数说明：

- `--input`: 输入文档目录或文件，支持 `.md` / `.markdown` / `.txt`
- `--output`: 输出目录
- `--config`: 配置文件，可省略
- `--target-size`: 目标 QA 数量上限。实际数量取决于可用 evidence、LLM 生成结果和校验通过率
- `--force`: 清理输出目录里的已有文件后重新生成
- `--dry-run`: 禁用 LLM，只测试读取、清洗、切分和 evidence card fallback；由于问题现在只由 LLM 生成，dry-run 通常不会产生最终 QA

## 新流程

```text
raw documents
→ PDF-like text cleanup
→ semantic sectioning
→ evidence cards
→ LLM question plans
→ grounded answers
→ validation
→ dataset.deep.final.jsonl
```

核心设计：

- **语义 sectioning**: 先清理 PDF 复制文本里的排版换行、页码、重复页眉页脚、版权/参考文献噪声，再按句子做轻量语义切分。
- **数量自适应**: section size 只作为硬性上下限；主要目标是让 section 数量落在合理范围内。
- **Evidence cards**: 抽取 claim、mechanism、condition、tradeoff、implication、caveat、result 等真正适合出题的证据单元。
- **LLM-only question plans**: 问题只由 LLM 生成，不再使用本地 cross-card 模板，避免“AAAA 的论述和 BBB 的论述之间有什么关系”这种硬凑题。
- **自然 RAG 问题**: 问题应像真实用户查询，足够明确但不过度泄题。例如优先生成 `What mechanisms make the Transformer architecture better than recurrent neural networks?`，而不是把所有机制塞进括号里。
- **Grounded validation**: 最终 QA 必须可回答、有证据支撑、问题自包含、非重复，并且具有一定推理价值。

## 关键配置

`configs/default.yaml` 中的新流程相关配置：

```yaml
semantic_sectioning:
  enabled: true
  min_section_chars: 1200
  target_section_chars: 3800
  max_section_chars: 6500
  target_sections_per_40k_chars: 10
  target_count_tolerance: 0.35
  overlap_sentences: 0

qa_generation:
  target_size: 200
```

调参建议：

- 如果 section 太少，提高 `target_sections_per_40k_chars`，或降低 `max_section_chars`
- 如果 section 太多，降低 `target_sections_per_40k_chars`，或提高 `min_section_chars`
- 如果最终 QA 数量太少，先检查 `evidence_cards.json` 和 `question_plans.deep.json` 的数量，而不是只调 `target_size`

## 新流程输出

`generate-deep` 会生成：

- `documents.json`: 原始读取文档
- `documents.cleaned.json`: 清洗后的文档文本
- `document_structure.json`: 语义 section 切分结果
- `evidence_cards.json`: 可出题证据卡
- `evidence_spans.json`: evidence card 绑定的原文证据
- `question_plans.deep.json`: LLM 生成的问题计划
- `qa_candidates.deep.jsonl`: QA 候选
- `qa_validated.deep.jsonl`: QA 校验结果
- `qa_rejected.deep.jsonl`: 未通过校验的 QA
- `dataset.deep.final.jsonl`: 最终 RAG QA benchmark
- `errors.jsonl`: 读取或处理中出现的问题

排查数量不足时，建议按这个漏斗看：

```text
document_structure.json
→ evidence_cards.json
→ question_plans.deep.json
→ qa_candidates.deep.jsonl
→ qa_validated.deep.jsonl
→ dataset.deep.final.jsonl
```

## Legacy 流程

旧流程仍保留，命令如下：

```bash
.venv/bin/python main.py generate \
  --input ./docs \
  --output ./output_legacy \
  --config ./configs/default.yaml
```

旧流程大致为：

```text
documents
→ structure mapper
→ concepts
→ facts
→ concept-fact graph
→ fact combinations
→ question blueprints
→ QA
```

也可以分阶段运行：

```bash
.venv/bin/python main.py build-structure --input ./docs --output ./output
.venv/bin/python main.py extract-concepts --input ./docs --output ./output
.venv/bin/python main.py extract-facts --input ./docs --output ./output
.venv/bin/python main.py build-graph --input ./docs --output ./output
.venv/bin/python main.py analyze-combinations --input ./docs --output ./output
.venv/bin/python main.py generate-qa --input ./docs --output ./output
.venv/bin/python main.py validate-qa --input ./docs --output ./output
.venv/bin/python main.py export-final --input ./docs --output ./output
```

Legacy 输出包括 `concepts.json`、`facts.json`、`concept_fact_graph.json`、`question_blueprints.json`、`dataset.final.jsonl` 等。新项目建议优先使用 `generate-deep`。
