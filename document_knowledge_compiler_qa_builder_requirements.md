# Document Knowledge Compiler & QA Benchmark Builder 需求文档

## 1. 项目背景

我们希望开发一个程序，根据已有文档自动生成高质量 RAG QA 问答测试集。输入文档暂时只支持纯文本类文件，包括：

- Markdown (`.md`, `.markdown`)
- Plain Text (`.txt`)

本项目暂不处理 PDF、DOCX、PPTX、HTML 等复杂格式，避免将开发精力浪费在文档解析上。

传统方案通常是：

```text
文档 → chunking → 每个 chunk 生成 QA
```

但这种方式容易产生低质量、机械化、局部事实型问题，无法很好评估 RAG 系统的真实能力。

本项目采用新的主线：

```text
文档 → 文档结构索引 → 核心概念 → 事实抽取 → 概念-事实图 → 事实组合分析 → 问题蓝图 → QA 生成 → QA 校验 → 测试集导出
```

也就是说，本项目不是直接从文档生成 QA，而是先把文档编译成一个可追溯、可组合、可检查的知识层，再基于知识层生成更高质量的 QA Benchmark。

---

## 2. 项目目标

### 2.1 核心目标

开发一个命令行程序，能够从 Markdown/TXT 文档中生成：

1. 文档结构索引 `document_structure.json`
2. 核心概念列表 `concepts.json`
3. 事实列表 `facts.json`
4. 证据列表 `evidence.json`
5. 概念-事实关系 `concept_fact_graph.json`
6. 事实组合结果 `fact_combinations.json`
7. 问题蓝图 `question_blueprints.json`
8. QA 候选集 `qa_candidates.jsonl`
9. QA 校验结果 `qa_validated.jsonl`
10. 最终 QA 测试集 `dataset.final.jsonl`

### 2.2 质量目标

生成的 QA 应该满足：

- 问题自然，像真实用户会问的问题
- 问题不应机械地使用“根据文档”、“根据上文”这类表达
- 问题应覆盖核心概念、关键事实、流程、约束、对比、因果、条件、权衡等内容
- 参考答案必须有证据支持
- 每个 QA 应尽可能保留相关 concept、fact、evidence 元数据
- 支持生成更综合的问题，而不只是单点事实问题
- 支持后续用于 RAG 检索、生成、引用、评测等环节

### 2.3 非目标

当前版本不做：

- PDF 解析
- DOCX/PPTX 解析
- OCR
- Web UI
- 人工审核后台
- 完整知识图谱数据库
- 向量数据库集成
- RAG 系统本身
- 自动调用被测 RAG 系统做评估

---

## 3. 总体架构

### 3.1 Pipeline

```text
Input Documents
  ↓
Document Reader
  ↓
Document Structure Mapper
  ↓
Concept Extractor
  ↓
Concept Canonicalizer
  ↓
Fact Extractor
  ↓
Evidence Binder
  ↓
Concept-Fact Graph Builder
  ↓
Fact Combination Analyzer
  ↓
Question Blueprint Generator
  ↓
QA Generator
  ↓
QA Validator
  ↓
Dataset Exporter
```

### 3.2 推荐目录结构

```text
rag_qa_builder/
├── pyproject.toml
├── README.md
├── configs/
│   └── default.yaml
├── rag_qa_builder/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai_client.py
│   │   └── prompt_runner.py
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── markdown_reader.py
│   │   └── text_reader.py
│   ├── compiler/
│   │   ├── __init__.py
│   │   ├── structure_mapper.py
│   │   ├── concept_extractor.py
│   │   ├── concept_canonicalizer.py
│   │   ├── fact_extractor.py
│   │   └── evidence_binder.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── concept_fact_graph.py
│   │   └── graph_builder.py
│   ├── analyzer/
│   │   ├── __init__.py
│   │   ├── fact_combination_analyzer.py
│   │   ├── pattern_matcher.py
│   │   └── difficulty_estimator.py
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── question_blueprint_generator.py
│   │   ├── qa_generator.py
│   │   └── answer_generator.py
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── evidence_checker.py
│   │   ├── answerability_checker.py
│   │   ├── ambiguity_checker.py
│   │   ├── hallucination_checker.py
│   │   └── duplicate_checker.py
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── json_exporter.py
│   │   └── jsonl_exporter.py
│   └── utils/
│       ├── __init__.py
│       ├── ids.py
│       ├── json_utils.py
│       ├── text_utils.py
│       └── logging.py
└── tests/
    ├── test_readers.py
    ├── test_structure_mapper.py
    ├── test_models.py
    └── fixtures/
        ├── sample.md
        └── sample.txt
```

---

## 4. 技术栈要求

### 4.1 语言与版本

- Python 3.10+

### 4.2 推荐依赖

```toml
[project]
dependencies = [
  "pydantic>=2.0.0",
  "typer>=0.12.0",
  "pyyaml>=6.0.0",
  "rich>=13.0.0",
  "openai>=1.0.0",
  "python-dotenv>=1.0.0",
  "tenacity>=8.0.0",
  "orjson>=3.9.0",
  "numpy>=1.24.0"
]
```

可选依赖：

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "ruff>=0.5.0",
  "mypy>=1.0.0"
]
```

### 4.3 LLM 接入要求

LLM的调用参考 /Users/vincent/Projects/Claude/llm_client 路径下的实现
该实现支持多个模型 我们可以起手先用minimax的兼容anthropic的模型 
该仓库的实现让多个模型对外的接口一致 所以后续换模型也很方便

注意：

- 不要把项目强绑定某一个模型
- LLM 输出必须要求 JSON 格式
- 对 LLM 输出需要做 JSON 解析、重试和错误处理
- 对关键步骤需要支持缓存，避免反复调用 LLM

---

## 5. 命令行接口

### 5.1 主命令

```bash
rag-qa-builder generate \
  --input ./docs \
  --output ./output \
  --config ./configs/default.yaml
```

### 5.2 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--input` | 是 | 输入文件或目录 |
| `--output` | 是 | 输出目录 |
| `--config` | 否 | 配置文件路径 |
| `--language` | 否 | 生成语言，默认 `zh` |
| `--target-size` | 否 | 目标 QA 数量 |
| `--force` | 否 | 是否覆盖已有输出 |
| `--resume` | 否 | 是否从已有中间产物继续执行 |
| `--dry-run` | 否 | 只检查配置和输入，不调用 LLM |

### 5.3 分阶段命令

为了方便调试，必须支持分阶段执行：

```bash
rag-qa-builder build-structure --input ./docs --output ./output
rag-qa-builder extract-concepts --output ./output
rag-qa-builder extract-facts --output ./output
rag-qa-builder build-graph --output ./output
rag-qa-builder analyze-combinations --output ./output
rag-qa-builder generate-qa --output ./output
rag-qa-builder validate-qa --output ./output
rag-qa-builder export-final --output ./output
```

---

## 6. 配置文件设计

默认配置示例：

```yaml
project:
  name: rag_qa_dataset
  language: zh

input:
  file_types:
    - .md
    - .markdown
    - .txt
  encoding: utf-8

llm:
  provider: openai_compatible
  model: gpt-4.1-mini
  base_url: null
  temperature: 0.2
  max_tokens: 4096
  retry_times: 3
  request_timeout_seconds: 120

structure:
  markdown_heading_as_section: true
  txt_section_detection: true
  min_section_chars: 300
  max_section_chars_for_single_llm_call: 50000

concept_extraction:
  max_concepts_per_doc: 80
  min_importance: 0.4
  include_aliases: true
  merge_similar_concepts: true

fact_extraction:
  max_facts_per_concept: 20
  min_confidence: 0.65
  allowed_fact_types:
    - definition
    - claim
    - constraint
    - procedure
    - comparison
    - condition
    - cause_effect
    - numeric
    - example
    - warning
    - decision
    - config
    - api_behavior

combination:
  max_facts_per_combination: 4
  min_combination_score: 0.7
  enabled_patterns:
    - common_goal_with_mechanism_difference
    - cause_effect_chain
    - condition_action
    - method_tradeoff
    - concept_comparison
    - procedure_chain
    - troubleshooting
    - constraint_reasoning

qa_generation:
  target_size: 200
  question_language: zh
  avoid_phrases:
    - 根据文档
    - 根据上文
    - 文中提到
  distribution:
    single_fact: 0.25
    concept_definition: 0.10
    comparison: 0.15
    cause_effect: 0.10
    procedure: 0.10
    constraint: 0.10
    scenario: 0.10
    multi_fact_synthesis: 0.10

validation:
  enabled: true
  min_overall_score: 4.0
  require_answerability: true
  require_faithfulness: true
  reject_ambiguous_question: true
  reject_external_knowledge: true
  deduplicate_questions: true
  duplicate_similarity_threshold: 0.88
```

---

## 7. 数据模型

所有核心数据模型使用 Pydantic 定义。

### 7.1 Document

```python
class Document(BaseModel):
    doc_id: str
    file_path: str
    file_name: str
    file_type: str
    text: str
    metadata: dict = Field(default_factory=dict)
```

### 7.2 DocumentSection

```python
class DocumentSection(BaseModel):
    section_id: str
    doc_id: str
    title: str | None = None
    level: int | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    char_start: int
    char_end: int
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
```

说明：

- 这里的 section 不是传统 chunk
- section 是文档结构索引，用于定位上下文、抽取概念、绑定 evidence
- Markdown 使用标题层级生成 section
- TXT 使用简单规则识别标题；无法识别时，整个文档作为一个 section

### 7.3 Concept

```python
class Concept(BaseModel):
    concept_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    concept_type: str
    definition: str | None = None
    importance: float = 0.0
    source_section_ids: list[str] = Field(default_factory=list)
    related_fact_ids: list[str] = Field(default_factory=list)
    related_concept_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
```

`concept_type` 可选值：

```text
method | component | process | metric | rule | constraint | entity | term | config | error | decision | dataset | evaluation | other
```

### 7.4 Evidence

```python
class Evidence(BaseModel):
    evidence_id: str
    doc_id: str
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    char_start: int | None = None
    char_end: int | None = None
    source_hint: str | None = None
```

说明：

- evidence 是事实的证据来源
- evidence 是动态片段，不是预切 chunk
- 每个进入 QA 生成阶段的 fact 必须有 evidence

### 7.5 Fact

```python
class Fact(BaseModel):
    fact_id: str
    fact_type: str
    subject_concept_id: str | None = None
    related_concept_ids: list[str] = Field(default_factory=list)
    statement: str
    structured: dict = Field(default_factory=dict)
    qualifiers: dict = Field(default_factory=dict)
    confidence: float = 0.0
    importance: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    depends_on_fact_ids: list[str] = Field(default_factory=list)
    contrasts_with_fact_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
```

`fact_type` 可选值：

```text
definition | claim | constraint | procedure | comparison | condition | cause_effect | numeric | example | warning | decision | config | api_behavior | other
```

### 7.6 ConceptFactRelation

```python
class ConceptFactRelation(BaseModel):
    relation_id: str
    concept_id: str
    fact_id: str
    relation_type: str
    confidence: float = 0.0
```

`relation_type` 可选值：

```text
subject_of | mentioned_by | defines | constrains | explains | compares | causes | requires | configures | example_of | other
```

### 7.7 FactCombination

```python
class FactCombination(BaseModel):
    combination_id: str
    fact_ids: list[str]
    concept_ids: list[str]
    pattern: str
    rationale: str
    expected_question_type: str
    expected_answer_points: list[str]
    difficulty: str
    score: float
```

### 7.8 QuestionBlueprint

```python
class QuestionBlueprint(BaseModel):
    blueprint_id: str
    source_combination_id: str | None = None
    pattern: str
    fact_ids: list[str]
    concept_ids: list[str]
    intended_question: str
    expected_answer_points: list[str]
    difficulty: str
    question_type: str
    answer_requirements: list[str] = Field(default_factory=list)
    unsupported_answer_patterns: list[str] = Field(default_factory=list)
```

### 7.9 QAPair

```python
class QAPair(BaseModel):
    qa_id: str
    question: str
    reference_answer: str
    concept_ids: list[str]
    fact_ids: list[str]
    evidence_ids: list[str]
    question_type: str
    difficulty: str
    answer_requirements: list[str] = Field(default_factory=list)
    unsupported_answer_patterns: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
```

### 7.10 QAValidationResult

```python
class QAValidationResult(BaseModel):
    qa_id: str
    is_answerable: bool
    is_faithful: bool
    is_ambiguous: bool
    requires_external_knowledge: bool
    has_hallucination: bool
    scores: dict
    issues: list[str] = Field(default_factory=list)
    passed: bool
```

---

## 8. 模块详细需求

## 8.1 Document Reader

### 功能

读取输入目录中的 `.md`, `.markdown`, `.txt` 文件。

### 要求

- 递归扫描目录
- 忽略隐藏文件和空文件
- 支持 UTF-8
- 如果读取失败，记录错误并跳过
- 每个文档生成稳定 `doc_id`

### 输出

`documents.json`

---

## 8.2 Document Structure Mapper

### 功能

把文档转为结构索引。

### Markdown 处理

- 按 `#`, `##`, `###` 等标题识别层级
- 每个标题下内容形成一个 section
- 保留 `section_path`
- 如果文档没有标题，则整个文档作为一个 section

### TXT 处理

简单启发式识别标题：

- 独占一行
- 行长度较短
- 上下有空行
- 以数字编号开头，例如 `1. xxx`, `2.3 xxx`
- 全大写英文标题

如果无法识别，整个文档作为一个 section。

### 输出

`document_structure.json`

---

## 8.3 Concept Extractor

### 功能

从文档结构中抽取核心概念。

### 输入

- `documents.json`
- `document_structure.json`

### 抽取策略

分两层抽取：

1. 文档级概念抽取
2. section 级概念抽取

然后合并结果。

### LLM Prompt 要求

概念包括但不限于：

- 方法
- 系统组件
- 业务对象
- 流程
- 指标
- 约束
- 配置项
- 错误类型
- 设计决策
- 评估维度

输出必须是 JSON。

示例输出：

```json
{
  "concepts": [
    {
      "name": "Late Chunking",
      "aliases": ["延迟切块"],
      "concept_type": "method",
      "definition": "...",
      "importance": 0.92,
      "source_section_ids": ["sec_001"]
    }
  ]
}
```

### 输出

`concepts.raw.json`

---

## 8.4 Concept Canonicalizer

### 功能

合并相同或近似概念，生成规范概念列表。

### 处理规则

- 合并中英文别名
- 合并大小写差异
- 合并缩写和全称
- 合并明显同义概念
- 保留 aliases
- 合并 source_section_ids

### 实现建议

第一版可以使用：

- 字符串规范化
- 简单相似度
- LLM 判断是否同一概念

### 输出

`concepts.json`

---

## 8.5 Fact Extractor

### 功能

围绕每个核心概念抽取事实。

### 输入

- `concepts.json`
- `document_structure.json`

### 关键要求

每个 fact 必须：

- 尽可能 atomic
- 有明确 statement
- 有 fact_type
- 有 confidence
- 有 importance
- 有 evidence
- 不得引入文档外知识

### Fact 类型

```text
definition
claim
constraint
procedure
comparison
condition
cause_effect
numeric
example
warning
decision
config
api_behavior
other
```

### LLM Prompt 原则

```text
Extract factual statements about the target concept.

Rules:
- Each fact must be atomic.
- Each fact must be supported by evidence from the document.
- Do not infer beyond the document.
- Preserve numbers, conditions, exceptions, and constraints.
- Return only facts that are useful for downstream QA generation.
- If the text does not provide enough evidence, return an empty list.
```

### 输出

- `facts.raw.json`
- `evidence.raw.json`

---

## 8.6 Evidence Binder

### 功能

清洗和规范化 evidence。

### 要求

- 为每个 evidence 分配 `evidence_id`
- evidence text 必须来自原文
- 记录 doc_id、section_id、section_path
- 尽量记录 char_start 和 char_end
- 去除重复 evidence
- 删除没有 evidence 的 fact，或者标记为 invalid

### 输出

- `facts.json`
- `evidence.json`

---

## 8.7 Concept-Fact Graph Builder

### 功能

建立概念和事实之间的关系。

### 输入

- `concepts.json`
- `facts.json`

### 输出

`concept_fact_graph.json`

格式示例：

```json
{
  "nodes": {
    "concepts": [...],
    "facts": [...]
  },
  "relations": [
    {
      "relation_id": "rel_001",
      "concept_id": "c_001",
      "fact_id": "f_001",
      "relation_type": "defines",
      "confidence": 0.95
    }
  ]
}
```

---

## 8.8 Fact Combination Analyzer

### 功能

分析哪些 facts 可以组合，形成更综合的问题。

### 支持的组合模式

#### 1. common_goal_with_mechanism_difference

多个方法解决同一个问题，但机制不同。

生成问题类型：对比、综合。

#### 2. cause_effect_chain

多个 facts 构成因果链。

生成问题类型：原因分析、影响分析。

#### 3. condition_action

条件与动作之间存在关系。

生成问题类型：场景题、条件题。

#### 4. method_tradeoff

某个方法有收益和代价。

生成问题类型：权衡题。

#### 5. concept_comparison

两个概念存在相似点和差异点。

生成问题类型：比较题。

#### 6. procedure_chain

多个步骤构成流程。

生成问题类型：流程题。

#### 7. troubleshooting

错误现象、原因、解决方式。

生成问题类型：排错题。

#### 8. constraint_reasoning

约束、原因、适用范围。

生成问题类型：约束理解题。

### 实现方式

第一版可以混合使用：

- 规则匹配
- concept overlap
- fact_type pattern
- LLM 判断组合价值

### 输出

`fact_combinations.json`

---

## 8.9 Question Blueprint Generator

### 功能

根据 fact combination 生成问题蓝图。

### 要求

蓝图不是最终问题，而是问题的结构化计划。

示例：

```json
{
  "blueprint_id": "qb_001",
  "source_combination_id": "comb_001",
  "pattern": "common_goal_with_mechanism_difference",
  "fact_ids": ["f_001", "f_002", "f_003"],
  "concept_ids": ["c_001", "c_002"],
  "intended_question": "比较两个方法如何解决同一问题",
  "expected_answer_points": [
    "二者共同目标是减少上下文割裂",
    "方法 A 通过更大检索单元解决",
    "方法 B 通过长上下文 embedding 解决",
    "二者可以互补"
  ],
  "difficulty": "hard",
  "question_type": "multi_fact_comparison"
}
```

### 输出

`question_blueprints.json`

---

## 8.10 QA Generator

### 功能

根据 question blueprint 生成自然语言问题和参考答案。

### 要求

问题要求：

- 自然
- 像真实用户提问
- 避免“根据文档”、“根据上文”、“文中提到”
- 不要暴露 fact_id、concept_id
- 不要把答案直接写进问题

答案要求：

- 只基于相关 facts 和 evidence
- 完整覆盖 expected_answer_points
- 不引入外部知识
- 尽量清晰、简洁
- 如果证据不足，标记为 invalid

### 输出

`qa_candidates.jsonl`

---

## 8.11 QA Validator

### 功能

校验 QA 质量。

### 校验项

#### 1. Answerability

问题是否能根据 evidence 回答。

#### 2. Faithfulness

参考答案是否完全被 evidence 支持。

#### 3. Ambiguity

问题是否模糊、有多种解释。

#### 4. External Knowledge

回答是否依赖外部知识。

#### 5. Hallucination

答案是否包含 evidence 不支持的信息。

#### 6. Triviality

问题是否过于简单或没有测试价值。

#### 7. Duplication

问题是否和已有问题高度重复。

### 输出

- `qa_validated.jsonl`
- `qa_rejected.jsonl`

---

## 8.12 Dataset Exporter

### 功能

导出最终 QA 数据集。

### 输出格式

`dataset.final.jsonl`

每行格式：

```json
{
  "id": "qa_001",
  "question": "...",
  "reference_answer": "...",
  "reference_context": [
    {
      "evidence_id": "ev_001",
      "doc_id": "doc_001",
      "section_path": ["..."],
      "text": "..."
    }
  ],
  "concept_ids": ["c_001"],
  "fact_ids": ["f_001", "f_002"],
  "evidence_ids": ["ev_001"],
  "question_type": "multi_fact_comparison",
  "difficulty": "hard",
  "answer_requirements": ["..."],
  "unsupported_answer_patterns": ["..."]
}
```

---

## 9. Prompt 设计要求

所有 prompt 应集中管理，例如：

```text
rag_qa_builder/prompts/
├── extract_concepts.md
├── canonicalize_concepts.md
├── extract_facts.md
├── analyze_fact_combinations.md
├── generate_question_blueprint.md
├── generate_qa.md
└── validate_qa.md
```

### 9.1 通用要求

每个 prompt 必须：

- 明确输入
- 明确输出 JSON schema
- 明确禁止外部知识
- 明确失败时返回空数组或 invalid
- 明确不要输出解释性文本

### 9.2 JSON 解析失败处理

如果 LLM 输出不是合法 JSON：

1. 尝试提取 JSON block
2. 如果失败，调用 repair prompt
3. 最多重试 3 次
4. 仍失败则记录 error，并跳过当前任务

---

## 10. 缓存与可恢复执行

### 10.1 缓存

对每次 LLM 调用进行缓存。

缓存 key 可以由以下内容 hash 得到：

- prompt name
- model name
- input payload
- prompt version

缓存目录：

```text
output/.cache/llm_calls/
```

### 10.2 Resume

如果启用 `--resume`：

- 已存在的中间产物不重复生成
- 缺失的阶段继续执行
- 如果配置变化，给出 warning

---

## 11. 日志与错误处理

### 11.1 日志要求

使用 rich 输出友好日志：

- 当前阶段
- 输入文件数量
- 抽取 concept 数量
- 抽取 fact 数量
- 生成 combination 数量
- 生成 QA 数量
- 通过校验数量
- 被拒绝数量

### 11.2 错误处理

单个文件、单个概念、单个 LLM 调用失败，不应导致整个任务失败。

应记录：

```text
output/errors.jsonl
```

每条错误包含：

```json
{
  "stage": "extract_facts",
  "source_id": "c_001",
  "error_type": "json_parse_error",
  "message": "..."
}
```

---

## 12. 质量控制策略

### 12.1 Concept 层

- 概念可以高召回
- 允许多抽取
- 后续 canonicalizer 合并

### 12.2 Fact 层

- fact 必须高精度
- 没有 evidence 的 fact 不进入 QA 生成
- confidence 低于阈值的 fact 不进入组合分析

### 12.3 Combination 层

优先选择：

- 涉及多个核心概念的组合
- 涉及对比、因果、约束、流程、条件的组合
- 能形成真实用户问题的组合

### 12.4 QA 层

QA 必须通过：

- answerability
- faithfulness
- ambiguity check
- external knowledge check

---

## 13. MVP 验收标准

### 13.1 功能验收

给定一个包含 3 个 Markdown/TXT 文件的目录，程序能够生成：

- `documents.json`
- `document_structure.json`
- `concepts.json`
- `facts.json`
- `evidence.json`
- `concept_fact_graph.json`
- `fact_combinations.json`
- `question_blueprints.json`
- `qa_candidates.jsonl`
- `qa_validated.jsonl`
- `dataset.final.jsonl`

### 13.2 质量验收

人工抽查最终 `dataset.final.jsonl` 中至少 30 条 QA：

- 80% 以上问题自然，不像模板题
- 90% 以上答案能被 evidence 支持
- 80% 以上问题具有实际测试价值
- 至少 30% 是多 fact 综合问题
- 至少包含以下题型：
  - definition
  - comparison
  - cause_effect
  - procedure
  - constraint
  - scenario
  - multi_fact_synthesis

### 13.3 工程验收

- CLI 可运行
- 配置文件可生效
- 中间产物可落盘
- 支持 resume
- LLM 调用有缓存
- 单个失败不会中断全流程
- 有基础单元测试

---

## 14. 示例最终 QA

```json
{
  "id": "qa_001",
  "question": "为什么这个系统不直接从文档片段生成 QA，而是先抽取概念和事实再生成问题？",
  "reference_answer": "因为直接从文档片段生成 QA 容易产生机械、局部、低区分度的问题。先抽取概念和事实，可以构建可追溯的知识层，再通过事实组合生成对比、因果、流程、约束等更综合的问题，从而提升 QA 测试集的质量和评测价值。",
  "reference_context": [
    {
      "evidence_id": "ev_001",
      "doc_id": "doc_001",
      "section_path": ["项目背景"],
      "text": "传统方案通常是：文档 → chunking → 每个 chunk 生成 QA。但这种方式容易产生低质量、机械化、局部事实型问题。"
    },
    {
      "evidence_id": "ev_002",
      "doc_id": "doc_001",
      "section_path": ["总体架构"],
      "text": "本项目采用新的主线：文档 → 文档结构索引 → 核心概念 → 事实抽取 → 概念-事实图 → 事实组合分析 → 问题蓝图 → QA 生成。"
    }
  ],
  "concept_ids": ["c_001", "c_002"],
  "fact_ids": ["f_001", "f_002", "f_003"],
  "evidence_ids": ["ev_001", "ev_002"],
  "question_type": "constraint_reasoning",
  "difficulty": "medium",
  "answer_requirements": [
    "说明直接 chunk 生成 QA 的问题",
    "说明概念和事实层的作用",
    "说明事实组合对综合问题生成的价值"
  ],
  "unsupported_answer_patterns": [
    "只说因为模型上下文更大",
    "只说为了减少 token 成本",
    "引入文档没有提到的外部工具或框架"
  ]
}
```

---

## 15. 开发建议顺序

建议 Codex 按以下顺序实现：

1. 建立项目结构和 Pydantic models
2. 实现 CLI 框架
3. 实现 Markdown/TXT reader
4. 实现 document structure mapper
5. 实现 LLM client 抽象和缓存
6. 实现 concept extractor
7. 实现 concept canonicalizer
8. 实现 fact extractor 和 evidence binder
9. 实现 concept-fact graph builder
10. 实现 fact combination analyzer
11. 实现 question blueprint generator
12. 实现 QA generator
13. 实现 QA validator
14. 实现 final dataset exporter
15. 补充 tests 和 README

---

## 16. 关键实现提醒

1. 不要做传统 chunking。
2. section 只是文档结构索引，不是 RAG chunk。
3. 不要直接从 section 生成 QA。
4. QA 必须来自 facts 或 fact combinations。
5. 每个 fact 必须绑定 evidence。
6. 每个最终 QA 必须保留 concept_ids、fact_ids、evidence_ids。
7. 所有 LLM 输出必须结构化。
8. 所有中间产物都要落盘。
9. 支持 resume 和 cache。
10. 先保证数据质量，再追求生成数量。

---

## 17. 未来扩展方向

当前版本完成后，可继续扩展：

- PDF 支持
- DOCX/PPTX 支持
- HTML/Web 页面支持
- 多文档跨文档 fact combination
- embedding 去重
- retrieval difficulty 评估
- RAG 系统自动回归评测
- 人工审核 UI
- QA 数据集版本管理
- 知识图谱可视化
- Graph database 存储
- HuggingFace Dataset 导出
