# Taiwan Value Chain QA Datasets - Complete Summary

This workspace contains comprehensive QA datasets for evaluating LLMs on Taiwan's industrial value chain knowledge.

## Overview

We have created **Three types of QA tasks** with different difficulty levels:

### 1. Chain → Firms (Harder)
**Task**: Given a value chain, list all companies in it.
- **Questions**: 47 chains
- **Difficulty**: ⭐⭐⭐⭐ Very Hard

### 2. Firm → Chains (Easier)
**Task**: Given a company, list all value chains it belongs to.
- **Questions**: 2,363 local companies
- **Difficulty**: ⭐⭐ Moderate

### 3. Competitors analysis (Hardest)
Task: Given a company and its value chains, list all of its competitors across shared chains.
Questions: 666 local companies
Difficulty: ⭐⭐⭐⭐⭐⭐ Extremely Hard

---

## Dataset Files

### Chain → Firms QA Dataset

| File | Description | Questions | Avg Answer Size |
|------|-------------|-----------|-----------------|
| `chain_firms_qa.jsonl` | All value chains (all companies) | 47 | ~111 companies |
| `chain_firms_qa_local.jsonl` | All value chains (local only) | 47 | ~85 companies |

**Total unique chains**: 47
- All chains included regardless of size
- Range: 11-341 companies per chain
- Total company mentions: 3,257 (all) / 2,363 (local)

### Firm → Chains QA Datasets

| File | Description | Questions | Avg Answer Size |
|------|-------------|-----------|-----------------|
| `firm_chains_qa_local.jsonl` | Local companies only | 2,363 | 1.5 chains |
| `firm_chains_qa_foreign.jsonl` | Foreign companies only | 894 | 1.2 chains |

**Total unique companies**: 3,257
- Local: 2,363 (72.6%)
- Foreign: 894 (27.4%)

**Chain distribution**:
- 71% single-chain companies
- 29% multi-chain companies
- Max chains per company: 23

### Competitors Analysis QA Datasets

| File | Description | Questions | Avg Answer Size |
|------|-------------|-----------|-----------------|
| `competitors_qa_local.jsonl` | Local companies only | 666 | Varies (multi-chain aggregation) |
| `competitors_qa_foreign.jsonl` | Foreign companies only | 148 | Varies (multi-chain aggregation) |

**Total evaluated qa**: 814  
- Local: 666 (81.8%)  
- Foreign: 148 (18.2%)

**Competitor distribution**:
- Highly skewed by chain size  
- Multi-chain companies generate significantly larger competitor sets  
- Competitor count grows non-linearly with overlapping chains  
- Max competitors per company: depends on shared chain coverage

---

---

## Evaluation Metrics

All datasets support evaluation using:

1. **Recall**: `|Predicted ∩ Actual| / |Actual|`
   - More critical for Chain→Firms (large answer sets)

2. **Precision**: `|Predicted ∩ Actual| / |Predicted|`
   - Important to avoid hallucination

3. **F1 Score**: `2 × (P × R) / (P + R)`
   - Balanced metric

4. **mAP**: Mean Average Precision
   - For ranked predictions

5. **Exact Match Rate**: Perfectly correct predictions
   - Very strict metric

---

## Task Comparison

| Aspect | Chain → Firms | Firm → Chains | Competitors Analysis |
|--------|---------------|---------------|----------------------|
| **Questions** | 47 | 2,363+894 | 666+148 |
| **Avg Answers** | 107.6 | 1.4 | 50–300+ |
| **Max Answers** | 341 | 23 | 821 |
| **Difficulty** | Very Hard | Moderate | Extremely Hard |
| **Best For** | Comprehensive knowledge | Basic knowledge | Multi-hop reasoning |
| **Recall Challenge** | Very High | Low | Extreme |
| **Precision Challenge** | High | Moderate | Extreme |
| **Exact Match** | Very Rare (<5%) | Achievable (~65%) | Very Rare (<5%) |

---

## Use Cases

### Chain → Firms
✓ Comprehensive knowledge testing
✓ Large-scale recall evaluation
✓ RAG system benchmarking
✓ Hallucination detection
✓ Domain expertise assessment

### Firm → Chains
✓ Quick knowledge assessment
✓ Company classification
✓ Industry relationship mapping
✓ Multi-label classification benchmarks

### Competitor Analysis
✓ Multi-hop structural reasoning evaluation  
✓ Cross-chain aggregation capability testing  
✓ Graph-level domain understanding assessment 

---

## Data Sources

All data is sourced from:
- **Neo4j Graph Database**: Taiwan value chain graph
- **Nodes**: ValueChain, Category, Subcategory, SubSubcategory, Company
- **Relationships**: CONTAINS (hierarchy), INCLUDES (company membership)
- **Query Pattern**: Traverse 1-10 levels through hierarchy

---

## Example QA Pairs

### Chain → Firms (Harder)

**Medium chain:**
```json
{
  "question": "列出產業鏈 交通運輸及航運產業鏈 包含的所有公司。",
  "chain": "交通運輸及航運產業鏈",
  "answer": ["Angelicoussis Shipping Group", "中保科", "...64 total"],
  "answer_count": 64,
  "local_count": 39,
  "foreign_count": 25
}
```

**Large chain:**
```json
{
  "question": "列出產業鏈 人工智慧產業鏈 包含的所有公司。",
  "chain": "人工智慧產業鏈",
  "answer": ["91APP*-KY", "Anthropic", "Google", "...104 total"],
  "answer_count": 104,
  "local_count": 80,
  "foreign_count": 24
}
```

### Firm → Chains (Easier)

**Single-chain example:**
```json
{
  "question": "列出公司 A&D 所屬的所有產業鏈。",
  "company": "A&D",
  "answer": ["醫療器材產業鏈"],
  "answer_count": 1,
  "is_foreign": true
}
```

**Multi-chain example:**
```json
{
  "question": "列出公司 91APP*-KY 所屬的所有產業鏈。",
  "company": "91APP*-KY",
  "answer": [
    "人工智慧產業鏈",
    "大數據產業鏈",
    "金融科技產業鏈",
    "雲端運算產業鏈",
    "電子商務產業鏈"
  ],
  "answer_count": 5,
  "is_foreign": false
}
```

### Competitors Analysis (Hardest)

**Single-chain example:**
```json
{
  "question": "台積電在 半導體產業鏈，這家公司競爭對手有哪些公司？",
  "company": "台積電",
  "chains": ["半導體產業鏈"],
  "is_foreign": false,
  "answer": ["聯電", "三星電子", "中芯國際", "力積電", "... 25 in total"],
  "answer_count": 25
}
```

**Multi-chain example:**
```json
{
  "question": "鴻海在 通信網路產業鏈、連接器產業鏈、電動車輛產業產業鏈、電腦及週邊設備產業鏈，這家公司競爭對手有哪些公司？",
  "company": "鴻海",
  "chains": [
    "通信網路產業鏈",
    "連接器產業鏈",
    "電動車輛產業產業鏈",
    "電腦及週邊設備產業鏈"
  ],
  "is_foreign": false,
  "answer": [
    "廣達", "緯創", "和碩", "戴爾", "華碩", "... 359 in total"],
  "answer_count": 359
}
```

---

## Directory Structure

```
tivac_eval/
└── datasets/
    └── demo/
        └── qa/
            ├── Chain → Firms
            │   ├── chain_firms_qa.jsonl
            │   ├── chain_firms_qa.meta.json
            │   ├── chain_firms_qa_local.jsonl
            │   └── chain_firms_qa_local.meta.json
            │
            ├── Firm → Chains
            │   ├── firm_chains_qa_local.jsonl
            │   ├── firm_chains_qa_local.meta.json
            │   ├── firm_chains_qa_foreign.jsonl
            │   └── firm_chains_qa_foreign.meta.json
            │
            ├── Competitors Analysis
            │   ├── competitors_qa_local.jsonl
            │   ├── competitors_qa_local.meta.json
            │   ├── competitors_qa_foreign.jsonl
            │   └── competitors_qa_foreign.meta.json
```

---

## Cost Estimates (OpenAI GPT-4o-mini)

| Dataset | Questions | Est. Cost |
|---------|-----------|-----------|
| Test (5 samples) | 5 | $0.01 |
| firm_chains_qa_local | 2,309 | $0.50-1.00 |
| firm_chains_qa_foreign | 878 | $0.20-0.40 |
| firm_chains_qa | 3,187 | $0.70-1.40 |
| chain_firms_qa_large | 47 | $0.05-0.10 |

**Note**: Chain→Firms is cheaper (fewer questions) but harder to answer correctly.

---

## Performance Expectations

### Firm → Chains

| Performance | F1 Score | Exact Match | Interpretation |
|-------------|----------|-------------|----------------|
| Strong | >0.75 | >0.65 | Excellent domain knowledge |
| Good | 0.60-0.75 | 0.45-0.65 | Solid knowledge, minor gaps |
| Moderate | 0.40-0.60 | 0.25-0.45 | Partial knowledge |
| Weak | <0.40 | <0.25 | Limited knowledge |

### Chain → Firms

| Performance | F1 Score | Recall | Interpretation |
|-------------|----------|--------|----------------|
| Strong | >0.50 | >0.60 | Comprehensive knowledge |
| Good | 0.35-0.50 | 0.45-0.60 | Good coverage |
| Moderate | 0.20-0.35 | 0.30-0.45 | Knows major players |
| Weak | <0.20 | <0.30 | Very limited knowledge |

### Competitors Analysis

| Performance | F1 Score | Recall | Interpretation |
|-------------|----------|--------|----------------|
| Strong | >0.35 | >0.45 | Captures multi-chain aggregated competitor structures; demonstrates strong graph-level reasoning ability |
| Good | 0.25–0.35 | 0.35–0.45 | Identifies major competitor groups but has limited cross-chain overlap coverage |
| Moderate | 0.15–0.25 | 0.25–0.35 | Captures partial shared-chain competitors; frequently misses multi-chain intersections |
| Weak | <0.15 | <0.25 | Fails to perform effective cross-chain structural reasoning |

---

## Key Findings

From initial testing:

1. **Firm→Chains is more feasible** for current LLMs
   - Smaller answer sets (1-5 chains typically)
   - Higher exact match rates possible
   - Good for quick assessment

2. **Chain→Firms is very challenging** even for strong models
   - Large answer sets (20-300 companies)
   - Requires comprehensive knowledge
   - Better for RAG evaluation
   - Exact matches extremely rare

3. **Competitors Analysis is extremely challenging for current LLMs**
   - Requires multi-hop graph reasoning across multiple shared value chains  
   - Current LLMs have limited inference capabilities for multi-hop reasoning.

4. **Local vs Foreign Companies**
   - Models tend to perform better on local companies
   - Foreign companies have less coverage
   - Consider separate evaluation tracks

5. **Chain Size Matters**
   - Performance degrades with chain size
   - Small chains: Recall >0.7 possible
   - Large chains: Recall <0.3 typical

---

## Future Work

Potential extensions:

1. **Additional Tasks**
   - Category→Subcategory hierarchy
   - Company→Company relationships
   - Multi-hop reasoning queries (3-hop or 4-hop ...)

2. **Additional Metrics**
   - Semantic similarity (beyond exact match)
   - Coverage metrics
   - Hallucination rates

3. **Additional Models**
   - Fine-tuned models
   - RAG systems

4. **Temporal Analysis**
   - Track model improvements over time
   - Dataset versioning

---

## Citation

If you use these datasets in your research, please cite appropriately and acknowledge the Taiwan Economic Journal (TEJ) as the original data source.

---

## Contact & Support

For questions, issues, or contributions:
- Check the detailed README files for each dataset
- Review the evaluation documentation
- See example scripts and outputs

**Last Updated**: February 17, 2026
**Version**: 2.0
