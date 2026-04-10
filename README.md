# 🧠 AAOS — Academic AI Operating System

> **Assoc. Prof. Dr. Arjin Numsomran** | KMITL — Dept. of Instrumentation & Control Engineering  
> CEO, Envision I&C Engineering Groups

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph)
[![Anthropic](https://img.shields.io/badge/Claude-Sonnet_4-orange.svg)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19492175.svg)](https://doi.org/10.5281/zenodo.19492175)

---

## 📖 Overview

**AAOS (Academic AI Operating System)** is a self-improving multi-agent framework designed for academic professionals who manage multiple roles simultaneously — teaching, research, and industry consulting.

Unlike generic AI assistants, AAOS is:
- **Domain-specialized** — 4 agents covering Engineering, Research, Medical Education, and Business
- **Self-improving** — LangGraph loop evaluates and retries until quality score ≥ 7/10
- **Cost-effective** — ~$3–5/month vs commercial solutions ($100+)
- **Integrated** — outputs to Obsidian Vault, Notion, ChromaDB automatically

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────┐
    User Input ───▶ │     Orchestrator (L5.5)      │
                    │   Keyword-based routing      │
                    └────────────┬────────────────┘
                                 │ route()
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ A1          │   │ A2          │   │ A3          │   │ A4          │
    │ Engineering │   │ Researcher  │   │ Medical     │   │ Business    │
    │ 8 tasks     │   │ 6 tasks     │   │ 7 tasks     │   │ 7 tasks     │
    └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
              │                  │                  │                  │
              └──────────────────┴──────────────────┴──────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   ChromaDB Memory   │
                              │   Obsidian Vault    │
                              │   Notion Databases  │
                              └─────────────────────┘

    LangGraph L6 (Self-improving):
    Plan ──▶ Execute ──▶ Evaluate ──▶ Output
               ▲              │ score < 7
               └── Improve ◀──┘
```

---

## 🤖 4 Specialized Agents

| Agent | Domain | Task Types |
|---|---|---|
| **A1 Engineering** | LabVIEW, SIS/IEC 61511, Envision I&C | lecture, lab, exam, feedback, sil, hazop, rca, proposal |
| **A2 Researcher** | IEEE papers, IDE-IPA, Proposals | abstract, lit_review, ide_ipa, reviewer, proposal, summary |
| **A3 Medical** | Anatomy, Vajira MD Curriculum 2567 | anki, summary, quiz, case_study, vr_script, cpg, study_plan |
| **A4 Business** | ESG/SET50, CFA, Envision Strategy | esg, financial, cfa, strategy, pitch, market, report |

---

## 📊 AAOS Maturity Model (L1–L6)

| Level | Name | Description | Status |
|---|---|---|---|
| L1 | Manual | Everything done by hand | Baseline |
| L2 | Assisted | ChatGPT Q&A, copy-paste | Past |
| L3 | Structured | Prompt Library + Obsidian + Notion | ✅ |
| L4 | Automated | FastAPI Dispatcher + ChromaDB Pipeline | ✅ |
| L5.5 | Agentic | 4 Agents + Orchestrator (keyword routing) | ✅ |
| **L6** | **Self-improving** | **LangGraph: Plan→Execute→Evaluate→Improve** | ✅ |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install anthropic httpx langgraph langchain langchain-anthropic
```

### 2. Set API Key

```powershell
# Windows PowerShell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","sk-ant-xxxxx","User")
```

Or create `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### 3. Run Orchestrator (L5.5)

```bash
# Demo mode
python orchestrator.py

# Interactive mode
python orchestrator.py --interactive
🎯 AAOS > เขียน abstract IEEE Access เรื่อง Arduino SIS IEC 61511
🎯 AAOS > SIL verification Biogas project SIL 2
🎯 AAOS > anki brachial plexus 20 cards
🎯 AAOS > ESG screening PTT SET50
```

### 4. Run LangGraph (L6 — Self-improving)

```bash
pip install langgraph langchain langchain-anthropic

# Demo
python langgraph_orchestrator.py

# Interactive
python langgraph_orchestrator.py --interactive
🧠 AAOS-L6 > เขียน abstract IEEE Access เรื่อง BPCS-SIS separation
# → Planner selects A2 → abstract
# → Evaluator scores 6.5 → triggers Improver
# → Improver enhances prompt → retry
# → Evaluator scores 7.5 → accepts → Output
```

---

## 📁 Project Structure

```
AAOS-Academic-AI-OS/
├── agents/
│   ├── engineering_agent.py      # A1: LabVIEW, SIS, Envision
│   ├── researcher_agent.py       # A2: Papers, IDE-IPA, Proposals
│   ├── medical_agent.py          # A3: Anatomy, Vajira, Anki
│   ├── business_agent.py         # A4: ESG, Finance, Strategy
│   ├── orchestrator.py           # L5.5: Keyword-based routing
│   └── langgraph_orchestrator.py # L6: Self-improving loop
├── docs/
│   ├── architecture.md           # System design
│   ├── maturity_model.md         # L1-L6 explanation
│   └── api_reference.md          # Task types & payloads
├── examples/
│   ├── abstract_ieee_access.json # Sample A2 output
│   ├── sil_verification.json     # Sample A1 output
│   ├── anki_brachial_plexus.json # Sample A3 output
│   └── esg_ptt_set50.json        # Sample A4 output
├── .env.template                 # Environment variables template
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## ⚙️ Output Structure

Each agent saves two files automatically:

```
D:\arjin-vault\
├── 06-OUTPUT\AAOS\results\           # JSON (all agents)
└── Obs_Dr_Arjin\
    ├── 01-TEACHING\AAOS-notes\       # A1: Lecture, Lab, Exam
    ├── 02-RESEARCH\AAOS-notes\       # A2: Abstract, IDE-IPA
    ├── 03-ENVISION\AAOS-notes\       # A1+A4: SIL, Strategy
    ├── 04-KNOWLEDGE\medical-notes\   # A3: Anki, Quiz, Case
    └── 08-PROFESSIONAL\AAOS-notes\   # A4: ESG, Financial
```

---

## 💰 Cost Comparison

| Solution | Cost/month | Self-improving | Multi-agent | Domain-specific |
|---|---|---|---|---|
| **AAOS** | **~$5** | **✅** | **✅** | **✅** |
| Notion AI | $16 | ❌ | ❌ | ❌ |
| GitHub Copilot | $19 | ❌ | ❌ | ❌ |
| Custom GPT (OpenAI) | $20 | ❌ | Limited | ❌ |
| Enterprise AI Tools | $100+ | Some | Some | Configurable |

---

## 🎯 Use Cases

### Teaching (A1)
```python
result = await agent.run({
    "task_type": "lecture",
    "payload": {
        "course": "01068012 Virtual Instrumentation/LabVIEW",
        "topic": "DAQ Fundamentals",
        "week": 5,
        "students": 40
    }
})
```

### Research (A2)
```python
result = await agent.run({
    "task_type": "ide_ipa",
    "payload": {
        "project_name": "AI-Integrated Safety Systems",
        "agency": "บพข.",
        "budget": 3000000
    }
})
```

### Medical Education (A3)
```python
result = await agent.run({
    "task_type": "anki",
    "payload": {
        "subject": "Anatomy",
        "topic": "Brachial Plexus",
        "card_count": 20
    }
})
```

### Business (A4)
```python
result = await agent.run({
    "task_type": "esg",
    "payload": {
        "company": "PTT",
        "ticker": "PTT",
        "index": "SET50"
    }
})
```

---

## 📝 Citation

If you use AAOS in your research, please cite:

```bibtex
@software{numsomran2026aaos,
  author    = {Numsomran, Arjin},
  title     = {AAOS: Academic AI Operating System — 
               A Self-improving Multi-Agent Framework for Academic Professionals},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19492175},
  url       = {https://doi.org/10.5281/zenodo.19492175}
}
```

---

## 📬 Contact

**Assoc. Prof. Dr. Arjin Numsomran**  
Department of Instrumentation and Control Engineering  
School of Engineering, KMITL  
📧 arjin.nu@kmitl.ac.th  
🏢 EN-03 Building, KMITL, Bangkok 10520, Thailand

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using Claude Sonnet 4.6 | Anthropic*
