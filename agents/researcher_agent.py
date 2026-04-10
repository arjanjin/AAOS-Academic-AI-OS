"""
╔══════════════════════════════════════════════════════════════╗
║        AAOS — Agent A2: Researcher Agent v1.0               ║
║        รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE                  ║
║        เชื่อม: FastAPI Dispatcher → ChromaDB → Notion        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import uuid
import httpx
import asyncio
from datetime import datetime
from typing import Optional
import anthropic
from pathlib import Path

# ── Load .env (ANTHROPIC_API_KEY) ────────────────────────
def _load_env():
    env_paths = [
        Path(r"D:\arjin-vault\Obs_Dr_Arjin\.env"),
        Path(r"D:\arjin-vault\.env"),
        Path(__file__).parent / ".env",
        Path.home() / ".env",
    ]
    for p in env_paths:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            print(f"[ENV] ✅ Loaded: {p}")
            return
    print("[ENV] ⚠️  .env not found — using system ANTHROPIC_API_KEY")

_load_env()

# ─────────────────────────────────────────
#  CONFIG — แก้ค่าตรงนี้ให้ตรงกับเครื่อง
# ─────────────────────────────────────────
FASTAPI_URL     = "http://localhost:8000"          # FastAPI Dispatcher
CHROMA_URL      = "http://localhost:8001"          # ChromaDB (ถ้าใช้ HTTP API)

# ── Paths (ตรงกับโฟลเดอร์จริงบนเครื่อง) ─────────────────
#
#  D:\arjin-vault\
#  ├── 06-OUTPUT\AAOS\          ← AAOS pipeline (FastAPI, ChromaDB, results)
#  │   ├── chromadb\
#  │   ├── fastapi-dispatcher\
#  │   ├── results\              ← JSON output ที่นี่
#  │   └── agents\              ← script อยู่ที่นี่
#  └── Obs_Dr_Arjin\            ← Obsidian Vault
#      ├── 02-RESEARCH\
#      ├── 05-INBOX\
#      └── 04-KNOWLEDGE\
#
AAOS_ROOT       = r"D:\arjin-vault\06-OUTPUT\AAOS"
VAULT_ROOT      = r"D:\arjin-vault\Obs_Dr_Arjin"

OUTPUT_DIR      = rf"{AAOS_ROOT}\results"                      # JSON raw output (มีอยู่แล้ว)
NOTES_DIR       = rf"{VAULT_ROOT}\02-RESEARCH\AAOS-notes"     # .md → Obsidian
INBOX_DIR       = rf"{VAULT_ROOT}\05-INBOX\researcher"        # Draft inbox
KNOWLEDGE_DIR   = rf"{VAULT_ROOT}\04-KNOWLEDGE\researcher"    # Distilled knowledge

NOTION_PAPER_DB = "e3e48b24-02f0-47f6-9963-05a33fe91fa0"  # Research Papers DB

CLAUDE_MODEL    = "claude-sonnet-4-20250514"

# Trigger keywords สำหรับ Orchestrator route มาหา A2
TRIGGERS = ["paper", "abstract", "ide-ipa", "proposal", "literature",
            "reviewer", "research", "scite", "funding", "@research"]

# ─────────────────────────────────────────
#  TASK TYPES ที่ A2 รองรับ
# ─────────────────────────────────────────
TASK_TYPES = {
    "abstract"   : "เขียน Abstract สำหรับ Journal Paper",
    "lit_review" : "ทำ Literature Review + Research Gap",
    "ide_ipa"    : "วิเคราะห์ IDE-IPA Impact Pathway",
    "reviewer"   : "เขียน Response to Reviewer",
    "proposal"   : "เขียน Research Proposal",
    "summary"    : "สรุป Paper / เอกสารวิชาการ",
}


# ═══════════════════════════════════════════
#  CLASS: ResearcherAgent
# ═══════════════════════════════════════════
class ResearcherAgent:

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "\n❌ ANTHROPIC_API_KEY not found!\n"
                "   สร้างไฟล์ .env ใน vault root:\n"
                "   D:\\arjin-vault\\Obs_Dr_Arjin\\.env\n"
                "   แล้วเพิ่ม: ANTHROPIC_API_KEY=sk-ant-xxxxx"
            )
        self.client   = anthropic.Anthropic(api_key=api_key)
        self.agent_id = "A2-Researcher"
        self.version  = "1.0.0"
        for d in [OUTPUT_DIR, NOTES_DIR, INBOX_DIR, KNOWLEDGE_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"[{self.agent_id}] ✅ Initialized — AAOS Researcher Agent v{self.version}")
        print(f"[{self.agent_id}] 📁 Vault   : {VAULT_ROOT}")
        print(f"[{self.agent_id}] 📁 Output  : {OUTPUT_DIR}")
        print(f"[{self.agent_id}] 📁 Notes   : {NOTES_DIR}")

    # ─────────────────────────────────────
    #  MAIN: รับ task แล้ว route ไป method
    # ─────────────────────────────────────
    async def run(self, task: dict) -> dict:
        task_type = task.get("task_type", "summary").lower()
        payload   = task.get("payload", {})
        task_id   = task.get("task_id", str(uuid.uuid4())[:8])

        print(f"\n[{self.agent_id}] 🔄 Task #{task_id} | Type: {task_type}")
        print(f"[{self.agent_id}] 📥 Payload keys: {list(payload.keys())}")

        # Route ไป method ที่เหมาะสม
        handlers = {
            "abstract"   : self.write_abstract,
            "lit_review" : self.literature_review,
            "ide_ipa"    : self.ide_ipa_analysis,
            "reviewer"   : self.reviewer_response,
            "proposal"   : self.write_proposal,
            "summary"    : self.summarize_paper,
        }

        handler = handlers.get(task_type, self.summarize_paper)
        result  = await handler(payload, task_id)

        # Save output
        await self._save_output(task_id, task_type, result)
        return result

    # ─────────────────────────────────────
    #  T1: เขียน Abstract
    # ─────────────────────────────────────
    async def write_abstract(self, payload: dict, task_id: str) -> dict:
        title    = payload.get("title", "")
        journal  = payload.get("journal", "IEEE Access")
        problem  = payload.get("problem", "")
        method   = payload.get("method", "")
        result   = payload.get("result", "")
        contrib  = payload.get("contribution", "")
        keywords = payload.get("keywords", [])

        prompt = f"""You are an expert academic writer for engineering and education research.

Write a professional Abstract for the following paper:

Title: {title}
Target Journal: {journal}
Problem: {problem}
Method: {method}
Result: {result}
Contribution: {contrib}
Keywords: {', '.join(keywords)}

Requirements:
- Length: 250-300 words
- Structure: Background → Problem → Method → Result → Contribution
- Style: Match {journal} guidelines
- End with 5 keywords

Output JSON:
{{
  "abstract": "...",
  "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
  "word_count": 0
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw)
            data = json.loads(json_match.group()) if json_match else json.loads(raw.strip())
        except Exception:
            data = {"abstract": raw, "keywords": keywords, "word_count": len(raw.split())}

        return {
            "task_id"  : task_id,
            "task_type": "abstract",
            "title"    : title,
            "journal"  : journal,
            "output"   : data,
            "timestamp": datetime.now().isoformat()
        }

    # ─────────────────────────────────────
    #  T2: Literature Review
    # ─────────────────────────────────────
    async def literature_review(self, payload: dict, task_id: str) -> dict:
        topic    = payload.get("topic", "")
        clusters = payload.get("keyword_clusters", [])
        years    = payload.get("year_range", "2020-2025")

        prompt = f"""You are a systematic review expert in engineering and education.

Perform a Literature Review analysis for:
Topic: {topic}
Year Range: {years}
Keyword Clusters: {json.dumps(clusters, ensure_ascii=False)}

Generate:
1. Research Gap Matrix (3-5 gaps)
2. Theme Classification (5 themes max)
3. 10 Scite.ai search queries (Boolean format)
4. Suggested citation structure for Introduction section
5. Research position statement

Output JSON:
{{
  "gaps": [
    {{"gap": "...", "evidence": "...", "opportunity": "..."}}
  ],
  "themes": [
    {{"theme": "...", "papers_count": 0, "description": "..."}}
  ],
  "scite_queries": ["query1", "query2"],
  "citation_structure": "...",
  "position_statement": "..."
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text
        try:
            data = json.loads(raw.strip().replace("```json","").replace("```",""))
        except Exception:
            data = {"raw_output": raw}

        return {
            "task_id"  : task_id,
            "task_type": "lit_review",
            "topic"    : topic,
            "output"   : data,
            "timestamp": datetime.now().isoformat()
        }

    # ─────────────────────────────────────
    #  T3: IDE-IPA Analysis
    # ─────────────────────────────────────
    async def ide_ipa_analysis(self, payload: dict, task_id: str) -> dict:
        project_name = payload.get("project_name", "")
        agency       = payload.get("agency", "สกสว.")
        budget       = payload.get("budget", 0)
        duration     = payload.get("duration", "2 ปี")
        objectives   = payload.get("objectives", "")
        methodology  = payload.get("methodology", "")

        prompt = f"""คุณเป็นผู้เชี่ยวชาญ IDE-IPA Framework ของ สกสว./บพข.

วิเคราะห์โครงการวิจัยนี้ตาม IDE-IPA Framework:

ชื่อโครงการ: {project_name}
แหล่งทุน: {agency}
งบประมาณ: {budget:,} บาท
ระยะเวลา: {duration}
วัตถุประสงค์: {objectives}
วิธีการ: {methodology}

วิเคราะห์ครบ:
1. 7 Impact Dimensions
2. 6 Impact Pathways
3. IPA Logic Model (Input→Activity→Output→Outcome→Impact)
4. Plausibility Score (0-100 ต่อ dimension)
5. Gap Analysis และข้อเสนอแนะ

Output JSON:
{{
  "dimensions": [
    {{"name": "...", "score": 0, "description": "...", "evidence": "..."}}
  ],
  "pathways": [
    {{"pathway": "...", "strength": "high/medium/low", "rationale": "..."}}
  ],
  "logic_model": {{
    "input": [...],
    "activity": [...],
    "output": [...],
    "outcome": [...],
    "impact": [...]
  }},
  "overall_score": 0,
  "gaps": [...],
  "recommendations": [...]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text
        try:
            # Extract JSON even if wrapped in markdown code blocks
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw.strip())
        except Exception:
            data = {"raw_output": raw}

        return {
            "task_id"     : task_id,
            "task_type"   : "ide_ipa",
            "project_name": project_name,
            "agency"      : agency,
            "output"      : data,
            "timestamp"   : datetime.now().isoformat()
        }

    # ─────────────────────────────────────
    #  T4: Response to Reviewer
    # ─────────────────────────────────────
    async def reviewer_response(self, payload: dict, task_id: str) -> dict:
        journal   = payload.get("journal", "")
        decision  = payload.get("decision", "Major Revision")
        comments  = payload.get("reviewer_comments", [])

        prompt = f"""You are an expert academic writer handling peer review responses.

Journal: {journal}
Decision: {decision}
Reviewer Comments:
{json.dumps(comments, ensure_ascii=False, indent=2)}

Write a professional point-by-point Response to Reviewers letter.

Requirements:
- Formal academic tone
- Address EVERY comment specifically
- For changes: explain what was revised and where (page/line)
- For disagreements: provide evidence-based argument
- Start with a brief thank-you paragraph

Output JSON:
{{
  "cover_letter": "...",
  "responses": [
    {{
      "comment_number": 1,
      "reviewer_comment": "...",
      "response": "...",
      "action_taken": "Revised page X, paragraph Y"
    }}
  ],
  "summary_of_changes": "..."
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text
        try:
            data = json.loads(raw.strip().replace("```json","").replace("```",""))
        except Exception:
            data = {"raw_output": raw}

        return {
            "task_id"  : task_id,
            "task_type": "reviewer",
            "journal"  : journal,
            "output"   : data,
            "timestamp": datetime.now().isoformat()
        }

    # ─────────────────────────────────────
    #  T5: Research Proposal
    # ─────────────────────────────────────
    async def write_proposal(self, payload: dict, task_id: str) -> dict:
        title     = payload.get("title", "")
        agency    = payload.get("agency", "บพข.")
        team      = payload.get("team", [])
        problem   = payload.get("problem", "")
        budget    = payload.get("budget", 0)
        duration  = payload.get("duration", "2 ปี")

        prompt = f"""คุณเป็นผู้เชี่ยวชาญการเขียน Research Proposal สำหรับแหล่งทุนไทย

เขียน Research Proposal สำหรับ:
ชื่อโครงการ: {title}
แหล่งทุน: {agency}
ทีมวิจัย: {', '.join(team)}
ปัญหา/โจทย์วิจัย: {problem}
งบประมาณ: {budget:,} บาท
ระยะเวลา: {duration}

เขียนให้ครบทุกหัวข้อตามรูปแบบ {agency}:

Output JSON:
{{
  "title_th": "...",
  "title_en": "...",
  "rationale": "...",
  "objectives": ["obj1", "obj2", "obj3"],
  "scope": "...",
  "methodology": {{
    "phase1": "...",
    "phase2": "...",
    "phase3": "..."
  }},
  "expected_outputs": [...],
  "expected_outcomes": [...],
  "expected_impacts": [...],
  "budget_breakdown": {{
    "personnel": 0,
    "equipment": 0,
    "operations": 0,
    "total": 0
  }},
  "kpis": [...]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text
        try:
            data = json.loads(raw.strip().replace("```json","").replace("```",""))
        except Exception:
            data = {"raw_output": raw}

        return {
            "task_id"  : task_id,
            "task_type": "proposal",
            "title"    : title,
            "agency"   : agency,
            "output"   : data,
            "timestamp": datetime.now().isoformat()
        }

    # ─────────────────────────────────────
    #  T6: Summarize Paper
    # ─────────────────────────────────────
    async def summarize_paper(self, payload: dict, task_id: str) -> dict:
        text  = payload.get("text", "")
        focus = payload.get("focus", "general")

        prompt = f"""Summarize the following academic content concisely.
Focus: {focus}

Content:
{text[:3000]}

Output JSON:
{{
  "summary": "...",
  "key_findings": ["...", "...", "..."],
  "methodology": "...",
  "limitations": "...",
  "future_work": "...",
  "relevance_to_aaos": "..."
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text
        try:
            data = json.loads(raw.strip().replace("```json","").replace("```",""))
        except Exception:
            data = {"raw_output": raw}

        return {
            "task_id"  : task_id,
            "task_type": "summary",
            "output"   : data,
            "timestamp": datetime.now().isoformat()
        }

    # ─────────────────────────────────────
    #  SAVE: บันทึก output ลงไฟล์
    # ─────────────────────────────────────
    async def _save_output(self, task_id: str, task_type: str, result: dict):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{task_type}_{task_id}"

        # 1) Save JSON
        json_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[{self.agent_id}] 💾 JSON  → {json_path}")

        # 2) Save Obsidian Markdown Note
        await self._save_obsidian_note(filename, task_type, result, result.get("task_id", "unknown"))

    async def _save_obsidian_note(self, filename: str, task_type: str, result: dict, task_id: str = "unknown"):
        """สร้าง Obsidian Markdown note พร้อม YAML frontmatter"""
        ts_readable = datetime.now().strftime("%Y-%m-%d %H:%M")
        output      = result.get("output", {})

        # ── YAML Frontmatter ──
        frontmatter = f"""---
title: "{task_type.upper()} — {result.get('title', result.get('project_name', task_id))}"
date: {ts_readable}
tags: [AAOS, researcher-agent, {task_type}]
agent: A2-Researcher
task_id: {result.get('task_id', '')}
journal: "{result.get('journal', '')}"
status: draft
---
"""

        # ── Body ──
        body_lines = [
            f"# {task_type.upper()} Output\n",
            f"> Generated by **AAOS A2 Researcher Agent** | {ts_readable}\n",
            "---\n",
        ]

        if task_type == "abstract":
            body_lines += [
                "## Abstract\n",
                output.get("abstract", ""),
                f"\n\n**Word Count:** {output.get('word_count', 0)}\n",
                "\n## Keywords\n",
                ", ".join(output.get("keywords", [])),
            ]

        elif task_type == "lit_review":
            body_lines += ["## Research Gaps\n"]
            for g in output.get("gaps", []):
                body_lines.append(f"- **{g.get('gap','')}** — {g.get('opportunity','')}")
            body_lines += ["\n## Scite.ai Queries\n"]
            for q in output.get("scite_queries", []):
                body_lines.append(f"- `{q}`")
            body_lines += [f"\n## Position Statement\n{output.get('position_statement','')}"]

        elif task_type == "ide_ipa":
            body_lines += [
                f"## Overall Score: {output.get('overall_score', 'N/A')}/100\n",
                "## Impact Dimensions\n",
            ]
            for d in output.get("dimensions", []):
                body_lines.append(f"- **{d.get('name','')}** ({d.get('score',0)}/100): {d.get('description','')}")
            body_lines += ["\n## Recommendations\n"]
            for r in output.get("recommendations", []):
                body_lines.append(f"- {r}")

        elif task_type == "reviewer":
            body_lines += [
                "## Cover Letter\n",
                output.get("cover_letter", ""),
                "\n## Point-by-Point Responses\n",
            ]
            for resp in output.get("responses", []):
                body_lines.append(f"\n### Comment #{resp.get('comment_number','')}")
                body_lines.append(f"**Reviewer:** {resp.get('reviewer_comment','')}")
                body_lines.append(f"**Response:** {resp.get('response','')}")
                body_lines.append(f"**Action:** {resp.get('action_taken','')}")

        elif task_type == "proposal":
            body_lines += [
                f"## {output.get('title_th','')}\n",
                f"*{output.get('title_en','')}*\n",
                "\n## Rationale\n", output.get("rationale", ""),
                "\n## Objectives\n",
            ]
            for obj in output.get("objectives", []):
                body_lines.append(f"- {obj}")
            body_lines += [f"\n## Expected Impact\n"]
            for imp in output.get("expected_impacts", []):
                body_lines.append(f"- {imp}")

        else:  # summary
            body_lines += [
                "## Summary\n", output.get("summary", ""),
                "\n## Key Findings\n",
            ]
            for kf in output.get("key_findings", []):
                body_lines.append(f"- {kf}")

        body_lines += ["\n\n---\n*AAOS A2 Researcher Agent — Auto-generated*"]

        md_content = frontmatter + "\n".join(str(l) for l in body_lines)
        md_path    = os.path.join(NOTES_DIR, f"{filename}.md")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[{self.agent_id}] 📝 Note  → {md_path}")

    # ─────────────────────────────────────
    #  REGISTER กับ FastAPI Dispatcher
    # ─────────────────────────────────────
    async def register_with_dispatcher(self):
        payload = {
            "agent_id"   : self.agent_id,
            "version"    : self.version,
            "task_types" : list(TASK_TYPES.keys()),
            "triggers"   : TRIGGERS,
            "endpoint"   : "http://localhost:8002/run",  # A2 endpoint
            "status"     : "active"
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{FASTAPI_URL}/agents/register",
                    json=payload,
                    timeout=5.0
                )
            print(f"[{self.agent_id}] ✅ Registered with Dispatcher: {resp.status_code}")
        except Exception as e:
            print(f"[{self.agent_id}] ⚠️  Dispatcher not available: {e}")
            print(f"[{self.agent_id}] 🔄 Running in standalone mode")


# ═══════════════════════════════════════════
#  DEMO: ทดสอบ A2 Agent
# ═══════════════════════════════════════════
async def demo():
    agent = ResearcherAgent()
    await agent.register_with_dispatcher()

    print("\n" + "═"*60)
    print("  DEMO — A2 Researcher Agent")
    print("═"*60)

    # ── Demo 1: เขียน Abstract สำหรับ IEEE Access Paper ──
    print("\n[DEMO 1] Abstract Generator — IEEE Access Paper")
    task_abstract = {
        "task_id"  : "demo-001",
        "task_type": "abstract",
        "payload"  : {
            "title"       : "Dual Arduino UNO R4 WiFi BPCS-SIS Separation for Process Safety Education",
            "journal"     : "IEEE Access",
            "problem"     : "Engineering students lack hands-on SIS education due to expensive equipment",
            "method"      : "Dual Arduino UNO R4 WiFi boards separating BPCS and SIS functions per IEC 61511",
            "result"      : "Students demonstrated 40% improvement in SIS design competency",
            "contribution": "Low-cost educational platform for IEC 61511 SIS concepts",
            "keywords"    : ["Safety Instrumented System", "IEC 61511", "Arduino", "BPCS", "Engineering Education"]
        }
    }
    result1 = await agent.run(task_abstract)
    print(f"\n  Abstract ({result1['output'].get('word_count',0)} words):")
    print(f"  {result1['output'].get('abstract','')[:200]}...")

    # ── Demo 2: IDE-IPA Analysis ──
    print("\n[DEMO 2] IDE-IPA Analysis — IDE-IPA Framework Paper")
    task_ideipa = {
        "task_id"  : "demo-002",
        "task_type": "ide_ipa",
        "payload"  : {
            "project_name": "IDE-IPA Framework for Research Impact Assessment in Thai Higher Education",
            "agency"      : "บพข.",
            "budget"      : 3000000,
            "duration"    : "2 ปี",
            "objectives"  : "พัฒนา Framework การประเมินผลกระทบงานวิจัย IDE-IPA สำหรับมหาวิทยาลัยไทย",
            "methodology" : "Mixed-method research, case study, survey, framework validation"
        }
    }
    result2 = await agent.run(task_ideipa)
    score = result2['output'].get('overall_score', 'N/A')
    print(f"\n  IDE-IPA Overall Score: {score}/100")
    gaps = result2['output'].get('gaps', [])
    print(f"  Gaps identified: {len(gaps)}")

    print("\n" + "═"*60)
    print("  ✅ A2 Researcher Agent — Demo Complete")
    print(f"  📁 Results saved to: {OUTPUT_DIR}")
    print("═"*60)


# ═══════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AAOS — Agent A2: Researcher Agent v1.0                  ║
║     รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE Department          ║
╚══════════════════════════════════════════════════════════════╝

Usage:
  python researcher_agent.py          → Run demo
  
API Usage:
  agent = ResearcherAgent()
  result = await agent.run({
      "task_type": "abstract|lit_review|ide_ipa|reviewer|proposal|summary",
      "payload"  : { ... }
  })
""")
    asyncio.run(demo())
