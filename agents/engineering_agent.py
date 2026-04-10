"""
╔══════════════════════════════════════════════════════════════╗
║        AAOS — Agent A1: Engineering Agent v1.0              ║
║        รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE                  ║
║        Domain: LabVIEW, Control, SIS/IEC61511, Envision      ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import uuid
import httpx
import asyncio
import re
from datetime import datetime
from pathlib import Path
import anthropic

# ── Load .env ─────────────────────────────────────────────────
def _load_env():
    env_paths = [
        Path(r"D:\arjin-vault\.env"),
        Path(r"D:\arjin-vault\Obs_Dr_Arjin\.env"),
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
#  CONFIG
# ─────────────────────────────────────────
FASTAPI_URL   = "http://localhost:8000"

AAOS_ROOT     = r"D:\arjin-vault\06-OUTPUT\AAOS"
VAULT_ROOT    = r"D:\arjin-vault\Obs_Dr_Arjin"

OUTPUT_DIR    = rf"{AAOS_ROOT}\results"
NOTES_DIR     = rf"{VAULT_ROOT}\01-TEACHING\AAOS-notes"   # Teaching notes → 01-TEACHING
ENVISION_DIR  = rf"{VAULT_ROOT}\03-ENVISION\AAOS-notes"   # Envision notes → 03-ENVISION
INBOX_DIR     = rf"{VAULT_ROOT}\05-INBOX\engineering"

CLAUDE_MODEL  = "claude-sonnet-4-20250514"

TRIGGERS = ["labview", "control", "sil", "iec 61511", "iec 61508",
            "bpcs", "sis", "arduino", "envision", "hazop", "lopa",
            "pid", "bode", "root locus", "fopdt", "lecture", "@engineering"]

TASK_TYPES = {
    "lecture"   : "สร้าง Lecture Outline + Slide structure",
    "lab"       : "สร้าง Lab Manual (LabVIEW / Control)",
    "exam"      : "สร้างข้อสอบ + เฉลย",
    "feedback"  : "เขียน Feedback นักศึกษา",
    "sil"       : "SIL Verification Summary (IEC 61511)",
    "hazop"     : "HAZOP/LOPA Analysis Summary",
    "proposal"  : "Technical Proposal (Envision)",
    "rca"       : "Root Cause Analysis (RCA) Report",
}


# ═══════════════════════════════════════════
#  CLASS: EngineeringAgent
# ═══════════════════════════════════════════
class EngineeringAgent:

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "\n❌ ANTHROPIC_API_KEY not found!\n"
                "   Set: [System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY','sk-ant-xxx','User')"
            )
        self.client   = anthropic.Anthropic(api_key=api_key)
        self.agent_id = "A1-Engineering"
        self.version  = "1.0.0"
        for d in [OUTPUT_DIR, NOTES_DIR, ENVISION_DIR, INBOX_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"[{self.agent_id}] ✅ Initialized — AAOS Engineering Agent v{self.version}")
        print(f"[{self.agent_id}] 📁 Vault    : {VAULT_ROOT}")
        print(f"[{self.agent_id}] 📁 Output   : {OUTPUT_DIR}")
        print(f"[{self.agent_id}] 📁 Teaching : {NOTES_DIR}")
        print(f"[{self.agent_id}] 📁 Envision : {ENVISION_DIR}")

    # ─────────────────────────────────────
    #  MAIN: route task
    # ─────────────────────────────────────
    async def run(self, task: dict) -> dict:
        task_type = task.get("task_type", "lecture").lower()
        payload   = task.get("payload", {})
        task_id   = task.get("task_id", str(uuid.uuid4())[:8])

        print(f"\n[{self.agent_id}] 🔄 Task #{task_id} | Type: {task_type}")
        print(f"[{self.agent_id}] 📥 Payload keys: {list(payload.keys())}")

        handlers = {
            "lecture" : self.create_lecture,
            "lab"     : self.create_lab_manual,
            "exam"    : self.create_exam,
            "feedback": self.write_feedback,
            "sil"     : self.sil_verification,
            "hazop"   : self.hazop_lopa,
            "proposal": self.technical_proposal,
            "rca"     : self.root_cause_analysis,
        }

        handler = handlers.get(task_type, self.create_lecture)
        result  = await handler(payload, task_id)
        await self._save_output(task_id, task_type, result)
        return result

    # ─────────────────────────────────────
    #  T1: Lecture Outline
    # ─────────────────────────────────────
    async def create_lecture(self, payload: dict, task_id: str) -> dict:
        course  = payload.get("course", "")
        topic   = payload.get("topic", "")
        week    = payload.get("week", 1)
        level   = payload.get("level", "ปริญญาตรี")
        students = payload.get("students", 40)
        duration = payload.get("duration", 90)

        prompt = f"""You are Assoc. Prof. Dr. Arjin Numsomran, expert in Instrumentation & Control Engineering at KMITL.

Create a detailed Lecture Outline for:
Course: {course}
Topic: {topic}
Week: {week}/16
Level: {level} ({students} students)
Duration: {duration} minutes

Structure the lecture with:
1. Learning Objectives (3-5 objectives, Bloom's Taxonomy)
2. Lecture Flow:
   - Introduction/Review (10 min)
   - Core Content Part 1 (25 min)
   - Core Content Part 2 (25 min)
   - Application/Demo (20 min)
   - Q&A + Summary (10 min)
3. Key Concepts (5-7 concepts with brief explanation)
4. Industry Examples (link to Envision I&C projects: Solar/Biogas/HIPPS/CEMS)
5. In-class Activity
6. Assessment (Quiz/Assignment)
7. References (IEEE/ISA standards if applicable)

Output JSON:
{{
  "course": "{course}",
  "topic": "{topic}",
  "week": {week},
  "objectives": ["obj1", "obj2", "obj3"],
  "lecture_flow": [
    {{"segment": "Introduction", "duration": 10, "content": "..."}}
  ],
  "key_concepts": [
    {{"concept": "...", "explanation": "...", "standard": "..."}}
  ],
  "industry_example": "...",
  "activity": "...",
  "assessment": "...",
  "references": ["ref1", "ref2"]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "lecture",
                           {"course": course, "topic": topic, "week": week})

    # ─────────────────────────────────────
    #  T2: Lab Manual
    # ─────────────────────────────────────
    async def create_lab_manual(self, payload: dict, task_id: str) -> dict:
        course   = payload.get("course", "LabVIEW")
        lab_no   = payload.get("lab_no", 1)
        topic    = payload.get("topic", "")
        duration = payload.get("duration", 3)
        equipment = payload.get("equipment", [])
        language = payload.get("language", "Thai")

        prompt = f"""You are Assoc. Prof. Dr. Arjin Numsomran, LabVIEW/Control Systems expert at KMITL ICE.

Create a complete Lab Manual in {language}:
Course: {course}
Lab No: {lab_no}
Topic: {topic}
Duration: {duration} hours
Equipment: {', '.join(equipment) if equipment else 'LabVIEW, DAQ, PC'}

Include all sections:
1. Objective (3-5 items)
2. Background Theory (concise, with equations if needed)
3. Equipment & Software List
4. Pre-lab Preparation
5. Procedure (numbered, step-by-step)
6. Expected Results / Screenshots
7. Discussion Questions (5 questions)
8. Report Template / Rubric

Output JSON:
{{
  "lab_no": {lab_no},
  "topic": "{topic}",
  "course": "{course}",
  "language": "{language}",
  "objective": ["..."],
  "theory": "...",
  "equipment": ["..."],
  "prelab": "...",
  "procedure": [
    {{"step": 1, "action": "...", "expected": "..."}}
  ],
  "discussion_questions": ["q1", "q2", "q3", "q4", "q5"],
  "report_rubric": {{
    "intro": 10,
    "procedure": 20,
    "results": 40,
    "discussion": 20,
    "conclusion": 10
  }}
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "lab",
                           {"course": course, "lab_no": lab_no, "topic": topic})

    # ─────────────────────────────────────
    #  T3: Exam Generator
    # ─────────────────────────────────────
    async def create_exam(self, payload: dict, task_id: str) -> dict:
        course   = payload.get("course", "")
        topic    = payload.get("topic", "")
        exam_type = payload.get("exam_type", "Midterm")
        q_types  = payload.get("question_types", ["MCQ", "Short Answer", "Problem"])
        total    = payload.get("total_score", 100)
        language = payload.get("language", "Thai")

        prompt = f"""You are Assoc. Prof. Dr. Arjin Numsomran, expert exam writer for I&C Engineering.

Create a {exam_type} Exam in {language}:
Course: {course}
Topic: {topic}
Question Types: {', '.join(q_types)}
Total Score: {total}

Create questions covering Bloom's levels: Remember, Understand, Apply, Analyze.
Include industry-relevant scenarios (process control, safety systems).

Output JSON:
{{
  "exam_type": "{exam_type}",
  "course": "{course}",
  "topic": "{topic}",
  "total_score": {total},
  "questions": [
    {{
      "no": 1,
      "type": "MCQ",
      "bloom_level": "Remember",
      "question": "...",
      "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "B",
      "explanation": "...",
      "score": 5
    }}
  ],
  "answer_key": {{}},
  "rubric": {{}}
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "exam",
                           {"course": course, "topic": topic, "exam_type": exam_type})

    # ─────────────────────────────────────
    #  T4: Student Feedback
    # ─────────────────────────────────────
    async def write_feedback(self, payload: dict, task_id: str) -> dict:
        student  = payload.get("student_name", "")
        work     = payload.get("work_title", "")
        score    = payload.get("score", 0)
        max_score = payload.get("max_score", 100)
        strengths = payload.get("strengths", "")
        improvements = payload.get("improvements", "")
        language = payload.get("language", "Thai")

        prompt = f"""You are Assoc. Prof. Dr. Arjin Numsomran, providing constructive academic feedback.

Write encouraging and constructive feedback in {language} for:
Student: {student}
Work: {work}
Score: {score}/{max_score} ({score/max_score*100:.1f}%)
Strengths observed: {strengths}
Areas to improve: {improvements}

Requirements:
- Start with positive recognition
- Be specific about strengths with examples
- Give actionable improvement suggestions
- Connect to real engineering practice
- End with encouragement
- Tone: professional, supportive, growth-oriented

Output JSON:
{{
  "student": "{student}",
  "work": "{work}",
  "score": "{score}/{max_score}",
  "grade_level": "...",
  "feedback_th": "...",
  "feedback_en": "...",
  "key_strengths": ["..."],
  "action_items": ["..."],
  "next_steps": "..."
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "feedback",
                           {"student": student, "work": work, "score": score})

    # ─────────────────────────────────────
    #  T5: SIL Verification (IEC 61511)
    # ─────────────────────────────────────
    async def sil_verification(self, payload: dict, task_id: str) -> dict:
        project   = payload.get("project", "")
        proj_type = payload.get("project_type", "Solar")
        sif_desc  = payload.get("sif_description", "")
        sil_target = payload.get("sil_target", 2)
        pfd_value = payload.get("pfd_value", None)
        pfh_value = payload.get("pfh_value", None)
        demand_mode = payload.get("demand_mode", "Low Demand")

        prompt = f"""You are a Functional Safety Engineer expert in IEC 61511 / IEC 61508.

Perform SIL Verification for:
Project: {project} ({proj_type})
SIF Description: {sif_desc}
SIL Target: SIL {sil_target}
Demand Mode: {demand_mode}
PFD Value: {pfd_value if pfd_value else 'To be calculated'}
PFH Value: {pfh_value if pfh_value else 'To be calculated'}

Provide complete SIL Verification Report per IEC 61511:

Output JSON:
{{
  "project": "{project}",
  "sif_description": "{sif_desc}",
  "sil_target": {sil_target},
  "demand_mode": "{demand_mode}",
  "sil_ranges": {{
    "SIL1": "PFD 0.1-0.01",
    "SIL2": "PFD 0.01-0.001",
    "SIL3": "PFD 0.001-0.0001"
  }},
  "pfd_calculation": {{
    "formula": "...",
    "result": "...",
    "unit": "per demand"
  }},
  "verification_result": "Pass/Fail",
  "sil_achieved": "SIL X",
  "safety_function": "...",
  "safe_state": "...",
  "response_time": "...",
  "architecture": "...",
  "recommendations": ["..."],
  "applicable_standards": ["IEC 61511-1:2016", "IEC 61508-1:2010"]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "sil",
                           {"project": project, "sil_target": sil_target})

    # ─────────────────────────────────────
    #  T6: HAZOP/LOPA
    # ─────────────────────────────────────
    async def hazop_lopa(self, payload: dict, task_id: str) -> dict:
        project    = payload.get("project", "")
        node       = payload.get("node", "")
        deviation  = payload.get("deviation", "")
        consequence = payload.get("consequence", "")
        safeguards = payload.get("existing_safeguards", [])

        prompt = f"""You are a Process Safety expert in HAZOP and LOPA methodology.

Perform HAZOP/LOPA Analysis for:
Project: {project}
Node: {node}
Deviation: {deviation}
Consequence: {consequence}
Existing Safeguards: {', '.join(safeguards) if safeguards else 'None listed'}

Output JSON:
{{
  "project": "{project}",
  "node": "{node}",
  "deviation": "{deviation}",
  "consequence": "{consequence}",
  "severity": "Catastrophic/Critical/Marginal/Negligible",
  "causes": [
    {{"cause": "...", "likelihood": "..."}}
  ],
  "existing_safeguards": [
    {{"safeguard": "...", "ipl": true, "pfd": "..."}}
  ],
  "mitigated_risk": "...",
  "required_sil": "SIL X",
  "lopa_result": {{
    "initiating_event_frequency": "...",
    "ipl_credit": "...",
    "mitigated_frequency": "...",
    "risk_tolerance": "...",
    "gap": "..."
  }},
  "recommendations": ["..."],
  "required_actions": ["..."]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "hazop",
                           {"project": project, "node": node, "deviation": deviation})

    # ─────────────────────────────────────
    #  T7: Technical Proposal (Envision)
    # ─────────────────────────────────────
    async def technical_proposal(self, payload: dict, task_id: str) -> dict:
        client_name = payload.get("client", "")
        project     = payload.get("project", "")
        scope       = payload.get("scope", "")
        standards   = payload.get("standards", ["IEC 61511"])
        budget_range = payload.get("budget_range", "")
        language    = payload.get("language", "English")

        prompt = f"""You are CEO of Envision I&C Engineering Groups, expert in Safety Instrumented Systems.

Write a professional Technical Proposal in {language}:
Client: {client_name}
Project: {project}
Scope of Work: {scope}
Applicable Standards: {', '.join(standards)}
Budget Range: {budget_range}

Include all standard proposal sections:

Output JSON:
{{
  "client": "{client_name}",
  "project": "{project}",
  "proposal_no": "ENV-{datetime.now().strftime('%Y%m')}-001",
  "date": "{datetime.now().strftime('%Y-%m-%d')}",
  "executive_summary": "...",
  "technical_approach": {{
    "methodology": "...",
    "phases": [
      {{"phase": 1, "name": "...", "duration": "...", "deliverables": ["..."]}}
    ]
  }},
  "scope_of_work": ["..."],
  "deliverables": ["..."],
  "project_schedule": {{
    "total_duration": "...",
    "milestones": ["..."]
  }},
  "team_qualifications": {{
    "lead": "รศ.ดร.อาจินต์ น่วมสำราญ — CEO, Functional Safety Expert",
    "certifications": ["IEC 61511", "IEC 61508", "TÜV"]
  }},
  "applicable_standards": {json.dumps(standards)},
  "commercial_terms": {{
    "validity": "30 days",
    "payment_terms": "30/40/30"
  }}
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "proposal",
                           {"client": client_name, "project": project})

    # ─────────────────────────────────────
    #  T8: Root Cause Analysis
    # ─────────────────────────────────────
    async def root_cause_analysis(self, payload: dict, task_id: str) -> dict:
        project    = payload.get("project", "")
        incident   = payload.get("incident_description", "")
        date       = payload.get("incident_date", datetime.now().strftime("%Y-%m-%d"))
        system     = payload.get("system", "SIS")
        data       = payload.get("available_data", "")

        prompt = f"""You are a Process Safety and Reliability expert performing Root Cause Analysis.

Perform RCA for:
Project: {project}
System: {system}
Incident Date: {date}
Incident: {incident}
Available Data/Logs: {data}

Use structured RCA methodology (5-Why + Fishbone):

Output JSON:
{{
  "project": "{project}",
  "system": "{system}",
  "incident_date": "{date}",
  "incident_summary": "...",
  "timeline": [
    {{"time": "...", "event": "...", "source": "..."}}
  ],
  "five_why": [
    {{"level": 1, "why": "...", "answer": "..."}}
  ],
  "fishbone": {{
    "Man": ["..."],
    "Machine": ["..."],
    "Method": ["..."],
    "Material": ["..."],
    "Measurement": ["..."],
    "Environment": ["..."]
  }},
  "root_causes": ["..."],
  "contributing_factors": ["..."],
  "corrective_actions": [
    {{"action": "...", "owner": "...", "due_date": "...", "priority": "High/Medium/Low"}}
  ],
  "preventive_actions": ["..."],
  "lessons_learned": ["..."]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "rca",
                           {"project": project, "incident": incident[:50]})

    # ─────────────────────────────────────
    #  HELPER: Parse JSON response
    # ─────────────────────────────────────
    def _parse(self, response, task_id, task_type, meta):
        raw = response.content[0].text
        try:
            match = re.search(r'\{[\s\S]*\}', raw)
            data  = json.loads(match.group()) if match else json.loads(raw.strip())
        except Exception:
            data  = {"raw_output": raw}
        return {"task_id": task_id, "task_type": task_type,
                **meta, "output": data, "timestamp": datetime.now().isoformat()}

    # ─────────────────────────────────────
    #  SAVE: JSON + Obsidian .md
    # ─────────────────────────────────────
    async def _save_output(self, task_id, task_type, result):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{task_type}_{task_id}"

        # JSON
        json_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[{self.agent_id}] 💾 JSON  → {json_path}")

        # Obsidian .md
        is_envision = task_type in ["sil", "hazop", "proposal", "rca"]
        notes_path  = ENVISION_DIR if is_envision else NOTES_DIR
        md_path     = os.path.join(notes_path, f"{filename}.md")

        output  = result.get("output", {})
        title   = (result.get("topic") or result.get("project") or
                   result.get("work_title") or result.get("client") or task_type.upper())
        ts_r    = datetime.now().strftime("%Y-%m-%d %H:%M")
        folder  = "03-ENVISION" if is_envision else "01-TEACHING"

        md = f"""---
title: "{task_type.upper()} — {title}"
date: {ts_r}
tags: [AAOS, engineering-agent, {task_type}, {folder}]
agent: A1-Engineering
task_id: {task_id}
status: draft
---

# {task_type.upper()} — {title}

> Generated by **AAOS A1 Engineering Agent** | {ts_r}

---

"""
        # Task-specific sections
        if task_type == "lecture":
            md += f"## Learning Objectives\n"
            for o in output.get("objectives", []):
                md += f"- {o}\n"
            md += f"\n## Lecture Flow\n"
            for seg in output.get("lecture_flow", []):
                md += f"- **{seg.get('segment','')}** ({seg.get('duration',0)} min): {seg.get('content','')}\n"
            md += f"\n## Industry Example\n{output.get('industry_example','')}\n"
            md += f"\n## Assessment\n{output.get('assessment','')}\n"

        elif task_type == "lab":
            md += f"## Objective\n"
            for o in output.get("objective", []):
                md += f"- {o}\n"
            md += f"\n## Theory\n{output.get('theory','')}\n"
            md += f"\n## Procedure\n"
            for s in output.get("procedure", []):
                md += f"{s.get('step','')}. {s.get('action','')}\n"
            md += f"\n## Discussion Questions\n"
            for q in output.get("discussion_questions", []):
                md += f"- {q}\n"

        elif task_type == "exam":
            md += f"## Questions ({output.get('total_score',0)} pts)\n"
            for q in output.get("questions", []):
                md += f"\n**Q{q.get('no','')}** [{q.get('score',0)} pts | {q.get('bloom_level','')}]\n"
                md += f"{q.get('question','')}\n"
                if q.get("choices"):
                    for c in q.get("choices", []):
                        md += f"  {c}\n"
                md += f"*Answer: {q.get('answer','')}*\n"

        elif task_type == "sil":
            md += f"## SIL Verification Result\n"
            md += f"- **Target:** SIL {output.get('sil_target','')}\n"
            md += f"- **Achieved:** {output.get('sil_achieved','')}\n"
            md += f"- **Result:** {output.get('verification_result','')}\n"
            md += f"\n## Recommendations\n"
            for r in output.get("recommendations", []):
                md += f"- {r}\n"

        elif task_type == "hazop":
            md += f"## HAZOP Summary\n"
            md += f"- **Node:** {output.get('node','')}\n"
            md += f"- **Deviation:** {output.get('deviation','')}\n"
            md += f"- **Severity:** {output.get('severity','')}\n"
            md += f"- **Required SIL:** {output.get('required_sil','')}\n"
            md += f"\n## Recommendations\n"
            for r in output.get("recommendations", []):
                md += f"- {r}\n"

        elif task_type == "proposal":
            md += f"## Executive Summary\n{output.get('executive_summary','')}\n"
            md += f"\n## Deliverables\n"
            for d in output.get("deliverables", []):
                md += f"- {d}\n"

        elif task_type == "rca":
            md += f"## Root Causes\n"
            for rc in output.get("root_causes", []):
                md += f"- {rc}\n"
            md += f"\n## Corrective Actions\n"
            for ca in output.get("corrective_actions", []):
                md += f"- [{ca.get('priority','')}] {ca.get('action','')} ({ca.get('due_date','')})\n"

        md += "\n\n---\n*AAOS A1 Engineering Agent — Auto-generated*"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[{self.agent_id}] 📝 Note  → {md_path}")

    # ─────────────────────────────────────
    #  REGISTER กับ FastAPI Dispatcher
    # ─────────────────────────────────────
    async def register_with_dispatcher(self):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{FASTAPI_URL}/agents/register",
                    json={"agent_id": self.agent_id, "version": self.version,
                          "task_types": list(TASK_TYPES.keys()),
                          "triggers": TRIGGERS, "status": "active"},
                    timeout=5.0
                )
            print(f"[{self.agent_id}] ✅ Registered with Dispatcher: {resp.status_code}")
        except Exception as e:
            print(f"[{self.agent_id}] ⚠️  Dispatcher not available: {e}")
            print(f"[{self.agent_id}] 🔄 Running in standalone mode")


# ═══════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════
async def demo():
    agent = EngineeringAgent()
    await agent.register_with_dispatcher()

    print("\n" + "═"*60)
    print("  DEMO — A1 Engineering Agent")
    print("═"*60)

    # ── Demo 1: Lecture Outline — LabVIEW ──
    print("\n[DEMO 1] Lecture Outline — LabVIEW Virtual Instrumentation")
    r1 = await agent.run({
        "task_id"  : "eng-001",
        "task_type": "lecture",
        "payload"  : {
            "course"  : "01068012 Virtual Instrumentation/LabVIEW",
            "topic"   : "DAQ Fundamentals & Signal Conditioning",
            "week"    : 5,
            "level"   : "ปริญญาตรี",
            "students": 40,
            "duration": 90
        }
    })
    objs = r1["output"].get("objectives", [])
    print(f"  Objectives: {len(objs)} items")
    if objs:
        print(f"  → {objs[0][:80]}...")

    # ── Demo 2: SIL Verification — Biogas ──
    print("\n[DEMO 2] SIL Verification — Envision Biogas Project")
    r2 = await agent.run({
        "task_id"  : "eng-002",
        "task_type": "sil",
        "payload"  : {
            "project"        : "Biogas Power Plant — Nakhon Ratchasima",
            "project_type"   : "Biogas",
            "sif_description": "High pressure shutdown for biogas compressor",
            "sil_target"     : 2,
            "demand_mode"    : "Low Demand",
            "pfd_value"      : "8.5e-3"
        }
    })
    print(f"  SIL Achieved : {r2['output'].get('sil_achieved', 'N/A')}")
    print(f"  Result       : {r2['output'].get('verification_result', 'N/A')}")

    print("\n" + "═"*60)
    print("  ✅ A1 Engineering Agent — Demo Complete")
    print(f"  📁 Results: {OUTPUT_DIR}")
    print("═"*60)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AAOS — Agent A1: Engineering Agent v1.0                 ║
║     รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE Department          ║
╚══════════════════════════════════════════════════════════════╝

Task Types:
  lecture   → Lecture Outline (LabVIEW/Control/PM)
  lab       → Lab Manual
  exam      → Exam + Answer Key
  feedback  → Student Feedback
  sil       → SIL Verification (IEC 61511)
  hazop     → HAZOP/LOPA Analysis
  proposal  → Technical Proposal (Envision)
  rca       → Root Cause Analysis
""")
    asyncio.run(demo())
