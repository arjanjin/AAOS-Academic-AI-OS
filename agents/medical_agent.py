"""
╔══════════════════════════════════════════════════════════════╗
║        AAOS — Agent A3: Medical Agent v1.0                  ║
║        รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE                  ║
║        Domain: Anatomy, Pharmacology, Vajira MD 2567        ║
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

# ── Load .env ──────────────────────────────────────────────────
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
NOTES_DIR     = rf"{VAULT_ROOT}\04-KNOWLEDGE\medical-notes"   # Medical notes → 04-KNOWLEDGE
INBOX_DIR     = rf"{VAULT_ROOT}\05-INBOX\medical"             # Draft inbox

CLAUDE_MODEL  = "claude-sonnet-4-20250514"

TRIGGERS = ["anatomy", "vajira", "anki", "md program", "medical",
            "pharmacology", "physiology", "cpg", "clinical", "vr anatomy",
            "flashcard", "medical education", "@medical"]

TASK_TYPES = {
    "anki"       : "สร้าง Anki Flashcard Deck สำหรับ Medical Students",
    "summary"    : "สรุปบทเรียนแพทย์ (Anatomy/Pharm/Physio)",
    "quiz"       : "สร้างข้อสอบ MCQ แพทย์ (USMLE/Thai Board style)",
    "case_study" : "สร้าง Clinical Case Study",
    "vr_script"  : "เขียน VR Anatomy Scenario Script",
    "cpg"        : "สรุป Clinical Practice Guideline",
    "study_plan" : "วางแผนการเรียน MD Curriculum 2567",
}


# ═══════════════════════════════════════════
#  CLASS: MedicalAgent
# ═══════════════════════════════════════════
class MedicalAgent:

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "\n❌ ANTHROPIC_API_KEY not found!\n"
                "   Set: [System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY','sk-ant-xxx','User')"
            )
        self.client   = anthropic.Anthropic(api_key=api_key)
        self.agent_id = "A3-Medical"
        self.version  = "1.0.0"
        for d in [OUTPUT_DIR, NOTES_DIR, INBOX_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"[{self.agent_id}] ✅ Initialized — AAOS Medical Agent v{self.version}")
        print(f"[{self.agent_id}] 📁 Vault   : {VAULT_ROOT}")
        print(f"[{self.agent_id}] 📁 Output  : {OUTPUT_DIR}")
        print(f"[{self.agent_id}] 📁 Notes   : {NOTES_DIR}")

    # ─────────────────────────────────────
    #  MAIN: route task
    # ─────────────────────────────────────
    async def run(self, task: dict) -> dict:
        task_type = task.get("task_type", "summary").lower()
        payload   = task.get("payload", {})
        task_id   = task.get("task_id", str(uuid.uuid4())[:8])

        print(f"\n[{self.agent_id}] 🔄 Task #{task_id} | Type: {task_type}")
        print(f"[{self.agent_id}] 📥 Payload keys: {list(payload.keys())}")

        handlers = {
            "anki"      : self.create_anki_deck,
            "summary"   : self.create_summary,
            "quiz"      : self.create_quiz,
            "case_study": self.create_case_study,
            "vr_script" : self.create_vr_script,
            "cpg"       : self.summarize_cpg,
            "study_plan": self.create_study_plan,
        }

        handler = handlers.get(task_type, self.create_summary)
        result  = await handler(payload, task_id)
        await self._save_output(task_id, task_type, result)
        return result

    # ─────────────────────────────────────
    #  T1: Anki Flashcard Deck
    # ─────────────────────────────────────
    async def create_anki_deck(self, payload: dict, task_id: str) -> dict:
        topic    = payload.get("topic", "")
        subject  = payload.get("subject", "Anatomy")
        year     = payload.get("year", "MD Year 1")
        count    = payload.get("card_count", 20)
        focus    = payload.get("focus", "")  # e.g., "clinical relevance"

        prompt = f"""You are a medical education expert creating Anki flashcards for Vajira Hospital MD students.

Create {count} high-quality Anki flashcards for:
Subject: {subject}
Topic: {topic}
Year: {year} (Vajira MD Curriculum 2567)
Focus: {focus if focus else 'Core concepts + Clinical relevance'}

Card quality requirements:
- Front: Clear, specific question (not too broad)
- Back: Concise answer with key detail
- Tag each card with subject, topic, difficulty
- Include clinical pearl where relevant
- Use mnemonics where helpful
- Mix card types: Basic, Cloze, Image occlusion description

Output JSON:
{{
  "deck_name": "Vajira::{subject}::{topic}",
  "subject": "{subject}",
  "topic": "{topic}",
  "year": "{year}",
  "total_cards": {count},
  "cards": [
    {{
      "id": 1,
      "type": "Basic|Cloze",
      "front": "...",
      "back": "...",
      "tags": ["{subject}", "{topic}", "Year1"],
      "difficulty": "Easy|Medium|Hard",
      "clinical_pearl": "...",
      "mnemonic": "..."
    }}
  ],
  "study_tips": "...",
  "estimated_review_time": "X mins/day"
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "anki",
                           {"topic": topic, "subject": subject, "year": year})

    # ─────────────────────────────────────
    #  T2: Lecture Summary
    # ─────────────────────────────────────
    async def create_summary(self, payload: dict, task_id: str) -> dict:
        topic   = payload.get("topic", "")
        subject = payload.get("subject", "Anatomy")
        year    = payload.get("year", "MD Year 1")
        content = payload.get("content", "")
        style   = payload.get("style", "structured")  # structured | concept_map | table

        prompt = f"""You are a medical education expert helping MD students at Vajira Hospital.

Create a comprehensive study summary for:
Subject: {subject}
Topic: {topic}
Year: {year} (Vajira MD Curriculum 2567)
Style: {style}
Source content: {content[:2000] if content else 'Generate from standard medical knowledge'}

Structure the summary for maximum retention:
1. Overview / Big Picture
2. Key Concepts (with definitions)
3. Mechanisms / Pathways (if applicable)
4. Clinical Relevance
5. High-yield Points (⭐ for exam focus)
6. Common Mistakes to Avoid
7. Quick Review Questions (5 questions)

Output JSON:
{{
  "subject": "{subject}",
  "topic": "{topic}",
  "year": "{year}",
  "overview": "...",
  "key_concepts": [
    {{"concept": "...", "definition": "...", "high_yield": true}}
  ],
  "mechanisms": "...",
  "clinical_relevance": "...",
  "high_yield_points": ["⭐ ...", "⭐ ..."],
  "common_mistakes": ["...", "..."],
  "quick_questions": [
    {{"q": "...", "a": "..."}}
  ],
  "study_time_estimate": "X hours"
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "summary",
                           {"topic": topic, "subject": subject})

    # ─────────────────────────────────────
    #  T3: MCQ Quiz (Thai Board / USMLE style)
    # ─────────────────────────────────────
    async def create_quiz(self, payload: dict, task_id: str) -> dict:
        topic   = payload.get("topic", "")
        subject = payload.get("subject", "Anatomy")
        count   = payload.get("count", 10)
        style   = payload.get("style", "Thai Board")  # Thai Board | USMLE Step 1
        year    = payload.get("year", "MD Year 1")

        prompt = f"""You are an expert medical exam writer in {style} style.

Create {count} MCQ questions for:
Subject: {subject}
Topic: {topic}
Year: {year}
Style: {style}

Requirements:
- Each question: stem + 5 choices (A-E)
- One best answer
- Clinical vignette style preferred (patient scenario)
- Cover different cognitive levels (recall, application, analysis)
- Include detailed explanation for correct AND wrong answers
- Tag difficulty: Easy/Medium/Hard

Output JSON:
{{
  "subject": "{subject}",
  "topic": "{topic}",
  "style": "{style}",
  "total_questions": {count},
  "questions": [
    {{
      "no": 1,
      "difficulty": "Medium",
      "stem": "A 45-year-old patient presents with...",
      "choices": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "...",
        "E": "..."
      }},
      "correct_answer": "B",
      "explanation": "...",
      "wrong_answer_explanations": {{
        "A": "Wrong because...",
        "C": "Wrong because..."
      }},
      "high_yield_concept": "...",
      "tags": ["{subject}", "{topic}"]
    }}
  ],
  "performance_guide": {{
    "8-10": "Excellent — ready for board exam",
    "6-7": "Good — review weak areas",
    "below_6": "Need more study on this topic"
  }}
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "quiz",
                           {"topic": topic, "subject": subject, "style": style})

    # ─────────────────────────────────────
    #  T4: Clinical Case Study
    # ─────────────────────────────────────
    async def create_case_study(self, payload: dict, task_id: str) -> dict:
        topic      = payload.get("topic", "")
        subject    = payload.get("subject", "")
        difficulty = payload.get("difficulty", "Intermediate")
        year       = payload.get("year", "MD Year 2")

        prompt = f"""You are a clinical educator creating PBL case studies for Vajira MD students.

Create a complete Clinical Case Study:
Subject: {subject}
Topic/Condition: {topic}
Difficulty: {difficulty}
Year: {year}

Structure following PBL format:

Output JSON:
{{
  "case_title": "...",
  "subject": "{subject}",
  "topic": "{topic}",
  "learning_objectives": ["LO1: ...", "LO2: ...", "LO3: ..."],
  "patient_info": {{
    "age": 0,
    "gender": "...",
    "chief_complaint": "...",
    "hpi": "History of Present Illness...",
    "pmh": "Past Medical History...",
    "medications": ["..."],
    "allergies": "...",
    "social_history": "...",
    "family_history": "..."
  }},
  "physical_exam": {{
    "vitals": {{"BP": "...", "HR": "...", "RR": "...", "Temp": "...", "O2Sat": "..."}},
    "general": "...",
    "relevant_findings": ["finding1", "finding2"]
  }},
  "investigations": [
    {{"test": "...", "result": "...", "interpretation": "..."}}
  ],
  "discussion_questions": [
    {{"q": "What is the most likely diagnosis?", "hint": "...", "answer": "..."}}
  ],
  "diagnosis": "...",
  "management": {{
    "immediate": ["..."],
    "definitive": ["..."],
    "monitoring": ["..."]
  }},
  "key_teaching_points": ["⭐ ...", "⭐ ..."],
  "complications_to_watch": ["..."],
  "prognosis": "..."
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "case_study",
                           {"topic": topic, "subject": subject})

    # ─────────────────────────────────────
    #  T5: VR Anatomy Scenario Script
    # ─────────────────────────────────────
    async def create_vr_script(self, payload: dict, task_id: str) -> dict:
        structure = payload.get("structure", "")
        region    = payload.get("region", "")
        duration  = payload.get("duration", 15)
        platform  = payload.get("platform", "Unity/Android")
        year      = payload.get("year", "MD Year 1")

        prompt = f"""You are a VR medical education developer for Vajira Hospital collaboration.

Create a VR Anatomy Scenario Script for:
Anatomical Structure: {structure}
Region: {region}
Duration: {duration} minutes
Platform: {platform}
Students: {year} (Vajira MD Curriculum 2567)

Design an immersive VR learning experience:

Output JSON:
{{
  "scenario_title": "VR Anatomy: {structure}",
  "region": "{region}",
  "platform": "{platform}",
  "duration_minutes": {duration},
  "learning_objectives": ["...", "...", "..."],
  "scene_overview": "...",
  "scenes": [
    {{
      "scene_no": 1,
      "title": "...",
      "duration_seconds": 60,
      "description": "What student sees...",
      "narration": "Narrator script...",
      "interaction": "Student action (grab/rotate/highlight)...",
      "learning_point": "...",
      "unity_notes": "GameObject, layer, shader hints..."
    }}
  ],
  "interactive_labels": [
    {{"structure": "...", "label": "...", "pronunciation": "...", "clinical_note": "..."}}
  ],
  "assessment_checkpoint": {{
    "question": "Identify this structure...",
    "correct_action": "...",
    "feedback_correct": "...",
    "feedback_wrong": "..."
  }},
  "technical_requirements": {{
    "platform": "{platform}",
    "minimum_gpu": "...",
    "recommended_headset": "...",
    "asset_notes": "..."
  }},
  "clinical_integration": "How this anatomy relates to clinical practice..."
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "vr_script",
                           {"structure": structure, "region": region})

    # ─────────────────────────────────────
    #  T6: CPG Summary
    # ─────────────────────────────────────
    async def summarize_cpg(self, payload: dict, task_id: str) -> dict:
        condition = payload.get("condition", "")
        guideline = payload.get("guideline", "")
        content   = payload.get("content", "")
        audience  = payload.get("audience", "MD Student")

        prompt = f"""You are a clinical educator summarizing evidence-based guidelines for medical students.

Summarize the Clinical Practice Guideline for:
Condition: {condition}
Guideline: {guideline if guideline else 'Latest international/Thai guideline'}
Audience: {audience}
Source: {content[:2000] if content else 'Standard current medical guidelines'}

Create a student-friendly CPG summary:

Output JSON:
{{
  "condition": "{condition}",
  "guideline_source": "{guideline}",
  "last_updated": "...",
  "epidemiology": "...",
  "diagnostic_criteria": ["criterion1", "criterion2"],
  "investigations": [
    {{"test": "...", "indication": "...", "interpretation": "..."}}
  ],
  "treatment_algorithm": {{
    "first_line": ["..."],
    "second_line": ["..."],
    "refractory": ["..."]
  }},
  "key_medications": [
    {{"drug": "...", "dose": "...", "monitoring": "...", "side_effects": ["..."]}}
  ],
  "red_flags": ["🚨 ...", "🚨 ..."],
  "referral_criteria": ["..."],
  "follow_up": "...",
  "high_yield_for_exam": ["⭐ ...", "⭐ ..."],
  "common_clinical_mistakes": ["..."]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "cpg",
                           {"condition": condition, "guideline": guideline})

    # ─────────────────────────────────────
    #  T7: Study Plan (MD Curriculum 2567)
    # ─────────────────────────────────────
    async def create_study_plan(self, payload: dict, task_id: str) -> dict:
        year      = payload.get("year", "MD Year 1")
        subjects  = payload.get("subjects", [])
        exam_date = payload.get("exam_date", "")
        weeks     = payload.get("weeks", 8)
        hours_day = payload.get("hours_per_day", 6)

        prompt = f"""You are an academic advisor for Vajira Hospital MD Program 2567.

Create a detailed Study Plan:
Year: {year}
Subjects: {', '.join(subjects) if subjects else 'All Year subjects per Curriculum 2567'}
Exam Date: {exam_date if exam_date else 'End of semester'}
Duration: {weeks} weeks
Study Hours/Day: {hours_day} hours

Design an optimal, realistic study plan:

Output JSON:
{{
  "year": "{year}",
  "total_weeks": {weeks},
  "hours_per_day": {hours_day},
  "subjects": {json.dumps(subjects if subjects else ["Anatomy", "Physiology", "Biochemistry"])},
  "weekly_schedule": [
    {{
      "week": 1,
      "theme": "...",
      "daily_plan": {{
        "Monday": {{"morning": "...", "afternoon": "...", "evening": "..."}},
        "Tuesday": {{"morning": "...", "afternoon": "...", "evening": "..."}},
        "Wednesday": {{"morning": "...", "afternoon": "...", "evening": "..."}},
        "Thursday": {{"morning": "...", "afternoon": "...", "evening": "..."}},
        "Friday": {{"morning": "...", "afternoon": "...", "evening": "..."}},
        "Saturday": {{"morning": "Review...", "afternoon": "Practice MCQ...", "evening": "Rest"}},
        "Sunday": {{"morning": "Light review", "afternoon": "Rest", "evening": "Plan next week"}}
      }},
      "anki_target": "X new cards + Y reviews",
      "milestone": "Complete Chapter X, Y"
    }}
  ],
  "subject_allocation": {{
    "Anatomy": "X%",
    "Physiology": "X%",
    "Biochemistry": "X%"
  }},
  "study_techniques": [
    {{"technique": "Active Recall", "when": "...", "how": "..."}},
    {{"technique": "Spaced Repetition (Anki)", "when": "Daily", "how": "..."}}
  ],
  "exam_week_strategy": "...",
  "wellness_reminders": ["Sleep 7-8 hrs", "Exercise 30 min/day", "..."],
  "progress_checkpoints": [
    {{"week": 2, "checkpoint": "Complete X, score >70% on practice quiz"}}
  ]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "study_plan",
                           {"year": year, "weeks": weeks})

    # ─────────────────────────────────────
    #  HELPER: Parse JSON
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
        md_path = os.path.join(NOTES_DIR, f"{filename}.md")
        output  = result.get("output", {})
        topic   = (result.get("topic") or result.get("structure") or
                   result.get("condition") or result.get("year") or task_type.upper())
        subject = result.get("subject", "Medicine")
        ts_r    = datetime.now().strftime("%Y-%m-%d %H:%M")

        md = f"""---
title: "{task_type.upper()} — {topic}"
date: {ts_r}
tags: [AAOS, medical-agent, {task_type}, {subject}, Vajira]
agent: A3-Medical
task_id: {task_id}
subject: {subject}
status: draft
---

# {task_type.upper()} — {topic}

> Generated by **AAOS A3 Medical Agent** | {ts_r}

---

"""
        if task_type == "anki":
            cards = output.get("cards", [])
            md += f"## Deck: {output.get('deck_name','')}\n"
            md += f"**Total Cards:** {len(cards)} | **Est. Review:** {output.get('estimated_review_time','')}\n\n"
            for card in cards[:5]:  # Preview first 5
                md += f"### Card {card.get('id','')} [{card.get('difficulty','')}]\n"
                md += f"**Q:** {card.get('front','')}\n"
                md += f"**A:** {card.get('back','')}\n"
                if card.get("clinical_pearl"):
                    md += f"🏥 *{card.get('clinical_pearl','')}*\n"
                md += "\n"
            if len(cards) > 5:
                md += f"*...and {len(cards)-5} more cards in JSON file*\n"
            md += f"\n## Study Tips\n{output.get('study_tips','')}\n"

        elif task_type == "summary":
            md += f"## Overview\n{output.get('overview','')}\n\n"
            md += f"## High-Yield Points\n"
            for pt in output.get("high_yield_points", []):
                md += f"- {pt}\n"
            md += f"\n## Key Concepts\n"
            for kc in output.get("key_concepts", []):
                star = "⭐ " if kc.get("high_yield") else ""
                md += f"- {star}**{kc.get('concept','')}**: {kc.get('definition','')}\n"
            md += f"\n## Clinical Relevance\n{output.get('clinical_relevance','')}\n"
            md += f"\n## Common Mistakes\n"
            for m in output.get("common_mistakes", []):
                md += f"- ⚠️ {m}\n"
            md += f"\n## Quick Questions\n"
            for qa in output.get("quick_questions", []):
                md += f"**Q:** {qa.get('q','')}\n**A:** {qa.get('a','')}\n\n"

        elif task_type == "quiz":
            qs = output.get("questions", [])
            md += f"## {output.get('style','')} Quiz — {len(qs)} Questions\n\n"
            for q in qs:
                md += f"### Q{q.get('no','')} [{q.get('difficulty','')}]\n"
                md += f"{q.get('stem','')}\n\n"
                for k, v in q.get("choices", {}).items():
                    prefix = "✅ " if k == q.get("correct_answer") else ""
                    md += f"{prefix}**{k}.** {v}\n"
                md += f"\n> **Answer: {q.get('correct_answer','')}** — {q.get('explanation','')}\n\n"

        elif task_type == "case_study":
            pt = output.get("patient_info", {})
            md += f"## Case: {output.get('case_title','')}\n\n"
            md += f"**Patient:** {pt.get('age','')} y/o {pt.get('gender','')}\n"
            md += f"**CC:** {pt.get('chief_complaint','')}\n\n"
            md += f"## Learning Objectives\n"
            for lo in output.get("learning_objectives", []):
                md += f"- {lo}\n"
            md += f"\n## Diagnosis\n{output.get('diagnosis','')}\n"
            md += f"\n## Key Teaching Points\n"
            for tp in output.get("key_teaching_points", []):
                md += f"- {tp}\n"

        elif task_type == "vr_script":
            md += f"## VR Scenario: {output.get('scenario_title','')}\n"
            md += f"**Platform:** {output.get('platform','')} | **Duration:** {output.get('duration_minutes','')} min\n\n"
            md += f"## Learning Objectives\n"
            for lo in output.get("learning_objectives", []):
                md += f"- {lo}\n"
            md += f"\n## Scenes ({len(output.get('scenes',[]))} scenes)\n"
            for sc in output.get("scenes", []):
                md += f"### Scene {sc.get('scene_no','')}: {sc.get('title','')}\n"
                md += f"{sc.get('description','')}\n"
                md += f"> 🎙️ *{sc.get('narration','')[:100]}...*\n\n"

        elif task_type == "cpg":
            md += f"## {output.get('condition','')} — CPG Summary\n"
            md += f"**Source:** {output.get('guideline_source','')}\n\n"
            md += f"## 🚨 Red Flags\n"
            for rf in output.get("red_flags", []):
                md += f"- {rf}\n"
            md += f"\n## Treatment\n"
            tx = output.get("treatment_algorithm", {})
            md += f"**First Line:** {', '.join(tx.get('first_line',[]))}\n"
            md += f"**Second Line:** {', '.join(tx.get('second_line',[]))}\n"
            md += f"\n## ⭐ High-Yield for Exam\n"
            for hy in output.get("high_yield_for_exam", []):
                md += f"- {hy}\n"

        elif task_type == "study_plan":
            md += f"## Study Plan — {output.get('year','')}\n"
            md += f"**Duration:** {output.get('total_weeks','')} weeks | **Hours/day:** {output.get('hours_per_day','')}\n\n"
            md += f"## Subject Allocation\n"
            for subj, pct in output.get("subject_allocation", {}).items():
                md += f"- {subj}: {pct}\n"
            md += f"\n## Study Techniques\n"
            for tech in output.get("study_techniques", []):
                md += f"- **{tech.get('technique','')}**: {tech.get('how','')}\n"
            md += f"\n## Wellness Reminders\n"
            for w in output.get("wellness_reminders", []):
                md += f"- 💚 {w}\n"

        md += "\n\n---\n*AAOS A3 Medical Agent — Auto-generated | Vajira MD Curriculum 2567*"

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
    agent = MedicalAgent()
    await agent.register_with_dispatcher()

    print("\n" + "═"*60)
    print("  DEMO — A3 Medical Agent")
    print("═"*60)

    # ── Demo 1: Anki Deck — Brachial Plexus ──
    print("\n[DEMO 1] Anki Deck — Brachial Plexus Anatomy")
    r1 = await agent.run({
        "task_id"  : "med-001",
        "task_type": "anki",
        "payload"  : {
            "subject"    : "Anatomy",
            "topic"      : "Brachial Plexus",
            "year"       : "MD Year 1",
            "card_count" : 15,
            "focus"      : "Nerve roots, injuries, clinical syndromes"
        }
    })
    cards = r1["output"].get("cards", [])
    print(f"  Deck: {r1['output'].get('deck_name','')}")
    print(f"  Cards generated: {len(cards)}")
    if cards:
        print(f"  Sample: {cards[0].get('front','')[:60]}...")

    # ── Demo 2: MCQ Quiz — Upper Limb ──
    print("\n[DEMO 2] Clinical Quiz — Upper Limb Nerve Injuries")
    r2 = await agent.run({
        "task_id"  : "med-002",
        "task_type": "quiz",
        "payload"  : {
            "subject": "Anatomy",
            "topic"  : "Upper Limb Nerve Injuries",
            "count"  : 5,
            "style"  : "Thai Board",
            "year"   : "MD Year 1"
        }
    })
    qs = r2["output"].get("questions", [])
    print(f"  Questions generated: {len(qs)}")
    if qs:
        print(f"  Sample Q1: {qs[0].get('stem','')[:80]}...")

    print("\n" + "═"*60)
    print("  ✅ A3 Medical Agent — Demo Complete")
    print(f"  📁 Results: {OUTPUT_DIR}")
    print(f"  📝 Notes  : {NOTES_DIR}")
    print("═"*60)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AAOS — Agent A3: Medical Agent v1.0                     ║
║     รศ.ดร.อาจินต์ น่วมสำราญ | KMITL / Vajira Collaboration  ║
╚══════════════════════════════════════════════════════════════╝

Task Types:
  anki        → Anki Flashcard Deck (Vajira MD Curriculum 2567)
  summary     → Lecture Summary (Anatomy/Pharm/Physio)
  quiz        → MCQ Quiz (Thai Board / USMLE style)
  case_study  → Clinical Case Study (PBL format)
  vr_script   → VR Anatomy Scenario Script (Unity/Android)
  cpg         → Clinical Practice Guideline Summary
  study_plan  → MD Study Plan with Anki integration
""")
    asyncio.run(demo())
