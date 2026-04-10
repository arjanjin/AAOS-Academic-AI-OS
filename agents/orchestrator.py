"""
╔══════════════════════════════════════════════════════════════╗
║        AAOS — Orchestrator v1.0                             ║
║        รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE                  ║
║        Route: Input → Agent A1/A2/A3/A4 → Output           ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path

# ── Add agents folder to path ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from engineering_agent import EngineeringAgent
from researcher_agent  import ResearcherAgent
from medical_agent     import MedicalAgent
from business_agent    import BusinessAgent

# ─────────────────────────────────────────
#  ROUTING TABLE — keyword → agent + task_type
# ─────────────────────────────────────────
ROUTING_TABLE = [

    # ── A1 Engineering ─────────────────────────────────────────
    {"agent": "A1", "task_type": "lecture",
     "keywords": ["lecture", "สอน", "outline", "บทเรียน", "labview", "control systems",
                  "virtual instrumentation", "project management course"]},

    {"agent": "A1", "task_type": "lab",
     "keywords": ["lab manual", "คู่มือปฏิบัติการ", "lab ", "ปฏิบัติการ", "experiment"]},

    {"agent": "A1", "task_type": "exam",
     "keywords": ["exam", "quiz", "ข้อสอบ", "โจทย์", "test", "assessment"]},

    {"agent": "A1", "task_type": "feedback",
     "keywords": ["feedback", "ความคิดเห็น", "student", "นักศึกษา", "grade", "คะแนน"]},

    {"agent": "A1", "task_type": "sil",
     "keywords": ["sil", "iec 61511", "iec 61508", "safety instrumented",
                  "sis verification", "pfd", "pfh", "functional safety"]},

    {"agent": "A1", "task_type": "hazop",
     "keywords": ["hazop", "lopa", "hazard", "deviation", "safeguard", "risk assessment"]},

    {"agent": "A1", "task_type": "rca",
     "keywords": ["rca", "root cause", "incident", "อุบัติการณ์", "5-why", "fishbone"]},

    {"agent": "A1", "task_type": "proposal",
     "keywords": ["technical proposal", "envision proposal", "scope of work",
                  "i&c proposal", "instrumentation proposal"]},

    # ── A2 Researcher ──────────────────────────────────────────
    {"agent": "A2", "task_type": "abstract",
     "keywords": ["abstract", "บทคัดย่อ", "journal paper", "ieee", "scopus", "write abstract"]},

    {"agent": "A2", "task_type": "lit_review",
     "keywords": ["literature review", "lit review", "research gap", "scite",
                  "systematic review", "ทบทวนวรรณกรรม"]},

    {"agent": "A2", "task_type": "ide_ipa",
     "keywords": ["ide-ipa", "ide ipa", "impact pathway", "เส้นทางผลกระทบ",
                  "สกสว", "บพข", "อววน", "วิเคราะห์ผลกระทบ"]},

    {"agent": "A2", "task_type": "reviewer",
     "keywords": ["reviewer", "peer review", "revision", "major revision",
                  "minor revision", "response to reviewer"]},

    {"agent": "A2", "task_type": "proposal",
     "keywords": ["research proposal", "ขอทุน", "โครงการวิจัย", "funding proposal",
                  "grant proposal", "วช", "jsps"]},

    {"agent": "A2", "task_type": "summary",
     "keywords": ["summarize paper", "paper summary", "สรุปงานวิจัย", "article summary"]},

    # ── A3 Medical ─────────────────────────────────────────────
    {"agent": "A3", "task_type": "anki",
     "keywords": ["anki", "flashcard", "flash card", "deck", "spaced repetition"]},

    {"agent": "A3", "task_type": "quiz",
     "keywords": ["thai board", "usmle", "medical quiz", "mcq medical",
                  "ข้อสอบแพทย์", "board exam"]},

    {"agent": "A3", "task_type": "summary",
     "keywords": ["anatomy", "กายวิภาค", "physiology", "pharmacology", "biochemistry",
                  "vajira", "md year", "นักศึกษาแพทย์", "medical summary"]},

    {"agent": "A3", "task_type": "case_study",
     "keywords": ["clinical case", "case study", "pbl", "patient", "case presentation",
                  "chief complaint", "diagnosis"]},

    {"agent": "A3", "task_type": "vr_script",
     "keywords": ["vr anatomy", "virtual reality", "unity", "3d anatomy", "vr script",
                  "immersive", "vr scenario"]},

    {"agent": "A3", "task_type": "cpg",
     "keywords": ["cpg", "clinical practice guideline", "treatment guideline",
                  "management guideline", "แนวทางเวชปฏิบัติ"]},

    {"agent": "A3", "task_type": "study_plan",
     "keywords": ["study plan", "แผนการเรียน", "md curriculum", "medical study",
                  "learning schedule", "medical timetable"]},

    # ── A4 Business ────────────────────────────────────────────
    {"agent": "A4", "task_type": "esg",
     "keywords": ["esg", "set50", "set 50", "sustainability", "environmental social",
                  "esg screening", "esg score", "esg analysis"]},

    {"agent": "A4", "task_type": "financial",
     "keywords": ["financial analysis", "valuation", "dcf", "p/e ratio", "ev/ebitda",
                  "stock analysis", "equity research", "financial report"]},

    {"agent": "A4", "task_type": "cfa",
     "keywords": ["cfa", "chartered financial", "level 1", "fixed income",
                  "equity investment", "derivatives", "portfolio management"]},

    {"agent": "A4", "task_type": "strategy",
     "keywords": ["business strategy", "strategic plan", "okr", "swot",
                  "envision strategy", "growth strategy", "business plan"]},

    {"agent": "A4", "task_type": "pitch",
     "keywords": ["pitch deck", "pitch", "investor presentation", "fundraising",
                  "startup deck", "investor"]},

    {"agent": "A4", "task_type": "market",
     "keywords": ["market research", "market analysis", "competitive analysis",
                  "tam sam som", "market size", "industry analysis"]},

    {"agent": "A4", "task_type": "report",
     "keywords": ["executive report", "business report", "monthly report",
                  "quarterly report", "board report", "management report"]},
]


# ═══════════════════════════════════════════
#  CLASS: Orchestrator
# ═══════════════════════════════════════════
class Orchestrator:

    def __init__(self):
        print("\n[Orchestrator] 🚀 Initializing AAOS Orchestrator v1.0...")
        self.agents = {
            "A1": EngineeringAgent(),
            "A2": ResearcherAgent(),
            "A3": MedicalAgent(),
            "A4": BusinessAgent(),
        }
        self.history = []
        print("[Orchestrator] ✅ All 4 Agents loaded and ready!")
        print("[Orchestrator] 📡 Waiting for input...\n")

    # ─────────────────────────────────────
    #  ROUTE: เลือก Agent + task_type
    # ─────────────────────────────────────
    def route(self, text: str) -> tuple[str, str]:
        """คืนค่า (agent_id, task_type) ที่เหมาะสมที่สุด"""
        text_lower = text.lower()
        best_match = None
        best_score = 0

        for rule in ROUTING_TABLE:
            score = sum(1 for kw in rule["keywords"] if kw in text_lower)
            if score > best_score:
                best_score = score
                best_match = rule

        if best_match and best_score > 0:
            return best_match["agent"], best_match["task_type"]

        # Default: A1 Engineering (lecture)
        return "A1", "lecture"

    # ─────────────────────────────────────
    #  PARSE: ดึง payload จาก text input
    # ─────────────────────────────────────
    def parse_payload(self, text: str, agent_id: str, task_type: str) -> dict:
        """สร้าง payload อัตโนมัติจากข้อความ"""
        payload = {"raw_input": text}

        if agent_id == "A1":
            if task_type == "lecture":
                payload.update({
                    "course": self._extract(text, ["วิชา", "course:", "01068"]) or "ICE Course",
                    "topic" : text[:100],
                    "week"  : self._extract_num(text, "week") or 1,
                    "level" : "ปริญญาตรี",
                    "students": 40,
                    "duration": 90,
                })
            elif task_type == "sil":
                payload.update({
                    "project"     : self._extract(text, ["project:", "โครงการ"]) or "Envision Project",
                    "project_type": "Biogas",
                    "sif_description": text[:150],
                    "sil_target"  : self._extract_num(text, "sil") or 2,
                    "demand_mode" : "Low Demand",
                })
            elif task_type == "hazop":
                payload.update({
                    "project"   : "Envision Project",
                    "node"      : self._extract(text, ["node:"]) or "Process Node",
                    "deviation" : self._extract(text, ["deviation:"]) or text[:80],
                    "consequence": self._extract(text, ["consequence:"]) or "Hazardous event",
                })
            else:
                payload["topic"] = text[:200]

        elif agent_id == "A2":
            if task_type == "abstract":
                payload.update({
                    "title"   : text[:100],
                    "journal" : "IEEE Access",
                    "problem" : text[:200],
                    "method"  : "",
                    "result"  : "",
                    "contribution": "",
                    "keywords": [],
                })
            elif task_type == "ide_ipa":
                payload.update({
                    "project_name": text[:100],
                    "agency"      : "บพข.",
                    "budget"      : 3000000,
                    "duration"    : "2 ปี",
                    "objectives"  : text[:200],
                    "methodology" : "",
                })
            else:
                payload["topic"] = text[:200]

        elif agent_id == "A3":
            if task_type == "anki":
                payload.update({
                    "subject"    : self._extract(text, ["anatomy", "physiology", "pharmacology"]) or "Anatomy",
                    "topic"      : text[:100],
                    "year"       : "MD Year 1",
                    "card_count" : self._extract_num(text, "card") or 15,
                })
            elif task_type == "quiz":
                payload.update({
                    "subject": "Medicine",
                    "topic"  : text[:100],
                    "count"  : self._extract_num(text, "question") or 5,
                    "style"  : "Thai Board",
                    "year"   : "MD Year 1",
                })
            else:
                payload.update({"subject": "Medicine", "topic": text[:100]})

        elif agent_id == "A4":
            if task_type == "esg":
                payload.update({
                    "company"  : self._extract(text, ["company:", "บริษัท"]) or "Thai Company",
                    "ticker"   : self._extract(text, ["ticker:", "หุ้น"]) or "TBD",
                    "sector"   : "General",
                    "framework": "GRI/SASB/TCFD",
                    "index"    : "SET50",
                })
            elif task_type == "strategy":
                payload.update({
                    "company" : "Envision I&C Engineering Groups",
                    "goal"    : text[:150],
                    "horizon" : "3 years",
                    "framework": "SWOT + OKR",
                })
            else:
                payload.update({"topic": text[:200], "company": "Envision I&C"})

        return payload

    def _extract(self, text: str, markers: list) -> str:
        for m in markers:
            if m.lower() in text.lower():
                idx = text.lower().find(m.lower()) + len(m)
                return text[idx:idx+50].strip().split("\n")[0].strip(" :")
        return ""

    def _extract_num(self, text: str, keyword: str) -> int:
        import re
        pattern = rf'{keyword}\s*[:#]?\s*(\d+)'
        m = re.search(pattern, text.lower())
        return int(m.group(1)) if m else 0

    # ─────────────────────────────────────
    #  PROCESS: รับ input แล้วส่งไป Agent
    # ─────────────────────────────────────
    async def process(self, user_input: str, task_id: str = None) -> dict:
        if not task_id:
            task_id = f"orch-{datetime.now().strftime('%H%M%S')}"

        # Route
        agent_id, task_type = self.route(user_input)
        agent = self.agents[agent_id]

        print(f"\n[Orchestrator] 📨 Input: {user_input[:60]}...")
        print(f"[Orchestrator] 🎯 Route → {agent_id} | task: {task_type}")

        # Parse payload
        payload = self.parse_payload(user_input, agent_id, task_type)

        # Run agent
        task = {"task_id": task_id, "task_type": task_type, "payload": payload}
        result = await agent.run(task)

        # Log
        log_entry = {
            "task_id"  : task_id,
            "timestamp": datetime.now().isoformat(),
            "input"    : user_input[:100],
            "route"    : f"{agent_id} → {task_type}",
            "status"   : "completed",
        }
        self.history.append(log_entry)

        print(f"[Orchestrator] ✅ Done: {agent_id} | {task_type} | #{task_id}")
        return result

    # ─────────────────────────────────────
    #  INTERACTIVE: โหมดพิมพ์โต้ตอบ
    # ─────────────────────────────────────
    async def interactive(self):
        print("═"*60)
        print("  AAOS Orchestrator — Interactive Mode")
        print("  พิมพ์คำสั่งหรือหัวข้อที่ต้องการ")
        print("  พิมพ์ 'quit' เพื่อออก | 'history' ดู log")
        print("═"*60)

        while True:
            try:
                user_input = input("\n🎯 AAOS > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Orchestrator] 👋 Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("[Orchestrator] 👋 Goodbye!")
                break
            if user_input.lower() == "history":
                self._show_history()
                continue
            if user_input.lower() == "help":
                self._show_help()
                continue

            await self.process(user_input)

    def _show_history(self):
        print(f"\n📋 Session History ({len(self.history)} tasks):")
        for h in self.history[-10:]:
            print(f"  [{h['task_id']}] {h['route']} — {h['input'][:50]}")

    def _show_help(self):
        print("""
📖 ตัวอย่างคำสั่ง:

  A1 Engineering:
    "สร้าง lecture outline วิชา LabVIEW week 5"
    "SIL verification สำหรับ Biogas project SIL 2"
    "HAZOP analysis node 3 deviation high pressure"

  A2 Researcher:
    "เขียน abstract สำหรับ IEEE Access paper เรื่อง Arduino SIS"
    "วิเคราะห์ IDE-IPA โครงการ AI safety systems บพข."
    "literature review เรื่อง safety instrumented systems"

  A3 Medical:
    "สร้าง anki deck เรื่อง brachial plexus 20 cards"
    "MCQ Thai Board เรื่อง upper limb nerve injuries 5 ข้อ"
    "clinical case study hypertension"

  A4 Business:
    "ESG screening บริษัท PTT SET50"
    "Envision business strategy 3 ปี"
    "market research Thai safety engineering sector"
""")


# ═══════════════════════════════════════════
#  DEMO: ทดสอบ Auto-routing
# ═══════════════════════════════════════════
async def demo():
    orch = Orchestrator()

    print("\n" + "═"*60)
    print("  DEMO — AAOS Orchestrator Auto-routing")
    print("═"*60)

    test_inputs = [
        ("orch-001", "สร้าง lecture outline วิชา Virtual Instrumentation LabVIEW week 5 DAQ"),
        ("orch-002", "เขียน abstract สำหรับ IEEE Access paper เรื่อง Arduino SIS IEC 61511"),
        ("orch-003", "สร้าง anki deck เรื่อง brachial plexus anatomy 15 cards Vajira"),
        ("orch-004", "ESG screening analysis PTT SET50 GRI framework"),
    ]

    for task_id, text in test_inputs:
        agent_id, task_type = orch.route(text)
        print(f"\n  Input  : {text[:55]}...")
        print(f"  Route  : {agent_id} → {task_type}")
        agent_names = {"A1":"Engineering","A2":"Researcher","A3":"Medical","A4":"Business"}
        print(f"  Agent  : {agent_names[agent_id]}")

    print("\n" + "─"*60)
    print("  Running 1 full task (A2 Abstract)...")
    print("─"*60)

    result = await orch.process(
        "เขียน abstract IEEE Access paper เรื่อง BPCS-SIS separation Arduino IEC 61511",
        task_id="orch-demo"
    )

    print(f"\n  ✅ Task complete!")
    print(f"  Words: {result.get('output',{}).get('word_count', 'N/A')}")

    print("\n" + "═"*60)
    print("  ✅ Orchestrator Demo Complete")
    print("═"*60)

    print("\n💡 Tip: รัน Interactive Mode ด้วย:")
    print("   python orchestrator.py --interactive\n")


# ═══════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AAOS — Orchestrator v1.0                                ║
║     รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE Department          ║
╚══════════════════════════════════════════════════════════════╝

Mode:
  python orchestrator.py                → Demo (auto-routing test)
  python orchestrator.py --interactive  → Interactive chat mode
""")

    import sys
    if "--interactive" in sys.argv:
        async def run_interactive():
            orch = Orchestrator()
            await orch.interactive()
        asyncio.run(run_interactive())
    else:
        asyncio.run(demo())
