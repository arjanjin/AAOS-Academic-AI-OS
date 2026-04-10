"""
╔══════════════════════════════════════════════════════════════╗
║        AAOS — LangGraph Orchestrator v1.0 (L6)             ║
║        รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE                  ║
║        Flow: Plan → Execute → Evaluate → Improve → Output   ║
╚══════════════════════════════════════════════════════════════╝

Install:
    pip install langgraph langchain langchain-anthropic

Usage:
    python langgraph_orchestrator.py
    python langgraph_orchestrator.py --interactive
"""

import os
import json
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional
import operator

# ── LangGraph imports ─────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# ── AAOS Agents ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from engineering_agent import EngineeringAgent
from researcher_agent  import ResearcherAgent
from medical_agent     import MedicalAgent
from business_agent    import BusinessAgent

# ── Load .env ─────────────────────────────────────────────────
def _load_env():
    for p in [Path(r"D:\arjin-vault\.env"), Path(__file__).parent / ".env"]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            print(f"[ENV] ✅ Loaded: {p}")
            return
_load_env()

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CLAUDE_MODEL  = "claude-sonnet-4-20250514"
AAOS_ROOT     = r"D:\arjin-vault\06-OUTPUT\AAOS"
OUTPUT_DIR    = rf"{AAOS_ROOT}\results"
QUALITY_THRESHOLD = 7  # ถ้า score < 7 → retry
MAX_RETRIES   = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
#  STATE: ข้อมูลที่ไหลใน Graph
# ─────────────────────────────────────────
class AAOSState(TypedDict):
    # Input
    user_input    : str
    task_id       : str

    # Plan
    plan          : dict          # {agent, task_type, steps, rationale}

    # Execute
    result        : dict          # output จาก Agent
    retry_count   : int

    # Evaluate
    score         : float         # 0-10
    weaknesses    : List[str]
    should_retry  : bool

    # Improve
    improved_prompt: Optional[str]

    # Final
    final_output  : dict
    log           : Annotated[List[str], operator.add]


# ═══════════════════════════════════════════
#  CLASS: LangGraphOrchestrator
# ═══════════════════════════════════════════
class LangGraphOrchestrator:

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("❌ ANTHROPIC_API_KEY not found!")

        # LLM สำหรับ Planner, Evaluator, Improver
        self.llm = ChatAnthropic(
            model=CLAUDE_MODEL,
            api_key=api_key,
            max_tokens=2000
        )

        # AAOS Agents
        self.agents = {
            "A1": EngineeringAgent(),
            "A2": ResearcherAgent(),
            "A3": MedicalAgent(),
            "A4": BusinessAgent(),
        }

        # สร้าง Graph
        self.graph = self._build_graph()
        print("[LangGraph] ✅ Orchestrator L6 initialized")
        print(f"[LangGraph] 🔄 Flow: Plan → Execute → Evaluate → Improve → Output")

    # ─────────────────────────────────────
    #  BUILD GRAPH
    # ─────────────────────────────────────
    def _build_graph(self):
        workflow = StateGraph(AAOSState)

        # เพิ่ม Nodes
        workflow.add_node("planner",   self._node_planner)
        workflow.add_node("executor",  self._node_executor)
        workflow.add_node("evaluator", self._node_evaluator)
        workflow.add_node("improver",  self._node_improver)
        workflow.add_node("output",    self._node_output)

        # Entry point
        workflow.set_entry_point("planner")

        # Edges
        workflow.add_edge("planner",  "executor")
        workflow.add_edge("executor", "evaluator")

        # Conditional: retry หรือ output
        workflow.add_conditional_edges(
            "evaluator",
            self._should_retry,
            {
                "retry"  : "improver",
                "accept" : "output",
            }
        )
        workflow.add_edge("improver", "executor")
        workflow.add_edge("output",   END)

        return workflow.compile()

    # ─────────────────────────────────────
    #  NODE 1: PLANNER
    # ─────────────────────────────────────
    async def _node_planner(self, state: AAOSState) -> dict:
        print(f"\n[Planner] 🧠 Analyzing: {state['user_input'][:60]}...")

        prompt = f"""You are the AAOS Planner for Assoc. Prof. Dr. Arjin Numsomran (KMITL ICE).

Analyze this request and create an execution plan:
"{state['user_input']}"

Available Agents:
- A1 Engineering: lecture, lab, exam, feedback, sil, hazop, rca, proposal
- A2 Researcher: abstract, lit_review, ide_ipa, reviewer, proposal, summary  
- A3 Medical: anki, summary, quiz, case_study, vr_script, cpg, study_plan
- A4 Business: esg, financial, cfa, strategy, pitch, market, report

Respond with ONLY this JSON (no markdown):
{{
  "agent": "A1|A2|A3|A4",
  "task_type": "one of the task types above",
  "rationale": "why this agent and task type",
  "key_requirements": ["req1", "req2"],
  "success_criteria": ["criterion1", "criterion2"],
  "payload_hints": {{
    "key": "value hints to help build the payload"
  }}
}}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content

        try:
            import re
            match = re.search(r'\{[\s\S]*\}', raw)
            plan = json.loads(match.group()) if match else {}
        except Exception:
            plan = {"agent": "A2", "task_type": "summary",
                    "rationale": "fallback", "key_requirements": [],
                    "success_criteria": [], "payload_hints": {}}

        print(f"[Planner] 🎯 Route: {plan.get('agent')} → {plan.get('task_type')}")
        print(f"[Planner] 📋 Rationale: {plan.get('rationale','')[:60]}")

        return {
            "plan"       : plan,
            "retry_count": 0,
            "log"        : [f"[Planner] {plan.get('agent')} → {plan.get('task_type')}"]
        }

    # ─────────────────────────────────────
    #  NODE 2: EXECUTOR
    # ─────────────────────────────────────
    async def _node_executor(self, state: AAOSState) -> dict:
        plan      = state["plan"]
        agent_id  = plan.get("agent", "A2")
        task_type = plan.get("task_type", "summary")
        hints     = plan.get("payload_hints", {})
        retry     = state.get("retry_count", 0)
        improved  = state.get("improved_prompt")

        print(f"\n[Executor] ⚙️  Running {agent_id} | {task_type} (attempt {retry+1})")

        # Build payload
        payload = self._build_payload(
            state["user_input"], task_type, hints, improved
        )

        task = {
            "task_id"  : f"{state['task_id']}-r{retry}",
            "task_type": task_type,
            "payload"  : payload
        }

        agent  = self.agents[agent_id]
        result = await agent.run(task)

        return {
            "result"     : result,
            "retry_count": retry + 1,
            "log"        : [f"[Executor] {agent_id}:{task_type} attempt {retry+1}"]
        }

    # ─────────────────────────────────────
    #  NODE 3: EVALUATOR
    # ─────────────────────────────────────
    async def _node_evaluator(self, state: AAOSState) -> dict:
        result   = state["result"]
        plan     = state["plan"]
        output   = result.get("output", {})

        print(f"\n[Evaluator] 🔍 Evaluating output quality...")

        # ถ้า raw_output → parse error → score ต่ำ
        if "raw_output" in output and len(output) == 1:
            print(f"[Evaluator] ⚠️  JSON parse failed → score 4")
            return {
                "score"       : 4.0,
                "weaknesses"  : ["JSON parsing failed", "Output not structured"],
                "should_retry": True,
                "log"         : ["[Evaluator] score=4 (parse error)"]
            }

        prompt = f"""You are a quality evaluator for AAOS academic outputs.

Evaluate this output for: "{state['user_input']}"
Agent used: {plan.get('agent')} | Task: {plan.get('task_type')}
Success criteria: {plan.get('success_criteria', [])}

Output summary: {json.dumps(output, ensure_ascii=False)[:1000]}

Score the output 1-10 based on:
- Completeness (all required sections present)
- Quality (depth, accuracy, usefulness)
- Relevance (matches the original request)
- Structure (well-organized JSON)

Respond with ONLY this JSON:
{{
  "score": 8.5,
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "specific_improvements": ["improve1", "improve2"]
}}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content

        try:
            import re
            match = re.search(r'\{[\s\S]*\}', raw)
            eval_result = json.loads(match.group()) if match else {}
        except Exception:
            eval_result = {"score": 6.0, "strengths": [], "weaknesses": [], "specific_improvements": []}

        score       = float(eval_result.get("score", 6.0))
        weaknesses  = eval_result.get("weaknesses", [])
        retry_count = state.get("retry_count", 1)
        should_retry = score < QUALITY_THRESHOLD and retry_count <= MAX_RETRIES

        print(f"[Evaluator] 📊 Score: {score}/10 | Retry: {should_retry}")
        if weaknesses:
            print(f"[Evaluator] ⚠️  Weaknesses: {', '.join(weaknesses[:2])}")

        return {
            "score"       : score,
            "weaknesses"  : weaknesses,
            "should_retry": should_retry,
            "log"         : [f"[Evaluator] score={score} retry={should_retry}"]
        }

    # ─────────────────────────────────────
    #  CONDITION: retry หรือ accept
    # ─────────────────────────────────────
    def _should_retry(self, state: AAOSState) -> str:
        if state.get("should_retry") and state.get("retry_count", 0) <= MAX_RETRIES:
            return "retry"
        return "accept"

    # ─────────────────────────────────────
    #  NODE 4: IMPROVER
    # ─────────────────────────────────────
    async def _node_improver(self, state: AAOSState) -> dict:
        weaknesses = state.get("weaknesses", [])
        score      = state.get("score", 5.0)

        print(f"\n[Improver] 🔧 Score {score} < {QUALITY_THRESHOLD} → Improving prompt...")

        prompt = f"""You are a prompt engineer improving AAOS agent instructions.

Original request: "{state['user_input']}"
Current score: {score}/10
Weaknesses found: {weaknesses}

Write an IMPROVED, more specific version of the original request that addresses all weaknesses.
Be more explicit about: depth, format, completeness, and specific requirements.
Keep it as a natural language instruction (not JSON).

Improved request:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        improved = response.content.strip()

        print(f"[Improver] ✨ Improved: {improved[:80]}...")

        return {
            "improved_prompt": improved,
            "log"            : [f"[Improver] prompt improved (was score {score})"]
        }

    # ─────────────────────────────────────
    #  NODE 5: OUTPUT
    # ─────────────────────────────────────
    async def _node_output(self, state: AAOSState) -> dict:
        result = state["result"]
        score  = state.get("score", 0)
        plan   = state["plan"]

        print(f"\n[Output] ✅ Final score: {score}/10")
        print(f"[Output] 💾 Saving final result...")

        final = {
            "task_id"    : state["task_id"],
            "user_input" : state["user_input"],
            "agent"      : plan.get("agent"),
            "task_type"  : plan.get("task_type"),
            "score"      : score,
            "retries"    : state.get("retry_count", 1) - 1,
            "result"     : result,
            "log"        : state.get("log", []),
            "timestamp"  : datetime.now().isoformat()
        }

        # Save JSON
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUT_DIR, f"{ts}_L6_{state['task_id']}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        print(f"[Output] 📁 Saved: {filepath}")

        return {
            "final_output": final,
            "log"         : [f"[Output] saved score={score}"]
        }

    # ─────────────────────────────────────
    #  HELPER: Build payload
    # ─────────────────────────────────────
    def _build_payload(self, user_input: str, task_type: str,
                       hints: dict, improved: Optional[str]) -> dict:
        """สร้าง payload จาก input + hints + improved prompt"""
        text = improved or user_input

        payload = {"raw_input": text, **hints}

        # เพิ่ม defaults ตาม task_type
        defaults = {
            "abstract"  : {"title": text[:100], "journal": "IEEE Access",
                           "problem": text, "method": "", "result": "",
                           "contribution": "", "keywords": []},
            "ide_ipa"   : {"project_name": text[:100], "agency": "บพข.",
                           "budget": 3000000, "duration": "2 ปี",
                           "objectives": text, "methodology": ""},
            "lecture"   : {"course": hints.get("course","ICE Course"),
                           "topic": text[:100], "week": 1,
                           "level": "ปริญญาตรี", "students": 40, "duration": 90},
            "sil"       : {"project": hints.get("project","Envision"),
                           "project_type": "Biogas",
                           "sif_description": text[:150],
                           "sil_target": 2, "demand_mode": "Low Demand"},
            "anki"      : {"subject": "Anatomy", "topic": text[:100],
                           "year": "MD Year 1", "card_count": 15},
            "quiz"      : {"subject": "Medicine", "topic": text[:100],
                           "count": 5, "style": "Thai Board", "year": "MD Year 1"},
            "esg"       : {"company": hints.get("company","Thai Company"),
                           "ticker": hints.get("ticker","TBD"),
                           "sector": "General", "framework": "GRI/SASB",
                           "index": "SET50"},
            "strategy"  : {"company": "Envision I&C Engineering Groups",
                           "goal": text[:150], "horizon": "3 years",
                           "framework": "SWOT + OKR"},
        }

        if task_type in defaults:
            for k, v in defaults[task_type].items():
                payload.setdefault(k, v)
        else:
            payload.setdefault("topic", text[:200])

        return payload

    # ─────────────────────────────────────
    #  PROCESS: Main entry
    # ─────────────────────────────────────
    async def process(self, user_input: str) -> dict:
        task_id = f"L6-{datetime.now().strftime('%H%M%S')}"
        print(f"\n{'═'*60}")
        print(f"[LangGraph] 🚀 Task #{task_id}")
        print(f"[LangGraph] 📝 Input: {user_input[:70]}...")
        print(f"{'═'*60}")

        initial_state: AAOSState = {
            "user_input"    : user_input,
            "task_id"       : task_id,
            "plan"          : {},
            "result"        : {},
            "retry_count"   : 0,
            "score"         : 0.0,
            "weaknesses"    : [],
            "should_retry"  : False,
            "improved_prompt": None,
            "final_output"  : {},
            "log"           : [],
        }

        final_state = await self.graph.ainvoke(initial_state)

        score = final_state.get("score", 0)
        print(f"\n{'═'*60}")
        print(f"[LangGraph] 🏁 COMPLETE | Score: {score}/10")
        print(f"[LangGraph] 🔄 Retries: {final_state.get('retry_count',1)-1}")
        print(f"{'═'*60}\n")

        return final_state.get("final_output", {})

    # ─────────────────────────────────────
    #  INTERACTIVE MODE
    # ─────────────────────────────────────
    async def interactive(self):
        print("\n" + "═"*60)
        print("  AAOS LangGraph L6 — Interactive Mode")
        print("  Plan → Execute → Evaluate → Improve → Output")
        print("  'quit' \u0e2d\u0e2d\u0e01 | 'demo' \u0e17\u0e14\u0e2a\u0e2d\u0e1a")
        print("═"*60)

        while True:
            try:
                user_input = input("\n\ud83e\udde0 AAOS-L6 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[LangGraph] \ud83d\udc4b Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("[LangGraph] \ud83d\udc4b Goodbye!")
                break
            if user_input.lower() == "demo":
                await self._run_demo()
                continue

            result = await self.process(user_input)
            score  = result.get("score", 0)
            agent  = result.get("agent", "")
            ttype  = result.get("task_type", "")
            print(f"\n\u2705 Done | {agent} \u2192 {ttype} | Score: {score}/10")

    async def _run_demo(self):
        demos = [
            "\u0e40\u0e02\u0e35\u0e22\u0e19 abstract IEEE Access \u0e40\u0e23\u0e37\u0e48\u0e2d\u0e07 BPCS-SIS separation \u0e14\u0e49\u0e27\u0e22 Arduino IEC 61511",
        ]
        for d in demos:
            await self.process(d)


# ═══════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551     AAOS \u2014 LangGraph Orchestrator v1.0 (L6)              \u2551
\u2551     \u0e23\u0e28.\u0e14\u0e23.\u0e2d\u0e32\u0e08\u0e34\u0e19\u0e15\u0e4c \u0e19\u0e48\u0e27\u0e21\u0e2a\u0e33\u0e23\u0e32\u0e0d | KMITL ICE Department          \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d

Flow: Plan \u2192 Execute \u2192 Evaluate \u2192 Improve \u2192 Output
      \u2191___________________________________|  (if score < 7)

Mode:
  python langgraph_orchestrator.py                \u2192 Demo
  python langgraph_orchestrator.py --interactive  \u2192 Interactive
""")

    async def main():
        orch = LangGraphOrchestrator()
        if "--interactive" in sys.argv:
            await orch.interactive()
        else:
            # Demo
            result = await orch.process(
                "\u0e40\u0e02\u0e35\u0e22\u0e19 abstract IEEE Access paper \u0e40\u0e23\u0e37\u0e48\u0e2d\u0e07 BPCS-SIS separation \u0e14\u0e49\u0e27\u0e22 Arduino \u0e41\u0e25\u0e30 IEC 61511"
            )
            print(f"\nFinal Score : {result.get('score',0)}/10")
            print(f"Agent       : {result.get('agent')} \u2192 {result.get('task_type')}")
            print(f"Retries     : {result.get('retries',0)}")

    asyncio.run(main())
