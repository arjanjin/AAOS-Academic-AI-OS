"""
╔══════════════════════════════════════════════════════════════╗
║        AAOS — Agent A4: Business Agent v1.0                 ║
║        รศ.ดร.อาจินต์ น่วมสำราญ | KMITL ICE                  ║
║        Domain: ESG, SET50, Finance, CFA, Envision Strategy  ║
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
NOTES_DIR     = rf"{VAULT_ROOT}\08-PROFESSIONAL\AAOS-notes"  # Business → 08-PROFESSIONAL
ENVISION_DIR  = rf"{VAULT_ROOT}\03-ENVISION\AAOS-notes"      # Envision strategy → 03-ENVISION
INBOX_DIR     = rf"{VAULT_ROOT}\05-INBOX\business"

CLAUDE_MODEL  = "claude-sonnet-4-20250514"

TRIGGERS = ["esg", "set50", "cfa", "finance", "investment", "stock",
            "business", "strategy", "envision", "revenue", "market",
            "portfolio", "valuation", "@business"]

TASK_TYPES = {
    "esg"        : "ESG Screening & Analysis (SET50)",
    "financial"  : "Financial Analysis & Valuation",
    "cfa"        : "CFA Level 1 Study Material",
    "strategy"   : "Business Strategy (Envision I&C)",
    "pitch"      : "Pitch Deck Content",
    "market"     : "Market Research & Competitive Analysis",
    "report"     : "Business Report / Executive Summary",
}


# ═══════════════════════════════════════════
#  CLASS: BusinessAgent
# ═══════════════════════════════════════════
class BusinessAgent:

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "\n❌ ANTHROPIC_API_KEY not found!\n"
                "   Set: [System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY','sk-ant-xxx','User')"
            )
        self.client   = anthropic.Anthropic(api_key=api_key)
        self.agent_id = "A4-Business"
        self.version  = "1.0.0"
        for d in [OUTPUT_DIR, NOTES_DIR, ENVISION_DIR, INBOX_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"[{self.agent_id}] ✅ Initialized — AAOS Business Agent v{self.version}")
        print(f"[{self.agent_id}] 📁 Vault      : {VAULT_ROOT}")
        print(f"[{self.agent_id}] 📁 Output     : {OUTPUT_DIR}")
        print(f"[{self.agent_id}] 📁 Professional: {NOTES_DIR}")
        print(f"[{self.agent_id}] 📁 Envision   : {ENVISION_DIR}")

    # ─────────────────────────────────────
    #  MAIN: route task
    # ─────────────────────────────────────
    async def run(self, task: dict) -> dict:
        task_type = task.get("task_type", "report").lower()
        payload   = task.get("payload", {})
        task_id   = task.get("task_id", str(uuid.uuid4())[:8])

        print(f"\n[{self.agent_id}] 🔄 Task #{task_id} | Type: {task_type}")
        print(f"[{self.agent_id}] 📥 Payload keys: {list(payload.keys())}")

        handlers = {
            "esg"       : self.esg_screening,
            "financial" : self.financial_analysis,
            "cfa"       : self.cfa_study,
            "strategy"  : self.business_strategy,
            "pitch"     : self.pitch_deck,
            "market"    : self.market_research,
            "report"    : self.executive_report,
        }

        handler = handlers.get(task_type, self.executive_report)
        result  = await handler(payload, task_id)
        await self._save_output(task_id, task_type, result)
        return result

    # ─────────────────────────────────────
    #  T1: ESG Screening
    # ─────────────────────────────────────
    async def esg_screening(self, payload: dict, task_id: str) -> dict:
        company   = payload.get("company", "")
        ticker    = payload.get("ticker", "")
        sector    = payload.get("sector", "")
        framework = payload.get("framework", "GRI/SASB/TCFD")
        index     = payload.get("index", "SET50")

        prompt = f"""You are an ESG analyst specializing in Thai capital markets and {index} companies.

Perform comprehensive ESG Screening for:
Company: {company} ({ticker})
Sector: {sector}
Index: {index}
Framework: {framework}

Analyze all ESG dimensions:

Output JSON:
{{
  "company": "{company}",
  "ticker": "{ticker}",
  "sector": "{sector}",
  "index": "{index}",
  "analysis_date": "{datetime.now().strftime('%Y-%m-%d')}",
  "esg_scores": {{
    "environmental": {{
      "score": 0,
      "max": 100,
      "grade": "A/B/C/D",
      "key_metrics": {{
        "carbon_intensity": "...",
        "renewable_energy_pct": "...",
        "water_usage": "...",
        "waste_management": "..."
      }},
      "strengths": ["..."],
      "risks": ["..."]
    }},
    "social": {{
      "score": 0,
      "max": 100,
      "grade": "A/B/C/D",
      "key_metrics": {{
        "employee_turnover": "...",
        "safety_record": "...",
        "community_investment": "...",
        "diversity_score": "..."
      }},
      "strengths": ["..."],
      "risks": ["..."]
    }},
    "governance": {{
      "score": 0,
      "max": 100,
      "grade": "A/B/C/D",
      "key_metrics": {{
        "board_independence": "...",
        "audit_quality": "...",
        "transparency": "...",
        "anti_corruption": "..."
      }},
      "strengths": ["..."],
      "risks": ["..."]
    }}
  }},
  "overall_esg_score": 0,
  "overall_grade": "A/B/C/D",
  "controversies": ["..."],
  "red_flags": ["🚨 ..."],
  "investment_recommendation": "Include/Exclude/Watch",
  "rationale": "...",
  "peer_comparison": "vs. sector average...",
  "improvement_areas": ["..."]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "esg",
                           {"company": company, "ticker": ticker, "sector": sector})

    # ─────────────────────────────────────
    #  T2: Financial Analysis
    # ─────────────────────────────────────
    async def financial_analysis(self, payload: dict, task_id: str) -> dict:
        company  = payload.get("company", "")
        ticker   = payload.get("ticker", "")
        period   = payload.get("period", "FY2024")
        method   = payload.get("valuation_method", ["DCF", "P/E", "EV/EBITDA"])
        data     = payload.get("financial_data", {})

        prompt = f"""You are a CFA-certified financial analyst specializing in Thai equity markets.

Perform Financial Analysis for:
Company: {company} ({ticker})
Period: {period}
Valuation Methods: {', '.join(method)}
Available Data: {json.dumps(data, ensure_ascii=False) if data else 'Use estimated figures for demonstration'}

Output JSON:
{{
  "company": "{company}",
  "ticker": "{ticker}",
  "period": "{period}",
  "income_statement_highlights": {{
    "revenue": 0,
    "revenue_growth": "X%",
    "gross_margin": "X%",
    "ebitda_margin": "X%",
    "net_income": 0,
    "eps": 0
  }},
  "balance_sheet_highlights": {{
    "total_assets": 0,
    "total_debt": 0,
    "net_debt": 0,
    "equity": 0,
    "book_value_per_share": 0
  }},
  "key_ratios": {{
    "pe_ratio": 0,
    "pb_ratio": 0,
    "ev_ebitda": 0,
    "roe": "X%",
    "roa": "X%",
    "debt_to_equity": 0,
    "current_ratio": 0,
    "dividend_yield": "X%"
  }},
  "valuation": {{
    "dcf_value": 0,
    "pe_value": 0,
    "ev_ebitda_value": 0,
    "average_target_price": 0,
    "current_price": 0,
    "upside_downside": "X%"
  }},
  "investment_thesis": "...",
  "risks": ["...", "..."],
  "catalysts": ["...", "..."],
  "recommendation": "Strong Buy/Buy/Hold/Sell/Strong Sell",
  "analyst_note": "..."
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "financial",
                           {"company": company, "ticker": ticker, "period": period})

    # ─────────────────────────────────────
    #  T3: CFA Study Material
    # ─────────────────────────────────────
    async def cfa_study(self, payload: dict, task_id: str) -> dict:
        topic   = payload.get("topic", "")
        level   = payload.get("level", "Level 1")
        area    = payload.get("area", "")  # e.g., Equity, Fixed Income, Ethics
        count   = payload.get("question_count", 10)

        prompt = f"""You are a CFA instructor creating study materials.

Create comprehensive CFA study content for:
Level: CFA {level}
Study Area: {area}
Topic: {topic}

Include:
1. Topic Summary (key concepts, formulas)
2. Practice Questions ({count} questions, exam-style)
3. Common exam traps and tricks

Output JSON:
{{
  "level": "{level}",
  "area": "{area}",
  "topic": "{topic}",
  "los": ["LOS: The candidate should be able to..."],
  "key_concepts": [
    {{"concept": "...", "formula": "...", "example": "..."}}
  ],
  "important_formulas": [
    {{"name": "...", "formula": "...", "variables": {{}}}}
  ],
  "practice_questions": [
    {{
      "no": 1,
      "difficulty": "Easy|Medium|Hard",
      "question": "...",
      "choices": {{"A": "...", "B": "...", "C": "..."}},
      "answer": "A/B/C",
      "explanation": "...",
      "los_reference": "..."
    }}
  ],
  "exam_tips": ["...", "..."],
  "common_mistakes": ["...", "..."],
  "memory_tricks": ["...", "..."],
  "estimated_study_time": "X hours",
  "exam_weight": "X% of {level} exam"
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "cfa",
                           {"topic": topic, "level": level, "area": area})

    # ─────────────────────────────────────
    #  T4: Business Strategy (Envision)
    # ─────────────────────────────────────
    async def business_strategy(self, payload: dict, task_id: str) -> dict:
        company   = payload.get("company", "Envision I&C Engineering Groups")
        goal      = payload.get("goal", "")
        horizon   = payload.get("horizon", "3 years")
        context   = payload.get("context", "")
        framework = payload.get("framework", "SWOT + OKR")

        prompt = f"""You are a strategic management consultant specializing in engineering and technology companies.

Develop a Business Strategy for:
Company: {company}
Strategic Goal: {goal}
Time Horizon: {horizon}
Context: {context if context else 'Thai engineering firm specializing in SIS/IEC 61511, growing AI integration'}
Framework: {framework}

Company background: Envision I&C Engineering Groups — SIL-rated safety instrumented systems,
projects: Solar, Biogas, Water KO Drum, CEMS, HIPPS. CEO: รศ.ดร.อาจินต์ น่วมสำราญ (KMITL).

Output JSON:
{{
  "company": "{company}",
  "strategic_goal": "{goal}",
  "horizon": "{horizon}",
  "vision": "...",
  "mission": "...",
  "swot": {{
    "strengths": ["...", "...", "..."],
    "weaknesses": ["...", "..."],
    "opportunities": ["...", "...", "..."],
    "threats": ["...", "..."]
  }},
  "strategic_options": [
    {{"option": "...", "rationale": "...", "risk": "High/Med/Low", "roi_potential": "High/Med/Low"}}
  ],
  "recommended_strategy": "...",
  "okrs": [
    {{
      "objective": "...",
      "key_results": [
        {{"kr": "...", "baseline": "...", "target": "...", "timeline": "..."}}
      ]
    }}
  ],
  "implementation_roadmap": [
    {{"phase": 1, "name": "...", "duration": "...", "actions": ["..."], "budget": "..."}}
  ],
  "kpis": [
    {{"metric": "...", "current": "...", "target": "...", "frequency": "Monthly/Quarterly"}}
  ],
  "risk_mitigation": [
    {{"risk": "...", "mitigation": "...", "owner": "..."}}
  ],
  "quick_wins": ["..."],
  "investment_required": "THB X million"
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "strategy",
                           {"company": company, "goal": goal, "horizon": horizon})

    # ─────────────────────────────────────
    #  T5: Pitch Deck Content
    # ─────────────────────────────────────
    async def pitch_deck(self, payload: dict, task_id: str) -> dict:
        company   = payload.get("company", "")
        audience  = payload.get("audience", "Investor")
        ask       = payload.get("ask", "")
        problem   = payload.get("problem", "")
        solution  = payload.get("solution", "")
        slides    = payload.get("slide_count", 12)

        prompt = f"""You are a pitch deck expert helping startups and engineering firms raise capital.

Create a compelling Pitch Deck outline for:
Company: {company}
Audience: {audience}
Ask: {ask}
Problem: {problem}
Solution: {solution}
Slides: {slides}

Output JSON:
{{
  "company": "{company}",
  "audience": "{audience}",
  "ask": "{ask}",
  "slides": [
    {{
      "slide_no": 1,
      "title": "Cover",
      "headline": "...",
      "content_points": ["...", "..."],
      "visual_suggestion": "...",
      "speaker_notes": "..."
    }}
  ],
  "key_metrics_to_highlight": ["...", "..."],
  "storytelling_arc": "Problem → Solution → Market → Traction → Team → Ask",
  "design_tips": ["...", "..."],
  "common_investor_questions": [
    {{"question": "...", "suggested_answer": "..."}}
  ]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "pitch",
                           {"company": company, "audience": audience, "ask": ask})

    # ─────────────────────────────────────
    #  T6: Market Research
    # ─────────────────────────────────────
    async def market_research(self, payload: dict, task_id: str) -> dict:
        market    = payload.get("market", "")
        geography = payload.get("geography", "Thailand")
        segment   = payload.get("segment", "")
        purpose   = payload.get("purpose", "Business development")

        prompt = f"""You are a market research analyst with expertise in Thai industrial and technology markets.

Conduct Market Research for:
Market/Industry: {market}
Geography: {geography}
Segment Focus: {segment}
Purpose: {purpose}

Output JSON:
{{
  "market": "{market}",
  "geography": "{geography}",
  "segment": "{segment}",
  "market_size": {{
    "current_tam": "USD/THB X billion",
    "sam": "USD/THB X billion",
    "som": "USD/THB X million",
    "growth_rate": "X% CAGR",
    "forecast_year": "2028"
  }},
  "market_drivers": ["...", "...", "..."],
  "market_barriers": ["...", "..."],
  "key_trends": [
    {{"trend": "...", "impact": "High/Med/Low", "timeframe": "..."}}
  ],
  "competitive_landscape": [
    {{
      "company": "...",
      "market_share": "X%",
      "strengths": ["..."],
      "weaknesses": ["..."],
      "positioning": "..."
    }}
  ],
  "customer_segments": [
    {{"segment": "...", "size": "...", "needs": ["..."], "pain_points": ["..."]}}
  ],
  "regulatory_environment": "...",
  "opportunity_areas": ["...", "..."],
  "threat_analysis": ["...", "..."],
  "strategic_recommendations": ["...", "..."],
  "data_sources": ["Industry reports", "SET filings", "BOI data"]
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "market",
                           {"market": market, "geography": geography})

    # ─────────────────────────────────────
    #  T7: Executive Report
    # ─────────────────────────────────────
    async def executive_report(self, payload: dict, task_id: str) -> dict:
        topic    = payload.get("topic", "")
        company  = payload.get("company", "Envision I&C Engineering Groups")
        period   = payload.get("period", datetime.now().strftime("%Y Q%q"))
        audience = payload.get("audience", "Board of Directors")
        data     = payload.get("data", "")

        prompt = f"""You are a business analyst writing executive-level reports.

Create an Executive Report:
Topic: {topic}
Company: {company}
Period: {period}
Audience: {audience}
Data/Context: {data[:2000] if data else 'Use industry-standard estimates'}

Output JSON:
{{
  "report_title": "...",
  "company": "{company}",
  "period": "{period}",
  "prepared_for": "{audience}",
  "prepared_by": "\u0e23\u0e28.\u0e14\u0e23.\u0e2d\u0e32\u0e08\u0e34\u0e19\u0e15\u0e4c \u0e19\u0e48\u0e27\u0e21\u0e2a\u0e33\u0e23\u0e32\u0e0d | CEO, Envision I&C",
  "date": "{datetime.now().strftime('%Y-%m-%d')}",
  "executive_summary": "...",
  "key_highlights": [
    {{"metric": "...", "value": "...", "vs_last_period": "▲/▼ X%", "status": "On Track/At Risk/Behind"}}
  ],
  "sections": [
    {{"title": "...", "content": "...", "findings": ["..."]}}
  ],
  "financial_summary": {{
    "revenue": "...",
    "expenses": "...",
    "profit": "...",
    "pipeline": "..."
  }},
  "risks_and_issues": [
    {{"risk": "...", "severity": "High/Med/Low", "mitigation": "..."}}
  ],
  "decisions_required": ["...", "..."],
  "next_steps": [
    {{"action": "...", "owner": "...", "due": "...", "priority": "High/Med/Low"}}
  ],
  "appendix": "..."
}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse(response, task_id, "report",
                           {"topic": topic, "company": company, "period": period})

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

        # Route: Envision tasks → 03-ENVISION, others → 08-PROFESSIONAL
        is_envision = task_type in ["strategy", "pitch", "report"]
        notes_path  = ENVISION_DIR if is_envision else NOTES_DIR
        md_path     = os.path.join(notes_path, f"{filename}.md")

        output  = result.get("output", {})
        title   = (result.get("topic") or result.get("company") or
                   result.get("market") or result.get("ticker") or task_type.upper())
        ts_r    = datetime.now().strftime("%Y-%m-%d %H:%M")
        folder  = "03-ENVISION" if is_envision else "08-PROFESSIONAL"

        md = f"""---
title: "{task_type.upper()} — {title}"
date: {ts_r}
tags: [AAOS, business-agent, {task_type}, {folder}]
agent: A4-Business
task_id: {task_id}
status: draft
---

# {task_type.upper()} — {title}

> Generated by **AAOS A4 Business Agent** | {ts_r}

---

"""
        if task_type == "esg":
            scores = output.get("esg_scores", {})
            md += f"## ESG Overall Score: {output.get('overall_esg_score','N/A')}/100 ({output.get('overall_grade','N/A')})\n\n"
            for dim in ["environmental", "social", "governance"]:
                d = scores.get(dim, {})
                md += f"### {dim.upper()} — {d.get('score','?')}/100 ({d.get('grade','?')})\n"
                for s in d.get("strengths", []):
                    md += f"- ✅ {s}\n"
                for r in d.get("risks", []):
                    md += f"- ⚠️ {r}\n"
                md += "\n"
            md += f"## Recommendation: {output.get('investment_recommendation','N/A')}\n"
            md += f"{output.get('rationale','')}\n"
            md += f"\n## 🚨 Red Flags\n"
            for rf in output.get("red_flags", []):
                md += f"- {rf}\n"

        elif task_type == "financial":
            val = output.get("valuation", {})
            md += f"## Recommendation: **{output.get('recommendation','N/A')}**\n\n"
            md += f"## Valuation\n"
            md += f"- Target Price: {val.get('average_target_price','N/A')}\n"
            md += f"- Current Price: {val.get('current_price','N/A')}\n"
            md += f"- Upside/Downside: {val.get('upside_downside','N/A')}\n"
            md += f"\n## Investment Thesis\n{output.get('investment_thesis','')}\n"
            md += f"\n## Key Risks\n"
            for r in output.get("risks", []):
                md += f"- {r}\n"
            md += f"\n## Catalysts\n"
            for c in output.get("catalysts", []):
                md += f"- {c}\n"

        elif task_type == "cfa":
            md += f"## {output.get('level','')} | {output.get('area','')} | {output.get('topic','')}\n"
            md += f"**Exam Weight:** {output.get('exam_weight','')}\n\n"
            md += f"## Key Concepts\n"
            for kc in output.get("key_concepts", []):
                md += f"- **{kc.get('concept','')}**: {kc.get('example','')}\n"
            md += f"\n## Important Formulas\n"
            for f in output.get("important_formulas", []):
                md += f"- **{f.get('name','')}**: `{f.get('formula','')}`\n"
            md += f"\n## Exam Tips\n"
            for tip in output.get("exam_tips", []):
                md += f"- 💡 {tip}\n"
            md += f"\n## Practice Questions ({len(output.get('practice_questions',[]))} ข้อ)\n"
            for q in output.get("practice_questions", [])[:3]:
                md += f"\n**Q{q.get('no','')}** [{q.get('difficulty','')}]\n{q.get('question','')}\n"
                md += f"*Answer: {q.get('answer','')}* — {q.get('explanation','')[:80]}...\n"

        elif task_type == "strategy":
            md += f"## Vision\n{output.get('vision','')}\n"
            md += f"\n## Mission\n{output.get('mission','')}\n"
            swot = output.get("swot", {})
            md += f"\n## SWOT\n"
            md += f"**Strengths:** {', '.join(swot.get('strengths',[]))}\n"
            md += f"**Opportunities:** {', '.join(swot.get('opportunities',[]))}\n"
            md += f"\n## OKRs\n"
            for okr in output.get("okrs", []):
                md += f"### 🎯 {okr.get('objective','')}\n"
                for kr in okr.get("key_results", []):
                    md += f"- KR: {kr.get('kr','')} → Target: {kr.get('target','')} by {kr.get('timeline','')}\n"
            md += f"\n## Quick Wins\n"
            for qw in output.get("quick_wins", []):
                md += f"- ⚡ {qw}\n"

        elif task_type == "pitch":
            md += f"## {output.get('company','')} — Pitch to {output.get('audience','')}\n"
            md += f"**Ask:** {output.get('ask','')}\n\n"
            for slide in output.get("slides", []):
                md += f"### Slide {slide.get('slide_no','')}: {slide.get('title','')}\n"
                md += f"**Headline:** {slide.get('headline','')}\n"
                for pt in slide.get("content_points", []):
                    md += f"- {pt}\n"
                md += "\n"

        elif task_type == "market":
            size = output.get("market_size", {})
            md += f"## Market Size\n"
            md += f"- TAM: {size.get('current_tam','')}\n"
            md += f"- SAM: {size.get('sam','')}\n"
            md += f"- Growth: {size.get('growth_rate','')}\n\n"
            md += f"## Key Trends\n"
            for t in output.get("key_trends", []):
                md += f"- [{t.get('impact','')}] {t.get('trend','')}\n"
            md += f"\n## Strategic Recommendations\n"
            for r in output.get("strategic_recommendations", []):
                md += f"- {r}\n"

        elif task_type == "report":
            md += f"## Executive Summary\n{output.get('executive_summary','')}\n"
            md += f"\n## Key Highlights\n"
            for h in output.get("key_highlights", []):
                md += f"- **{h.get('metric','')}**: {h.get('value','')} {h.get('vs_last_period','')} [{h.get('status','')}]\n"
            md += f"\n## Decisions Required\n"
            for d in output.get("decisions_required", []):
                md += f"- ❗ {d}\n"
            md += f"\n## Next Steps\n"
            for ns in output.get("next_steps", []):
                md += f"- [{ns.get('priority','')}] {ns.get('action','')} — {ns.get('owner','')} by {ns.get('due','')}\n"

        md += "\n\n---\n*AAOS A4 Business Agent — Auto-generated*"

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
    agent = BusinessAgent()
    await agent.register_with_dispatcher()

    print("\n" + "═"*60)
    print("  DEMO — A4 Business Agent")
    print("═"*60)

    # ── Demo 1: ESG Screening — PTT ──
    print("\n[DEMO 1] ESG Screening — PTT (SET50)")
    r1 = await agent.run({
        "task_id"  : "biz-001",
        "task_type": "esg",
        "payload"  : {
            "company"  : "PTT Public Company Limited",
            "ticker"   : "PTT",
            "sector"   : "Energy",
            "framework": "GRI/SASB/TCFD",
            "index"    : "SET50"
        }
    })
    scores = r1["output"].get("esg_scores", {})
    print(f"  Overall Score : {r1['output'].get('overall_esg_score','N/A')}/100")
    print(f"  Recommendation: {r1['output'].get('investment_recommendation','N/A')}")

    # ── Demo 2: Envision Business Strategy ──
    print("\n[DEMO 2] Business Strategy — Envision I&C")
    r2 = await agent.run({
        "task_id"  : "biz-002",
        "task_type": "strategy",
        "payload"  : {
            "company" : "Envision I&C Engineering Groups",
            "goal"    : "ขยายธุรกิจ AI-Integrated Safety Systems และเพิ่มรายได้ 3x ใน 3 ปี",
            "horizon" : "3 years",
            "context" : "Thai engineering firm, SIL expert, expanding into AI+SIS integration",
            "framework": "SWOT + OKR"
        }
    })
    okrs = r2["output"].get("okrs", [])
    print(f"  Vision : {r2['output'].get('vision','N/A')[:70]}...")
    print(f"  OKRs   : {len(okrs)} objectives")

    print("\n" + "═"*60)
    print("  ✅ A4 Business Agent — Demo Complete")
    print(f"  📁 Results: {OUTPUT_DIR}")
    print("═"*60)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AAOS — Agent A4: Business Agent v1.0                    ║
║     รศ.ดร.อาจินต์ น่วมสำราญ | KMITL / Envision I&C          ║
╚══════════════════════════════════════════════════════════════╝

Task Types:
  esg        → ESG Screening & Analysis (SET50)
  financial  → Financial Analysis & Valuation
  cfa        → CFA Level 1 Study Material
  strategy   → Business Strategy (Envision I&C)
  pitch      → Pitch Deck Content
  market     → Market Research & Competitive Analysis
  report     → Executive Report / Business Summary
""")
    asyncio.run(demo())
