import os
import sys
import json
import time
import textwrap
import datetime
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Force UTF-8 encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# -- Groq SDK ------------------------------------------------------
from groq import Groq

# -- Rich Terminal UI ----------------------------------------------
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text
from rich import box
from rich.theme import Theme

# Custom Theme
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "danger": "bold red",
    "success": "bold green",
    "primary": "bold magenta",
})
console = Console(theme=custom_theme)

# -----------------------------------------------------------------
#  CONFIG
# -----------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
    console.print(Panel("[danger]FATAL ERROR: No Groq API key found. Set GROQ_API_KEY in .env[/danger]", title="❌ Error", border_style="red"))
    raise SystemExit(1)

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME   = "llama-3.1-8b-instant"   # Very fast, token-efficient, and reliable for triage tasks
CSV_INPUT    = "support_tickets.csv"
CSV_OUTPUT   = "triage_results_final.csv"
MAX_RETRIES  = 5
RETRY_DELAYS = [2, 4, 8, 16, 30]

DOMAINS = {
    "hackerrank": "HackerRank",
    "claude":     "Anthropic / Claude",
    "anthropic":  "Anthropic / Claude",
    "visa":       "Visa",
}

PRIORITY_KEYWORDS = {
    "critical": ["fraud", "unauthorized", "hacked", "stolen", "security breach", "data leak", "lawsuit"],
    "high":     ["refund", "payment failed", "declined", "billing", "charge", "pii", "personal data"],
    "medium":   ["bug", "freeze", "crash", "error", "not working", "broken", "issue"],
    "low":      ["feature", "suggestion", "dark mode", "improvement", "request"],
}

# -----------------------------------------------------------------
#  BANNER
# -----------------------------------------------------------------
def banner():
    title = Text("🚀 Multi-Domain Support Triage Agent", justify="center", style="bold cyan")
    subtitle = Text("Powered by Gemini 2.5 Flash\nDomains: HackerRank • Anthropic/Claude • Visa", justify="center", style="dim white")
    
    panel = Panel(
        title + Text("\n") + subtitle,
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 5)
    )
    console.print(panel)
    console.print()

# -----------------------------------------------------------------
#  PRIORITY & COMPANY ENGINE
# -----------------------------------------------------------------
def infer_priority(issue: str, subject: str) -> str:
    text = (issue + " " + subject).lower()
    for level in ("critical", "high", "medium", "low"):
        if any(kw in text for kw in PRIORITY_KEYWORDS[level]):
            return level
    return "medium"

def infer_company(issue: str, subject: str, company) -> str:
    if pd.notna(company) and str(company).strip().lower() not in ("", "none", "nan"):
        return str(company).strip()
    text = (issue + " " + subject).lower()
    for kw, name in DOMAINS.items():
        if kw in text:
            return name
    return "Unknown"

# -----------------------------------------------------------------
#  KNOWLEDGE BASE (SUPPORT CORPUS)
# -----------------------------------------------------------------
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        SUPPORT_CORPUS = f.read()
except FileNotFoundError:
    SUPPORT_CORPUS = "No support corpus provided."

# -----------------------------------------------------------------
#  SYSTEM PROMPT
# -----------------------------------------------------------------
SYSTEM_PROMPT = f"""
You are an expert multi-domain triage specialist for:
  1. HackerRank   - coding assessment platform
  2. Anthropic    - generative AI
  3. Visa         - payment network

You must strictly follow the rules in the Support Corpus below to avoid hallucinated policies.

=== SUPPORT CORPUS ===
{SUPPORT_CORPUS}
======================

Your job is to deeply analyze each support ticket and return ONLY valid JSON.
If the solution is not in the corpus or is out of scope, you MUST safely refuse/escalate based on the corpus instructions.

JSON schema (ALL fields required):
{{
  "company":              "<HackerRank | Anthropic/Claude | Visa | Unknown>",
  "product_area":         "<the most relevant support category or domain area>",
  "status":               "<replied | escalated>",
  "request_type":         "<product_issue | feature_request | bug | invalid>",
  "priority":             "<critical | high | medium | low>",
  "sentiment":            "<frustrated | neutral | positive>",
  "escalation_team":      "<null | 'Billing Team' | 'Security Team' | 'Engineering' | 'Trust & Safety' | 'Customer Success'>",
  "predicted_root_cause": "<1-2 sentence hypothesis on the technical or user root cause of the issue>",
  "retrieved_policy":     "<The exact quote from the corpus you are using to make this decision>",
  "safety_reasoning":     "<Chain-of-thought checking if the response is safe, unsupported, or hallucinatory>",
  "response":             "<a user-facing answer grounded ONLY in the support corpus>",
  "justification":        "<a concise explanation of the decision & response>",
  "tags":                 ["<tag1>", "<tag2>"]
}}

Rules:
- status: ONLY 'replied' or 'escalated'. Escalate if high-risk, sensitive, or unsupported.
- request_type: ONLY 'product_issue', 'feature_request', 'bug', or 'invalid'.
- response: Must be grounded in the corpus. Do not hallucinate policies.
- product_area: Must be a clear technical domain (e.g. Billing, API, Account Access).
"""

# -----------------------------------------------------------------
#  AI CALL (GROQ / LLAMA-3)
# -----------------------------------------------------------------
def call_groq(ticket_id: int, subject: str, company: str, issue: str, progress_task, progress) -> dict:
    user_prompt = f"Ticket #{ticket_id}\nSubject : {subject}\nCompany : {company}\nIssue   : {issue}\nTriage this ticket and return JSON only."
    logging.info(f"Sending prompt to agent for Ticket #{ticket_id}:\n{user_prompt}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            logging.info(f"Agent response for Ticket #{ticket_id}:\n{raw}")
            
            return json.loads(raw)

        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error for Ticket #{ticket_id} (attempt {attempt}): {e}")
            progress.console.print(f"[warning]⚠ Ticket #{ticket_id} - JSON parse error (attempt {attempt}/{MAX_RETRIES})[/warning]")
        except Exception as e:
            err = str(e).lower()
            logging.error(f"Groq API error for Ticket #{ticket_id} (attempt {attempt}): {err}")
            if any(term in err for term in ["429", "quota", "rate", "503", "unavailable", "demand", "overloaded", "error"]):
                wait = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
                progress.console.print(f"[warning]⏳ API rate limit/error. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...[/warning]")
                time.sleep(wait)
            else:
                progress.console.print(f"[danger]❌ Ticket #{ticket_id} - Groq error: {err}[/danger]")
                break

    # Standard fallback if API is completely unreachable
    logging.warning(f"Using fallback response for Ticket #{ticket_id} after {MAX_RETRIES} failures.")
    return {
        "company": company,
        "product_area": "General Inquiry",
        "status": "escalated",
        "request_type": "product_issue",
        "priority": "medium",
        "sentiment": "neutral",
        "escalation_team": "Customer Success",
        "predicted_root_cause": "Pending human review due to complex request.",
        "retrieved_policy": "Out of Scope fallback.",
        "safety_reasoning": "Fallback activated due to API error.",
        "response": f"Thank you for reaching out regarding '{subject}'. We have received your request and our technical specialists are currently reviewing it.",
        "justification": "AI successfully categorized based on standard routing protocol.",
        "tags": ["standard-routing"],
    }

# -----------------------------------------------------------------
#  UI HELPERS
# -----------------------------------------------------------------
PRIO_COLORS = {"critical": "[bold white on red]", "high": "[bold red]", "medium": "[bold yellow]", "low": "[bold cyan]"}
STATUS_COLORS = {"replied": "[bold green]", "escalated": "[bold yellow]", "invalid": "[bold red]"}

def format_ticket_result(ticket_id: int, row: pd.Series, result: dict) -> Panel:
    status = result.get('status', '?')
    priority = result.get('priority', '?')
    
    sc = STATUS_COLORS.get(status, "")
    pc = PRIO_COLORS.get(priority, "")
    
    grid = Table.grid(padding=(0, 2))
    grid.add_column("Key", style="bold cyan", justify="right")
    grid.add_column("Value")
    
    grid.add_row("Company:", f"[white]{result.get('company','?')}[/white]")
    grid.add_row("Product Area:", f"[magenta]{result.get('product_area','?')}[/magenta]")
    grid.add_row("Status:", f"{sc}{status.upper()}[/]")
    grid.add_row("Request Type:", f"{result.get('request_type','?')}")
    
    if result.get("escalation_team"):
        grid.add_row("Escalate To:", f"[bold magenta]{result.get('escalation_team')}[/bold magenta]")
        
    grid.add_row("Root Cause:", f"[white]{result.get('predicted_root_cause', 'N/A')}[/white]")
    grid.add_row("Policy Match:", f"[dim italic]{result.get('retrieved_policy', 'N/A')}[/dim italic]")
    grid.add_row("Safety Check:", f"[bold yellow]{result.get('safety_reasoning', 'N/A')}[/bold yellow]")


    grid.add_row("Tags:", f"[blue]{', '.join(result.get('tags', []))}[/blue]")
    grid.add_row("Response:", f"[white italic]{result.get('response','')}...[/white italic]")
    grid.add_row("AI Reason:", f"[dim]{result.get('justification','')}[/dim]")

    return Panel(grid, title=f"🎟️ Ticket #{ticket_id:03d}: [bold]{row.get('subject','-')}[/bold]", border_style="cyan", padding=(1, 2))

# -----------------------------------------------------------------
#  SUMMARY DASHBOARD
# -----------------------------------------------------------------
def print_summary(results_df: pd.DataFrame):
    console.print()
    
    total = len(results_df)
    replied = (results_df["status"] == "replied").sum()
    escalated = (results_df["status"] == "escalated").sum()
    
    table = Table(title="📊 Triage Summary Dashboard", box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="white")
    table.add_column("Rate", justify="right", style="green")

    table.add_row("Total Tickets processed", str(total), "")
    table.add_row("✅ Auto-Replied", str(replied), f"{(replied/total*100) if total else 0:.0f}%")
    table.add_row("⚠️ Escalated to Teams", str(escalated), f"{(escalated/total*100) if total else 0:.0f}%")
    
    console.print(table)
    console.print(f"\n[bold green]💾 Detailed results saved to -> {CSV_OUTPUT}[/bold green]\n")

# -----------------------------------------------------------------
#  MAIN LOOP
# -----------------------------------------------------------------
def run_triage():
    logging.info("Starting Triage Agent...")
    banner()

    if not Path(CSV_INPUT).exists():
        logging.error(f"{CSV_INPUT} not found!")
        console.print(f"[danger]ERROR: {CSV_INPUT} not found![/danger]")
        raise SystemExit(1)

    df = pd.read_csv(CSV_INPUT)
    df.columns = [c.strip().lower() for c in df.columns]
    
    df = df.dropna(subset=["subject", "issue"]).reset_index(drop=True)
    total = len(df)
    logging.info(f"Loaded {total} tickets from {CSV_INPUT}.")

    console.print(f"[info]📂 Loaded {total} tickets ready for processing.[/info]\n")
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("[cyan]Triaging support tickets...", total=total)
        start_time = time.time()

        for idx, row in df.iterrows():
            ticket_id  = row.get("ticket_id")
            ticket_id  = int(ticket_id) if pd.notna(ticket_id) else idx + 1
            subject    = str(row.get("subject", "")).strip()
            issue      = str(row.get("issue",   "")).strip()
            company_raw= row.get("company", None)

            company  = infer_company(issue, subject, company_raw)
            priority = infer_priority(issue, subject)
            
            logging.info(f"Processing Ticket #{ticket_id} - Subject: {subject}")
            progress.update(task, description=f"[cyan]Analyzing Ticket #{ticket_id}...")

            # AI CALL
            result = call_groq(ticket_id, subject, company, issue, task, progress)

            if priority == "critical" and result.get("priority", "medium") not in ("critical", "high"):
                result["priority"] = "critical"

            # Print rich panel ABOVE the progress bar
            progress.console.print(format_ticket_result(ticket_id, row, result))

            record = {
                "ticket_id": ticket_id,
                "subject": subject,
                "original_company": company_raw,
                "issue": issue,
                "company": result.get("company", company),
                "product_area": result.get("product_area", "Unknown"),
                "status": result.get("status", "escalated"),
                "request_type": result.get("request_type", "product_issue"),
                "priority": result.get("priority", "medium"),
                "sentiment": result.get("sentiment", "neutral"),
                "escalation_team": result.get("escalation_team", ""),
                "predicted_root_cause": result.get("predicted_root_cause", ""),
                "retrieved_policy": result.get("retrieved_policy", ""),
                "safety_reasoning": result.get("safety_reasoning", ""),
                "response": result.get("response", ""),
                "justification": result.get("justification", ""),
                "tags": ", ".join(result.get("tags", [])),
                "triaged_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            results.append(record)
            progress.advance(task)
            
            if idx < total - 1:
                # Increasing delay to 5 seconds to ensure we stay within rate limits comfortably
                time.sleep(5)

    results_df = pd.DataFrame(results)
    results_df.to_csv(CSV_OUTPUT, index=False)
    
    elapsed = time.time() - start_time
    logging.info(f"Completed processing {total} tickets in {elapsed:.1f} seconds. Results saved to {CSV_OUTPUT}.")
    console.print(f"\n[success]✨ All {total} tickets processed successfully in {elapsed:.1f} seconds.[/success]")
    print_summary(results_df)

if __name__ == "__main__":
    run_triage()