"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path
from exercises.exercise_4_audit import (
    node_fetch_pr, node_analyze, node_route,
    node_auto_approve, node_human_approval,
    node_commit, node_escalate, node_synthesize,
    audit, AGENT_ID
)
from langgraph.graph import START, END, StateGraph
from common.schemas import ReviewState, PRAnalysis, risk_level_for, AuditEntry
from common.llm import get_llm
from common.db import db_conn
import time


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    async def fetch_recent_sessions():
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id, pr_url, 
                       MIN(timestamp) AS started, MAX(timestamp) AS last_event,
                       MAX(risk_level) AS worst_risk, COUNT(*) AS events
                FROM audit_events
                GROUP BY thread_id, pr_url
                ORDER BY MAX(timestamp) DESC LIMIT 25
                """
            ) as cur:
                return await cur.fetchall()
                
    recent_sessions = asyncio.run(fetch_recent_sessions())
    for r in recent_sessions:
        if st.button(f"{r['thread_id'][:8]} - {r['worst_risk']} ({r['events']} events)", key=f"btn_{r['thread_id']}"):
            st.session_state.thread_id = r['thread_id']
            st.session_state.pr_url = r['pr_url']
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.rerun()
            
    st.header("Metrics (Bonus 2)")
    async def fetch_metrics():
        async with db_conn() as conn:
            async with conn.execute(
                "SELECT AVG(confidence) as avg_conf, COUNT(*) as approvals FROM audit_events WHERE decision = 'approve'"
            ) as cur:
                return await cur.fetchone()
    
    metrics = asyncio.run(fetch_metrics())
    if metrics:
        st.metric("Avg Confidence of Approved", f"{metrics['avg_conf'] or 0:.2%}")
        st.metric("Total Approvals", metrics['approvals'] or 0)

    st.header("Time Travel (Bonus 1)")
    if st.session_state.thread_id:
        async def fetch_history():
            async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
                cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
                history = [s async for s in cp.aget_state_history(cfg)]
                return history
        
        try:
            history = asyncio.run(fetch_history())
            if history:
                options = {s.config['configurable']['checkpoint_id']: f"Checkpoint at step {len(history)-i}" for i, s in enumerate(history)}
                selected_checkpoint = st.selectbox("Select checkpoint to view/resume", options=list(options.keys()), format_func=lambda x: options[x])
                if st.button("Resume from checkpoint"):
                    st.write(f"Checkpoint {selected_checkpoint} selected. (Graph resume logic can be tied to this config)")
        except Exception as e:
            st.error("Could not fetch history")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for q in payload.get("questions", []):
            answers[q] = st.text_input(f"Q: {q}", key=f"q_{q}")
        submitted = st.form_submit_button("Submit answers")
        if submitted:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        async def node_auto_edit(state):
            t0 = time.monotonic()
            feedback = state.get("human_feedback")
            llm = get_llm().with_structured_output(PRAnalysis)
            with st.spinner("LLM rewriting review based on human feedback..."):
                refined = await llm.ainvoke([
                    {"role": "system", "content": "Refine review based on human feedback."},
                    {"role": "user", "content": f"Original Analysis Summary: {state['analysis'].summary}\nFeedback: {feedback}"}
                ])
            await audit(state, AuditEntry(
                agent_id=AGENT_ID,
                action="auto_edit",
                confidence=refined.confidence,
                risk_level=risk_level_for(refined.confidence),
                decision="pending",
                reason=f"Rewritten with feedback: {feedback}",
                execution_time_ms=int((time.monotonic() - t0) * 1000),
            ))
            return {"analysis": refined, "human_choice": "approve"}
            
        g = StateGraph(ReviewState)
        for name, fn in [
            ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
            ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
            ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
        ]:
            g.add_node(name, fn)
        g.add_node("auto_edit", node_auto_edit)
        g.add_edge(START, "fetch_pr")
        g.add_edge("fetch_pr", "analyze")
        g.add_edge("analyze", "route")
        g.add_conditional_edges(
            "route", lambda s: s["decision"],
            {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
        )
        g.add_edge("auto_approve", END)
        
        def human_approval_edge(state):
            if state.get("human_choice") == "edit":
                return "auto_edit"
            return "commit"
        g.add_conditional_edges("human_approval", human_approval_edge, {"auto_edit": "auto_edit", "commit": "commit"})
        
        g.add_edge("auto_edit", "commit")
        g.add_edge("commit", END)
        g.add_edge("escalate", "synthesize")
        g.add_edge("synthesize", "commit")
        
        app = g.compile(checkpointer=cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
