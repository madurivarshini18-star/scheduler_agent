import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.seed import seed
        seed()
    except Exception as e:
        print(f"Seed skipped: {e}")
    yield


from app.agent import run_agent
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

app = FastAPI(title="Schedule RAG Agent", version="2.0", lifespan=lifespan)

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentInput(BaseModel):
    input: str


class AgentOutput(BaseModel):
    output: str


@app.post("/agent/invoke", response_model=AgentOutput)
def agent_invoke(payload: AgentInput) -> AgentOutput:
    try:
        return AgentOutput(output=run_agent(payload.input))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def list_events(category: str = "all"):
    from pinecone import Pinecone
    from app.tools import _embed

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    today = datetime.now().date()
    end_date = today + timedelta(days=30)
    today_str = today.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        result = index.query(
            vector=_embed("schedule events"),
            top_k=100,
            filter={"date": {"$gte": today_str, "$lte": end_str}},
            include_metadata=True,
        )
        matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
    except Exception:
        result = index.query(
            vector=_embed("schedule events"),
            top_k=100,
            include_metadata=True,
        )
        matches = result.matches if hasattr(result, "matches") else result.get("matches", [])

    events = []
    seen = set()
    for m in matches:
        mid = m.id if hasattr(m, "id") else m.get("id", "")
        if mid in seen:
            continue
        seen.add(mid)
        meta = m.metadata if hasattr(m, "metadata") else m.get("metadata", {})
        cat = meta.get("event_type", "personal")
        if category != "all" and cat != category:
            continue
        events.append({
            "id": mid,
            "date": meta.get("date", ""),
            "time": meta.get("time", ""),
            "title": meta.get("title", meta.get("purpose", "")),
            "category": cat,
            "description": meta.get("description", ""),
        })

    events.sort(key=lambda x: (x["date"], x["time"]))
    return JSONResponse(content={"events": events})


@app.delete("/events/{event_id}")
def delete_event(event_id: str):
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(ids=[event_id])
        return {"status": "deleted", "id": event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_HTML_PART1 = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Schedule Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0f1e;
  --surface:#0f1629;
  --surface2:#151e38;
  --surface3:#1c2848;
  --border:#1e2d50;
  --accent:#38bdf8;
  --accent2:#0ea5e9;
  --accent-light:#7dd3fc;
  --accent-glow:rgba(56,189,248,0.2);
  --pink:#f472b6;
  --cyan:#34d399;
  --text:#e2eeff;
  --text-muted:#5a7aaa;
  --text-dim:#243660;
  --danger:#f87171;
  --success:#34d399;
  --warning:#fbbf24;
  --radius:16px;
  --radius-sm:10px;
  --shadow:0 8px 40px rgba(0,0,0,0.8);
  --shadow-sm:0 2px 16px rgba(0,0,0,0.6);
  --transition:.18s cubic-bezier(.4,0,.2,1);
}
body{font-family:'Inter',sans-serif;background:var(--bg);min-height:100vh;color:var(--text);overflow:hidden}
.app{display:grid;grid-template-columns:268px 1fr;height:100vh}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:var(--accent)}

/* ── SIDEBAR ── */
.sidebar{
  background:#070c1a;
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow-y:auto;overflow-x:hidden;
}
.logo-wrap{
  padding:24px 20px 20px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:12px;
}
.logo-icon{
  width:40px;height:40px;border-radius:12px;
  background:linear-gradient(135deg,var(--accent),var(--cyan));
  display:flex;align-items:center;justify-content:center;
  font-size:1.2rem;flex-shrink:0;
  box-shadow:0 0 18px var(--accent-glow);
}
.logo-text{font-size:.95rem;font-weight:700;color:var(--text);line-height:1.2}
.logo-sub{font-size:.68rem;color:var(--text-muted);font-weight:400;margin-top:1px}

.ns{
  padding:20px 16px 6px;
  font-size:.62rem;font-weight:700;color:var(--text-dim);
  letter-spacing:.1em;text-transform:uppercase;
}
.nav-btn{
  display:flex;align-items:center;gap:10px;
  width:calc(100% - 16px);margin:2px 8px;
  padding:10px 14px;background:none;border:none;
  color:var(--text-muted);font-size:.83rem;font-weight:500;
  cursor:pointer;border-radius:var(--radius-sm);text-align:left;
  transition:var(--transition);font-family:inherit;
}
.nav-btn:hover{background:var(--surface2);color:var(--text)}
.nav-btn.active{
  background:rgba(56,189,248,0.1);
  color:var(--accent);
  box-shadow:inset 0 0 0 1px rgba(56,189,248,0.28);
}
.nav-icon{font-size:1rem;width:20px;text-align:center}

.cat-btn{
  display:flex;align-items:center;gap:9px;
  margin:1px 8px;padding:7px 14px;
  background:none;border:none;color:var(--text-muted);font-size:.78rem;font-weight:400;
  cursor:pointer;border-radius:8px;text-align:left;
  transition:var(--transition);width:calc(100% - 16px);font-family:inherit;
}
.cat-btn:hover{background:var(--surface2);color:var(--text)}
.cat-btn.active{color:var(--accent);background:rgba(56,189,248,0.1)}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}

/* ── MAIN ── */
.main{display:flex;flex-direction:column;overflow:hidden;background:var(--bg)}

/* ── TOPBAR ── */
.topbar{
  background:rgba(7,12,26,0.96);
  backdrop-filter:blur(12px);
  padding:0 28px;
  height:64px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;gap:16px;
  flex-shrink:0;
}
.topbar-left{display:flex;align-items:center;gap:14px}
.topbar h1{font-size:1rem;font-weight:700;color:var(--text);letter-spacing:-.01em}
.view-badge{
  font-size:.68rem;font-weight:600;padding:3px 9px;border-radius:99px;
  background:rgba(56,189,248,0.12);color:var(--accent);
  letter-spacing:.03em;border:1px solid rgba(56,189,248,0.28);
}
.topbar-right{display:flex;align-items:center;gap:10px}
.today-pill{
  font-size:.75rem;color:var(--text-muted);font-weight:500;
  background:var(--surface2);padding:5px 12px;border-radius:99px;
  border:1px solid var(--border);
}
.count-pill{
  font-size:.72rem;font-weight:600;padding:4px 10px;border-radius:99px;
  background:rgba(56,189,248,0.12);color:var(--accent);
  border:1px solid rgba(56,189,248,0.28);
}

.content{padding:24px 28px;overflow-y:auto;flex:1}

/* ── EVENT CARDS ── */
.date-group{margin-bottom:24px}
.group-header{
  display:flex;align-items:center;gap:10px;margin-bottom:10px;
}
.group-date{font-size:.72rem;font-weight:700;color:var(--text-muted);letter-spacing:.06em;text-transform:uppercase}
.group-line{flex:1;height:1px;background:var(--border)}
.group-count{font-size:.65rem;color:var(--text-dim);font-weight:600}

.evt-list{display:flex;flex-direction:column;gap:6px}
.evt-card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:14px 16px;
  display:flex;align-items:center;gap:14px;
  transition:var(--transition);
  cursor:default;
  position:relative;
  overflow:hidden;
}
.evt-card::before{
  content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--card-color,var(--accent));border-radius:0 2px 2px 0;
}
.evt-card:hover{
  background:var(--surface2);
  border-color:#1e3a6e;
  transform:translateY(-1px);
  box-shadow:0 4px 20px rgba(0,0,0,0.6);
}
.evt-time-col{
  display:flex;flex-direction:column;align-items:center;
  min-width:46px;flex-shrink:0;
}
.evt-time{font-size:.78rem;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums}
.evt-ampm{font-size:.58rem;color:var(--text-muted);font-weight:500;margin-top:1px}
.evt-divider{width:1px;height:36px;background:var(--border);flex-shrink:0}
.evt-body{flex:1;min-width:0}
.evt-title{font-weight:600;font-size:.9rem;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.01em}
.evt-meta{display:flex;align-items:center;gap:6px;margin-top:3px;flex-wrap:wrap}
.evt-desc{font-size:.73rem;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}
.evt-badge{
  font-size:.63rem;padding:2px 8px;border-radius:99px;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;flex-shrink:0;
  border:1px solid;
}
.del-btn{
  background:none;border:none;color:var(--text-dim);cursor:pointer;
  width:30px;height:30px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;transition:var(--transition);flex-shrink:0;font-size:.85rem;
}
.del-btn:hover{color:var(--danger);background:rgba(248,113,113,0.12)}

.empty{
  text-align:center;padding:72px 0;color:var(--text-muted);font-size:.88rem;
  display:flex;flex-direction:column;align-items:center;gap:10px;
}
.empty-icon{font-size:2.5rem;opacity:.4}
.loading{
  text-align:center;padding:48px 0;color:var(--text-muted);font-size:.85rem;
  display:flex;flex-direction:column;align-items:center;gap:14px;
}
.spinner{
  width:28px;height:28px;border:2.5px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;
  animation:spin .7s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── PANEL FORMS ── */
.panel-wrap{max-width:540px}
.panel{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:20px;
  padding:28px 28px 24px;
  box-shadow:var(--shadow-sm);
}
.panel-header{display:flex;align-items:center;gap:12px;margin-bottom:24px}
.panel-icon{
  width:38px;height:38px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),var(--cyan));
  display:flex;align-items:center;justify-content:center;font-size:1rem;
}
.panel h2{font-size:1rem;font-weight:700;color:var(--text);letter-spacing:-.01em}
.panel h2 small{display:block;font-size:.72rem;font-weight:400;color:var(--text-muted);margin-top:1px}

label{display:block;font-size:.73rem;font-weight:600;color:var(--text-muted);margin-bottom:5px;margin-top:16px;letter-spacing:.02em;text-transform:uppercase}
label:first-of-type{margin-top:0}

input,select,textarea{
  width:100%;padding:10px 13px;
  background:var(--surface2);
  border:1.5px solid var(--border);
  border-radius:var(--radius-sm);
  font-size:.87rem;color:var(--text);
  outline:none;font-family:inherit;
  transition:border-color var(--transition),box-shadow var(--transition);
  appearance:none;
}
input::placeholder{color:var(--text-dim)}
input:focus,select:focus,textarea:focus{
  border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-glow);
  background:var(--surface3);
}
select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238b91c0' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:32px}
select option{background:var(--surface2)}
textarea{min-height:72px;resize:vertical}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}

.btn-primary{
  margin-top:20px;width:100%;padding:11px;
  background:var(--accent);
  color:#020c1f;border:none;border-radius:var(--radius-sm);
  font-size:.9rem;font-weight:700;cursor:pointer;font-family:inherit;
  transition:var(--transition);
  box-shadow:0 0 20px var(--accent-glow);
  letter-spacing:.01em;
}
.btn-primary:hover:not(:disabled){
  background:var(--accent-light);
  transform:translateY(-1px);box-shadow:0 0 32px var(--accent-glow);
}
.btn-primary:disabled{opacity:.3;cursor:not-allowed;transform:none}

.status-msg{margin-top:10px;font-size:.8rem;min-height:16px;text-align:center;color:var(--text-muted)}
.status-msg.ok{color:var(--success)}
.status-msg.err{color:var(--danger)}

/* ── ASK ── */
.ask-ans{
  margin-top:16px;
  background:var(--surface2);border:1.5px solid var(--border);
  border-radius:var(--radius-sm);padding:16px;
  font-size:.87rem;line-height:1.75;
  display:none;white-space:pre-wrap;color:var(--text);
  position:relative;
}
.ask-ans::before{
  content:'AI';position:absolute;top:-9px;left:14px;
  background:linear-gradient(135deg,var(--accent),var(--cyan));
  color:#020c1f;font-size:.6rem;font-weight:700;padding:1px 7px;border-radius:99px;
  letter-spacing:.05em;
}
.ask-err{margin-top:6px;font-size:.78rem;color:var(--danger);min-height:14px}
.shortcut-tip{font-size:.7rem;color:var(--text-dim);margin-top:7px;text-align:right}

/* ── WELCOME ── */
.welcome{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:80px 20px;text-align:center;gap:16px;
}
.welcome-icon{font-size:3.5rem;filter:drop-shadow(0 0 24px var(--accent-glow))}
.welcome h2{font-size:1.15rem;font-weight:700;color:var(--text);letter-spacing:-.01em}
.welcome p{font-size:.85rem;color:var(--text-muted);max-width:320px;line-height:1.7}
.load-btn{
  margin-top:8px;padding:10px 28px;
  background:var(--accent);color:#020c1f;
  border:none;border-radius:var(--radius-sm);
  font-size:.88rem;font-weight:700;cursor:pointer;font-family:inherit;
  transition:var(--transition);box-shadow:0 0 20px var(--accent-glow);
}
.load-btn:hover{background:var(--accent-light);transform:translateY(-1px);box-shadow:0 0 32px var(--accent-glow)}
</style>
</head>
<body>
<div class="app">

<div class="sidebar">
  <div class="logo-wrap">
    <div class="logo-icon">&#128197;</div>
    <div>
      <div class="logo-text">Schedule Assistant</div>
      <div class="logo-sub">RAG-powered planner</div>
    </div>
  </div>

  <div class="ns">Navigation</div>
  <button class="nav-btn active" onclick="showView('all')" id="nav-all">
    <span class="nav-icon">&#128197;</span>All Events
  </button>
  <button class="nav-btn" onclick="showView('add')" id="nav-add">
    <span class="nav-icon">&#43;&#65039;</span>Add / Manage
  </button>
  <button class="nav-btn" onclick="showView('ask')" id="nav-ask">
    <span class="nav-icon">&#128172;</span>Ask AI
  </button>

  <div class="ns" style="margin-top:8px">Categories</div>
  <button class="cat-btn active" onclick="filterCat('all')" id="cat-all"><span class="dot" style="background:#7b6cf6"></span>All</button>
  <button class="cat-btn" onclick="filterCat('education')" id="cat-education"><span class="dot" style="background:#0ea5e9"></span>Education</button>
  <button class="cat-btn" onclick="filterCat('work')" id="cat-work"><span class="dot" style="background:#818cf8"></span>Work</button>
  <button class="cat-btn" onclick="filterCat('health')" id="cat-health"><span class="dot" style="background:#22c55e"></span>Health</button>
  <button class="cat-btn" onclick="filterCat('family')" id="cat-family"><span class="dot" style="background:#f59e0b"></span>Family</button>
  <button class="cat-btn" onclick="filterCat('social')" id="cat-social"><span class="dot" style="background:#ec4899"></span>Social</button>
  <button class="cat-btn" onclick="filterCat('personal')" id="cat-personal"><span class="dot" style="background:#a78bfa"></span>Personal</button>
  <button class="cat-btn" onclick="filterCat('shopping')" id="cat-shopping"><span class="dot" style="background:#f97316"></span>Shopping</button>
  <button class="cat-btn" onclick="filterCat('entertainment')" id="cat-entertainment"><span class="dot" style="background:#c084fc"></span>Entertainment</button>
  <button class="cat-btn" onclick="filterCat('travel')" id="cat-travel"><span class="dot" style="background:#f43f5e"></span>Travel</button>
  <button class="cat-btn" onclick="filterCat('finance')" id="cat-finance"><span class="dot" style="background:#14b8a6"></span>Finance</button>
  <button class="cat-btn" onclick="filterCat('home')" id="cat-home"><span class="dot" style="background:#84cc16"></span>Home</button>
  <button class="cat-btn" onclick="filterCat('deadline')" id="cat-deadline"><span class="dot" style="background:#ef4444"></span>Deadlines</button>
  <button class="cat-btn" onclick="filterCat('recurring')" id="cat-recurring"><span class="dot" style="background:#06b6d4"></span>Recurring</button>
  <button class="cat-btn" onclick="filterCat('appointment')" id="cat-appointment"><span class="dot" style="background:#d946ef"></span>Appointments</button>
  <button class="cat-btn" onclick="filterCat('emergency')" id="cat-emergency"><span class="dot" style="background:#dc2626"></span>Emergencies</button>
  <button class="cat-btn" onclick="filterCat('meeting')" id="cat-meeting"><span class="dot" style="background:#3b82f6"></span>Meetings</button>
  <button class="cat-btn" onclick="filterCat('task')" id="cat-task"><span class="dot" style="background:#fbbf24"></span>Tasks</button>
  <button class="cat-btn" onclick="filterCat('workshop')" id="cat-workshop"><span class="dot" style="background:#10b981"></span>Workshops</button>
  <button class="cat-btn" onclick="filterCat('other')" id="cat-other"><span class="dot" style="background:#94a3b8"></span>Other</button>
</div>

<div class="main">
  <div class="topbar">
    <div class="topbar-left">
      <h1 id="view-title">All Events</h1>
      <span class="view-badge" id="view-badge">Schedule</span>
    </div>
    <div class="topbar-right">
      <span class="count-pill" id="count-pill" style="display:none"></span>
      <span class="today-pill" id="today-label"></span>
    </div>
  </div>
  <div class="content">

    <div id="view-all">
      <div id="events-out">
        <div class="welcome">
          <div class="welcome-icon">&#128197;</div>
          <h2>Your Schedule</h2>
          <p>Events you add from now on will appear here. Click below to load your upcoming events.</p>
          <button class="load-btn" onclick="loadEvents()">Load My Events</button>
        </div>
      </div>
    </div>

    <div id="view-add" style="display:none">
      <div class="panel-wrap">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-icon">&#9998;</div>
            <div>
              <h2>Manage Event<small>Add, update, or remove a scheduled item</small></h2>
            </div>
          </div>
          <label>Action</label>
          <select id="f-action">
            <option value="add">&#43; Add new event</option>
            <option value="update">&#9998; Update existing event</option>
            <option value="delete">&#128465; Delete event</option>
          </select>
          <div class="row2">
            <div><label>Date</label><input type="date" id="f-date"/></div>
            <div><label>Time</label><input type="time" id="f-time" value="09:00"/></div>
          </div>
          <label>Title / Purpose</label>
          <input type="text" id="f-title" placeholder="e.g. Doctor visit, Team meeting..."/>
          <label>Category</label>
          <select id="f-type">
            <option value="education">Education</option>
            <option value="work">Work</option>
            <option value="health">Health</option>
            <option value="family">Family</option>
            <option value="social">Social</option>
            <option value="personal">Personal</option>
            <option value="shopping">Shopping &amp; Errands</option>
            <option value="entertainment">Entertainment</option>
            <option value="travel">Travel</option>
            <option value="finance">Finance</option>
            <option value="home">Home</option>
            <option value="deadline">Deadline</option>
            <option value="recurring">Recurring Event</option>
            <option value="appointment">Appointment</option>
            <option value="emergency">Emergency</option>
            <option value="meeting">Meeting</option>
            <option value="task">Task</option>
            <option value="workshop">Workshop</option>
            <option value="other">Other</option>
          </select>
          <label>Description <span style="font-weight:400;text-transform:none;opacity:.6">(optional)</span></label>
          <input type="text" id="f-desc" placeholder="Extra details..."/>
          <button class="btn-primary" id="save-btn" onclick="saveEvent()">Save Event</button>
          <div class="status-msg" id="save-st"></div>
        </div>
      </div>
    </div>

    <div id="view-ask" style="display:none">
      <div class="panel-wrap">
        <div class="panel" style="max-width:600px">
          <div class="panel-header">
            <div class="panel-icon">&#128172;</div>
            <div>
              <h2>Ask the AI<small>Ask anything about your schedule in plain language</small></h2>
            </div>
          </div>
          <label>Your question</label>
          <textarea id="q-in" rows="3" placeholder="e.g. What do I have tomorrow?&#10;Am I free Friday afternoon?&#10;Move my 2 PM meeting to 4 PM"></textarea>
          <div class="shortcut-tip">Tip: Press Ctrl+Enter to send</div>
          <button class="btn-primary" id="ask-btn" onclick="askAgent()">Ask AI</button>
          <div class="ask-ans" id="ask-ans"></div>
          <div class="ask-err" id="ask-err"></div>
        </div>
      </div>
    </div>

  </div>
</div>
</div>
"""

_HTML_PART2 = """
<script>
const COLORS = {
  education:'#0ea5e9', work:'#818cf8', health:'#22c55e', family:'#f59e0b',
  social:'#ec4899', personal:'#a78bfa', shopping:'#f97316', entertainment:'#c084fc',
  travel:'#f43f5e', finance:'#14b8a6', home:'#84cc16', deadline:'#ef4444',
  recurring:'#06b6d4', appointment:'#d946ef', emergency:'#dc2626',
  meeting:'#3b82f6', task:'#fbbf24', workshop:'#10b981', other:'#94a3b8',
};

let allEvents = [];
let currentCat = 'all';

const today = new Date();
const todayStr = today.toISOString().split('T')[0];
const tomorrowStr = new Date(today.getTime()+86400000).toISOString().split('T')[0];

document.getElementById('today-label').textContent =
  today.toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric',year:'numeric'});
document.getElementById('f-date').value = todayStr;

// ── VIEWS ──
let eventsLoaded = false;
function showView(name) {
  ['all','add','ask'].forEach(v => {
    document.getElementById('view-'+v).style.display = v===name ? '' : 'none';
    document.getElementById('nav-'+v).classList.toggle('active', v===name);
  });
  const titles = {all:'My Schedule', add:'Manage Event', ask:'Ask AI'};
  const badges = {all:'Schedule', add:'Event Manager', ask:'AI Assistant'};
  document.getElementById('view-title').textContent = titles[name];
  document.getElementById('view-badge').textContent = badges[name];
  document.getElementById('count-pill').style.display = (name==='all' && eventsLoaded) ? '' : 'none';
}

// ── CATEGORY FILTER ──
function filterCat(cat) {
  currentCat = cat;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('cat-'+cat).classList.add('active');
  showView('all');
  if (eventsLoaded) loadEvents();
}

// ── LOAD EVENTS ──
async function loadEvents() {
  eventsLoaded = true;
  document.getElementById('count-pill').style.display = '';
  document.getElementById('events-out').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading your schedule...</div>';
  try {
    const url = currentCat === 'all' ? '/events' : '/events?category='+currentCat;
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to load');
    allEvents = data.events || [];
    renderEvents();
  } catch(e) {
    document.getElementById('events-out').innerHTML =
      '<div class="empty"><div class="empty-icon">&#128197;</div>Could not load events: '+e.message+'</div>';
  }
}

function fmtTime(t) {
  if (!t) return {time:'—', ampm:''};
  const [h, m] = t.split(':').map(Number);
  const ampm = h >= 12 ? 'PM' : 'AM';
  const h12 = h % 12 || 12;
  return {time: h12 + ':' + String(m).padStart(2,'0'), ampm};
}

function fmtDate(ds) {
  const d = new Date(ds+'T12:00:00');
  const fmt = d.toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric',year:'numeric'});
  if (ds === todayStr) return '&#128205; Today';
  if (ds === tomorrowStr) return '&#128198; Tomorrow';
  return fmt;
}

function renderEvents() {
  const out = document.getElementById('events-out');
  const cp = document.getElementById('count-pill');
  if (!allEvents.length) {
    cp.style.display='none';
    out.innerHTML = '<div class="empty"><div class="empty-icon">&#128197;</div>'
      + 'No events found'+(currentCat!=='all' ? ' in this category' : '')+'.</div>';
    return;
  }
  cp.style.display='';
  cp.textContent = allEvents.length + (allEvents.length===1?' event':' events');

  const grouped = {};
  allEvents.forEach(e => { if (!grouped[e.date]) grouped[e.date]=[]; grouped[e.date].push(e); });

  let html = '';
  Object.keys(grouped).sort().forEach(date => {
    const grp = grouped[date];
    const dateLabel = fmtDate(date);
    html += `<div class="date-group">
      <div class="group-header">
        <span class="group-date">${dateLabel}</span>
        <div class="group-line"></div>
        <span class="group-count">${grp.length}</span>
      </div>
      <div class="evt-list">`;

    grp.forEach(e => {
      const cat = e.category || 'personal';
      const c = COLORS[cat] || '#7b6cf6';
      const alpha = c + '22';
      const label = cat.charAt(0).toUpperCase() + cat.slice(1).replace('_',' ');
      const {time, ampm} = fmtTime(e.time);
      const safeTitle = (e.title||'(no title)').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      const safeDesc = (e.description||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      const safeTitleJs = (e.title||'').replace(/\\\\/g,'\\\\\\\\').replace(/'/g,"\\\\'");
      html += `<div class="evt-card" style="--card-color:${c}">
        <div class="evt-time-col">
          <div class="evt-time">${time}</div>
          <div class="evt-ampm">${ampm}</div>
        </div>
        <div class="evt-divider"></div>
        <div class="evt-body">
          <div class="evt-title">${safeTitle}</div>
          <div class="evt-meta">
            ${safeDesc ? `<span class="evt-desc">${safeDesc}</span>` : ''}
          </div>
        </div>
        <span class="evt-badge" style="background:${alpha};color:${c};border-color:${c}44">${label}</span>
        <button class="del-btn" title="Delete event" onclick="delEvent('${e.id}','${safeTitleJs}')">&#128465;</button>
      </div>`;
    });
    html += '</div></div>';
  });
  out.innerHTML = html;
}

// ── DELETE ──
async function delEvent(id, title) {
  if (!confirm('Delete "'+title+'"?')) return;
  try {
    const res = await fetch('/events/'+encodeURIComponent(id), {method:'DELETE'});
    if (!res.ok) { const d=await res.json(); throw new Error(d.detail); }
    allEvents = allEvents.filter(e => e.id !== id);
    renderEvents();
  } catch(e) { alert('Delete failed: '+e.message); }
}

// ── SAVE EVENT ──
async function saveEvent() {
  const action = document.getElementById('f-action').value;
  const date   = document.getElementById('f-date').value;
  const time   = document.getElementById('f-time').value || '09:00';
  const title  = document.getElementById('f-title').value.trim();
  const type   = document.getElementById('f-type').value;
  const desc   = document.getElementById('f-desc').value.trim();
  const btn    = document.getElementById('save-btn');
  const st     = document.getElementById('save-st');

  if (!date || !title) {
    st.className='status-msg err'; st.textContent='Date and title are required.'; return;
  }
  btn.disabled=true; st.className='status-msg'; st.textContent='Saving...';

  const msg = action==='delete'
    ? `Delete the event titled "${title}" on ${date}`
    : action==='update'
    ? `Update event "${title}" on ${date}: change time to ${time}, category ${type}. ${desc}`
    : `Add a ${type} event titled "${title}" on ${date} at ${time}. ${desc}`;

  try {
    const res = await fetch('/agent/invoke',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({input:msg})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    st.className='status-msg ok';
    st.textContent = action==='delete' ? '&#10003; Deleted.' : action==='update' ? '&#10003; Updated.' : '&#10003; Event saved!';
    if (action==='add') {
      document.getElementById('f-title').value='';
      document.getElementById('f-desc').value='';
    }
  } catch(e) {
    st.className='status-msg err'; st.textContent='Error: '+e.message;
  } finally { btn.disabled=false; }
}

// ── ASK AGENT ──
async function askAgent() {
  const q   = document.getElementById('q-in').value.trim();
  const btn = document.getElementById('ask-btn');
  const ans = document.getElementById('ask-ans');
  const err = document.getElementById('ask-err');
  if (!q) return;
  btn.disabled=true; btn.textContent='Thinking...';
  ans.style.display='none'; err.textContent='';
  try {
    const res = await fetch('/agent/invoke',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({input:q})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    ans.style.display='block'; ans.textContent=data.output;
  } catch(e) { err.textContent='Error: '+e.message; }
  finally { btn.disabled=false; btn.textContent='Ask AI'; }
}

document.getElementById('q-in').addEventListener('keydown', e => {
  if (e.key==='Enter' && e.ctrlKey) askAgent();
});

</script>
</body>
</html>
"""


@app.get("/agent/playground", response_class=HTMLResponse)
def playground():
    return _HTML_PART1 + _HTML_PART2


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
