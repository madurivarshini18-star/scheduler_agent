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
    """Fetch events from Pinecone for the next 30 days, optionally filtered by category."""
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
    """Delete a specific event by ID from Pinecone."""
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(ids=[event_id])
        return {"status": "deleted", "id": event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Schedule Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b1120;
  --surface:#101827;
  --surface2:#162033;
  --surface3:#1c2a42;
  --border:#1e3050;
  --accent:#38bdf8;
  --accent2:#818cf8;
  --accent-glow:rgba(56,189,248,0.18);
  --green:#34d399;
  --rose:#fb7185;
  --amber:#fbbf24;
  --text:#e8f0fe;
  --text-muted:#64748b;
  --text-dim:#283650;
  --danger:#f87171;
  --success:#34d399;
  --radius:14px;
  --radius-sm:9px;
  --shadow:0 8px 32px rgba(0,0,0,0.6);
  --shadow-sm:0 2px 12px rgba(0,0,0,0.4);
  --tr:.17s cubic-bezier(.4,0,.2,1);
}
body{font-family:'DM Sans',sans-serif;background:var(--bg);min-height:100vh;color:var(--text);overflow:hidden}
.app{display:grid;grid-template-columns:256px 1fr;height:100vh}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:var(--accent2)}

/* ── SIDEBAR ── */
.sidebar{
  background:var(--surface);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow-y:auto;overflow-x:hidden;
}
.logo-wrap{
  padding:22px 18px 18px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:11px;
}
.logo-icon{
  width:38px;height:38px;border-radius:11px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  font-size:1.1rem;flex-shrink:0;
  box-shadow:0 0 16px var(--accent-glow);
}
.logo-text{font-size:.9rem;font-weight:700;color:var(--text);line-height:1.2}
.logo-sub{font-size:.65rem;color:var(--text-muted);font-weight:400;margin-top:1px}

.ns{
  padding:18px 14px 5px;
  font-size:.6rem;font-weight:700;color:var(--text-dim);
  letter-spacing:.1em;text-transform:uppercase;
}
.nav-btn{
  display:flex;align-items:center;gap:9px;
  width:calc(100% - 14px);margin:2px 7px;
  padding:9px 13px;background:none;border:none;
  color:var(--text-muted);font-size:.82rem;font-weight:500;
  cursor:pointer;border-radius:var(--radius-sm);text-align:left;
  transition:var(--tr);font-family:inherit;
}
.nav-btn:hover{background:var(--surface2);color:var(--text)}
.nav-btn.active{
  background:rgba(56,189,248,0.1);color:var(--accent);
  box-shadow:inset 0 0 0 1px rgba(56,189,248,0.22);
}
.nav-icon{font-size:.95rem;width:19px;text-align:center;flex-shrink:0}

.cat-btn{
  display:flex;align-items:center;gap:8px;
  margin:1px 7px;padding:6px 13px;
  background:none;border:none;color:var(--text-muted);font-size:.76rem;font-weight:400;
  cursor:pointer;border-radius:8px;text-align:left;
  transition:var(--tr);width:calc(100% - 14px);font-family:inherit;
}
.cat-btn:hover{background:var(--surface2);color:var(--text)}
.cat-btn.active{color:var(--accent);background:rgba(56,189,248,0.08)}
.dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}

/* ── MAIN ── */
.main{display:flex;flex-direction:column;overflow:hidden;background:var(--bg)}

/* ── TOPBAR ── */
.topbar{
  background:rgba(11,17,32,0.95);
  backdrop-filter:blur(14px);
  padding:0 26px;height:60px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;gap:14px;
  flex-shrink:0;
}
.topbar-left{display:flex;align-items:center;gap:12px}
.topbar h1{font-size:.95rem;font-weight:700;color:var(--text);letter-spacing:-.01em}
.view-badge{
  font-size:.65rem;font-weight:600;padding:3px 9px;border-radius:99px;
  background:rgba(56,189,248,0.1);color:var(--accent);
  border:1px solid rgba(56,189,248,0.22);
}
.topbar-right{display:flex;align-items:center;gap:9px}
.today-pill{
  font-size:.72rem;color:var(--text-muted);font-weight:500;
  background:var(--surface);padding:4px 11px;border-radius:99px;
  border:1px solid var(--border);
}
.count-pill{
  font-size:.68rem;font-weight:600;padding:3px 9px;border-radius:99px;
  background:rgba(52,211,153,0.1);color:var(--success);
  border:1px solid rgba(52,211,153,0.2);
}

.content{padding:22px 26px;overflow-y:auto;flex:1}

/* ── DATE GROUPS ── */
.date-group{margin-bottom:22px}
.group-header{display:flex;align-items:center;gap:10px;margin-bottom:9px}
.group-date{font-size:.68rem;font-weight:700;color:var(--text-muted);letter-spacing:.06em;text-transform:uppercase}
.group-line{flex:1;height:1px;background:var(--border)}
.group-count{font-size:.62rem;color:var(--text-dim);font-weight:600}

/* ── EVENT CARDS ── */
.evt-list{display:flex;flex-direction:column;gap:6px}
.evt-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:13px 15px;
  display:flex;align-items:center;gap:13px;
  transition:var(--tr);position:relative;overflow:hidden;
}
.evt-card::before{
  content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--card-color,var(--accent));
}
.evt-card:hover{
  background:var(--surface2);border-color:rgba(56,189,248,0.25);
  transform:translateY(-1px);box-shadow:var(--shadow-sm);
}
.evt-time-col{display:flex;flex-direction:column;align-items:center;min-width:44px;flex-shrink:0}
.evt-time{font-size:.76rem;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums}
.evt-ampm{font-size:.56rem;color:var(--text-muted);font-weight:500;margin-top:1px}
.evt-divider{width:1px;height:34px;background:var(--border);flex-shrink:0}
.evt-body{flex:1;min-width:0}
.evt-title{font-weight:600;font-size:.88rem;color:var(--text);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.evt-desc{font-size:.71rem;color:var(--text-muted);margin-top:2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}
.evt-badge{
  font-size:.6rem;padding:2px 8px;border-radius:99px;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;flex-shrink:0;border:1px solid;
}
.del-btn{
  background:none;border:none;color:var(--text-dim);cursor:pointer;
  width:28px;height:28px;border-radius:7px;display:flex;align-items:center;
  justify-content:center;transition:var(--tr);flex-shrink:0;font-size:.82rem;
}
.del-btn:hover{color:var(--danger);background:rgba(248,113,113,0.1)}

/* ── EMPTY / LOADING ── */
.empty{
  text-align:center;padding:64px 0;color:var(--text-muted);
  display:flex;flex-direction:column;align-items:center;gap:10px;
}
.empty-icon{font-size:2.4rem;opacity:.35}
.loading{
  text-align:center;padding:44px 0;color:var(--text-muted);font-size:.84rem;
  display:flex;flex-direction:column;align-items:center;gap:12px;
}
.spinner{
  width:26px;height:26px;border:2.5px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .65s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── WELCOME ── */
.welcome{
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;padding:72px 20px;text-align:center;gap:14px;
}
.welcome-icon{font-size:3rem;opacity:.5}
.welcome h2{font-size:1.05rem;font-weight:700;color:var(--text)}
.welcome p{font-size:.82rem;color:var(--text-muted);max-width:300px;line-height:1.7}
.load-btn{
  padding:9px 26px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#0b1120;border:none;border-radius:var(--radius-sm);
  font-size:.85rem;font-weight:700;cursor:pointer;font-family:inherit;
  transition:var(--tr);box-shadow:0 0 18px var(--accent-glow);
}
.load-btn:hover{transform:translateY(-1px);box-shadow:0 0 28px var(--accent-glow)}

/* ── PANEL ── */
.panel-wrap{max-width:520px}
.panel{
  background:var(--surface);border:1px solid var(--border);
  border-radius:18px;padding:26px 26px 22px;box-shadow:var(--shadow-sm);
}
.panel-header{display:flex;align-items:center;gap:11px;margin-bottom:22px}
.panel-icon{
  width:36px;height:36px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;font-size:.95rem;
}
.panel h2{font-size:.95rem;font-weight:700;color:var(--text)}
.panel h2 small{display:block;font-size:.7rem;font-weight:400;color:var(--text-muted);margin-top:1px}

label{display:block;font-size:.7rem;font-weight:600;color:var(--text-muted);
  margin-bottom:5px;margin-top:15px;letter-spacing:.04em;text-transform:uppercase}
label:first-of-type{margin-top:0}

input,select,textarea{
  width:100%;padding:9px 12px;
  background:var(--surface2);border:1.5px solid var(--border);
  border-radius:var(--radius-sm);font-size:.85rem;color:var(--text);
  outline:none;font-family:inherit;
  transition:border-color var(--tr),box-shadow var(--tr);appearance:none;
}
input::placeholder{color:var(--text-dim)}
input:focus,select:focus,textarea:focus{
  border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow);
  background:var(--surface3);
}
select{
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='11' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2.5'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;padding-right:32px;
}
select option{background:var(--surface2)}
textarea{min-height:70px;resize:vertical}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:11px}

.btn-primary{
  margin-top:18px;width:100%;padding:10px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#0b1120;border:none;border-radius:var(--radius-sm);
  font-size:.88rem;font-weight:700;cursor:pointer;font-family:inherit;
  transition:var(--tr);box-shadow:0 0 18px var(--accent-glow);
}
.btn-primary:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 0 28px var(--accent-glow)}
.btn-primary:disabled{opacity:.3;cursor:not-allowed;transform:none}

.status-msg{margin-top:9px;font-size:.78rem;min-height:14px;text-align:center;color:var(--text-muted)}
.status-msg.ok{color:var(--success)}.status-msg.err{color:var(--danger)}

/* ── ASK ── */
.ask-ans{
  margin-top:15px;background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:14px;
  font-size:.85rem;line-height:1.75;display:none;
  white-space:pre-wrap;color:var(--text);position:relative;
}
.ask-ans::before{
  content:'AI';position:absolute;top:-9px;left:13px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  color:#0b1120;font-size:.58rem;font-weight:700;
  padding:1px 7px;border-radius:99px;letter-spacing:.05em;
}
.ask-err{margin-top:6px;font-size:.76rem;color:var(--danger);min-height:12px}
.shortcut-tip{font-size:.67rem;color:var(--text-dim);margin-top:6px;text-align:right}
</style>
</head>
<body>
<div class="app">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="logo-wrap">
    <div class="logo-icon">&#128197;</div>
    <div>
      <div class="logo-text">Schedule Assistant</div>
      <div class="logo-sub">Agentic RAG planner</div>
    </div>
  </div>

  <div class="ns">Navigation</div>
  <button class="nav-btn active" onclick="showView('all')" id="nav-all">
    <span class="nav-icon">&#128197;</span>All Events
  </button>
  <button class="nav-btn" onclick="showView('add')" id="nav-add">
    <span class="nav-icon">&#43;</span>Add / Manage
  </button>
  <button class="nav-btn" onclick="showView('ask')" id="nav-ask">
    <span class="nav-icon">&#129302;</span>Ask AI
  </button>

  <div class="ns" style="margin-top:8px">Categories</div>
  <button class="cat-btn active" onclick="filterCat('all')" id="cat-all"><span class="dot" style="background:#38bdf8"></span>All</button>
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

<!-- MAIN -->
<div class="main">
  <div class="topbar">
    <div class="topbar-left">
      <h1 id="view-title">My Schedule</h1>
      <span class="view-badge" id="view-badge">Events</span>
    </div>
    <div class="topbar-right">
      <span class="count-pill" id="count-pill" style="display:none"></span>
      <span class="today-pill" id="today-label"></span>
    </div>
  </div>

  <div class="content">

    <!-- ALL EVENTS VIEW -->
    <div id="view-all">
      <div id="events-out">
        <div class="welcome">
          <div class="welcome-icon">&#128197;</div>
          <h2>Your Schedule</h2>
          <p>Start fresh — add events using the AI agent or the form, then load them here.</p>
          <button class="load-btn" onclick="loadEvents()">Load My Schedule</button>
        </div>
      </div>
    </div>

    <!-- ADD / MANAGE VIEW -->
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
            <option value="add">Add new event</option>
            <option value="update">Update existing event</option>
            <option value="delete">Delete event</option>
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
          <label>Description <span style="font-weight:400;text-transform:none;opacity:.55">(optional)</span></label>
          <input type="text" id="f-desc" placeholder="Extra details..."/>
          <button class="btn-primary" id="save-btn" onclick="saveEvent()">Save Event</button>
          <div class="status-msg" id="save-st"></div>
        </div>
      </div>
    </div>

    <!-- ASK AI VIEW -->
    <div id="view-ask" style="display:none">
      <div class="panel-wrap">
        <div class="panel" style="max-width:580px">
          <div class="panel-header">
            <div class="panel-icon">&#129302;</div>
            <div>
              <h2>Ask the AI<small>Query or update your schedule in plain English</small></h2>
            </div>
          </div>
          <label>Your question</label>
          <textarea id="q-in" rows="3"
            placeholder="e.g. What do I have tomorrow?&#10;Am I free Friday afternoon?&#10;Add a meeting on Monday at 3pm"></textarea>
          <div class="shortcut-tip">Ctrl+Enter to send</div>
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

_JS = """
<script>
const COLORS = {
  education:'#0ea5e9', work:'#818cf8', health:'#22c55e', family:'#f59e0b',
  social:'#ec4899', personal:'#a78bfa', shopping:'#f97316', entertainment:'#c084fc',
  travel:'#f43f5e', finance:'#14b8a6', home:'#84cc16', deadline:'#ef4444',
  recurring:'#06b6d4', appointment:'#d946ef', emergency:'#dc2626',
  meeting:'#3b82f6', task:'#fbbf24', workshop:'#10b981', other:'#94a3b8',
};

let allEvents = [], currentCat = 'all', eventsLoaded = false;
const todayStr = new Date().toISOString().split('T')[0];
const tomorrowStr = new Date(Date.now()+86400000).toISOString().split('T')[0];

document.getElementById('today-label').textContent =
  new Date().toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric',year:'numeric'});
document.getElementById('f-date').value = todayStr;

function showView(name) {
  ['all','add','ask'].forEach(v => {
    document.getElementById('view-'+v).style.display = v===name?'':'none';
    document.getElementById('nav-'+v).classList.toggle('active', v===name);
  });
  const titles = {all:'My Schedule', add:'Manage Event', ask:'Ask AI'};
  const badges = {all:'Events', add:'Event Manager', ask:'AI Agent'};
  document.getElementById('view-title').textContent = titles[name];
  document.getElementById('view-badge').textContent = badges[name];
  document.getElementById('count-pill').style.display = (name==='all'&&eventsLoaded)?'':'none';
}

function filterCat(cat) {
  currentCat = cat;
  document.querySelectorAll('.cat-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('cat-'+cat).classList.add('active');
  showView('all');
  if (eventsLoaded) loadEvents();
}

async function loadEvents() {
  eventsLoaded = true;
  document.getElementById('count-pill').style.display = '';
  document.getElementById('events-out').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading schedule...</div>';
  try {
    const url = currentCat==='all'?'/events':'/events?category='+currentCat;
    const r = await fetch(url);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail||'Failed');
    allEvents = d.events||[];
    renderEvents();
  } catch(e) {
    document.getElementById('events-out').innerHTML =
      '<div class="empty"><div class="empty-icon">&#9888;</div><span>Could not load: '+e.message+'</span></div>';
  }
}

function fmtTime(t) {
  if (!t) return {time:'—',ampm:''};
  const [h,m] = t.split(':').map(Number);
  return {time:(h%12||12)+':'+(m<10?'0':'')+m, ampm:h>=12?'PM':'AM'};
}

function fmtDate(ds) {
  const d = new Date(ds+'T12:00:00');
  const fmt = d.toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric',year:'numeric'});
  if (ds===todayStr) return '&#128205; Today &mdash; '+fmt;
  if (ds===tomorrowStr) return '&#128198; Tomorrow &mdash; '+fmt;
  return fmt;
}

function renderEvents() {
  const out = document.getElementById('events-out');
  const cp = document.getElementById('count-pill');
  if (!allEvents.length) {
    cp.style.display='none';
    out.innerHTML='<div class="empty"><div class="empty-icon">&#128197;</div>'
      +'<span>No events found'+(currentCat!=='all'?' in this category':'')
      +'. Use Add / Manage or Ask AI to create one.</span></div>';
    return;
  }
  cp.textContent = allEvents.length+(allEvents.length===1?' event':' events');

  const grouped = {};
  allEvents.forEach(e=>{if(!grouped[e.date])grouped[e.date]=[];grouped[e.date].push(e);});

  let html = '';
  Object.keys(grouped).sort().forEach(date => {
    const grp = grouped[date];
    html += `<div class="date-group">
      <div class="group-header">
        <span class="group-date">${fmtDate(date)}</span>
        <div class="group-line"></div>
        <span class="group-count">${grp.length}</span>
      </div>
      <div class="evt-list">`;
    grp.forEach(e => {
      const cat = e.category||'personal';
      const c = COLORS[cat]||'#38bdf8';
      const bg = c+'20', bd = c+'40';
      const lbl = cat.charAt(0).toUpperCase()+cat.slice(1).replace('_',' ');
      const {time, ampm} = fmtTime(e.time);
      const safeT = (e.title||'(no title)').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      const safeD = (e.description||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      const safeJs = (e.title||'').replace(/\\/g,'\\\\').replace(/'/g,"\\'");
      html += `<div class="evt-card" style="--card-color:${c}">
        <div class="evt-time-col">
          <div class="evt-time">${time}</div>
          <div class="evt-ampm">${ampm}</div>
        </div>
        <div class="evt-divider"></div>
        <div class="evt-body">
          <div class="evt-title">${safeT}</div>
          ${safeD?`<div class="evt-desc">${safeD}</div>`:''}
        </div>
        <span class="evt-badge" style="background:${bg};color:${c};border-color:${bd}">${lbl}</span>
        <button class="del-btn" title="Delete" onclick="delEvent('${e.id}','${safeJs}')">&#128465;</button>
      </div>`;
    });
    html += '</div></div>';
  });
  out.innerHTML = html;
}

async function delEvent(id, title) {
  if (!confirm('Delete "'+title+'"?')) return;
  try {
    const r = await fetch('/events/'+encodeURIComponent(id),{method:'DELETE'});
    if (!r.ok){const d=await r.json();throw new Error(d.detail);}
    allEvents = allEvents.filter(e=>e.id!==id);
    renderEvents();
  } catch(e){alert('Delete failed: '+e.message);}
}

async function saveEvent() {
  const action = document.getElementById('f-action').value;
  const date   = document.getElementById('f-date').value;
  const time   = document.getElementById('f-time').value||'09:00';
  const title  = document.getElementById('f-title').value.trim();
  const type   = document.getElementById('f-type').value;
  const desc   = document.getElementById('f-desc').value.trim();
  const btn    = document.getElementById('save-btn');
  const st     = document.getElementById('save-st');
  if (!date||!title){st.className='status-msg err';st.textContent='Date and title are required.';return;}
  btn.disabled=true; st.className='status-msg'; st.textContent='Saving...';
  const msg = action==='delete'
    ? `Delete the event titled "${title}" on ${date}`
    : action==='update'
    ? `Update event "${title}" on ${date}: change time to ${time}, category ${type}. ${desc}`
    : `Add a ${type} event titled "${title}" on ${date} at ${time}. ${desc}`;
  try {
    const r = await fetch('/agent/invoke',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({input:msg})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    st.className='status-msg ok';
    st.textContent = action==='delete'?'&#10003; Deleted.':action==='update'?'&#10003; Updated.':'&#10003; Event saved!';
    if (action==='add'){document.getElementById('f-title').value='';document.getElementById('f-desc').value='';}
  } catch(e){st.className='status-msg err';st.textContent='Error: '+e.message;}
  finally{btn.disabled=false;}
}

async function askAgent() {
  const q   = document.getElementById('q-in').value.trim();
  const btn = document.getElementById('ask-btn');
  const ans = document.getElementById('ask-ans');
  const err = document.getElementById('ask-err');
  if (!q) return;
  btn.disabled=true; btn.textContent='Thinking...';
  ans.style.display='none'; err.textContent='';
  try {
    const r = await fetch('/agent/invoke',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({input:q})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    ans.style.display='block'; ans.textContent=d.output;
  } catch(e){err.textContent='Error: '+e.message;}
  finally{btn.disabled=false; btn.textContent='Ask AI';}
}

document.getElementById('q-in').addEventListener('keydown',e=>{
  if (e.key==='Enter'&&e.ctrlKey) askAgent();
});
</script>
</body>
</html>
"""


@app.get("/agent/playground", response_class=HTMLResponse)
def playground():
    return _HTML + _JS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
