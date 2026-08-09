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


_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MyPlanner</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f0717;
  --card:#1a0f2e;
  --card2:#221340;
  --border:#2e1a52;
  --accent:#c084fc;
  --accent2:#e879f9;
  --glow:rgba(192,132,252,0.25);
  --rose:#fb7185;
  --teal:#2dd4bf;
  --amber:#fbbf24;
  --text:#f1e8ff;
  --muted:#8b6faa;
  --dim:#3d2660;
  --danger:#f43f5e;
  --ok:#4ade80;
  --r:14px;--rsm:9px;
  --tr:.16s ease;
}
body{font-family:'Poppins',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

/* ── TOPNAV ── */
nav{
  background:rgba(15,7,23,0.92);
  backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:100;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 32px;height:62px;
}
.brand{display:flex;align-items:center;gap:10px}
.brand-dot{
  width:34px;height:34px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  font-size:1.1rem;box-shadow:0 0 16px var(--glow);
}
.brand-name{font-size:1rem;font-weight:700;letter-spacing:-.02em;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-tabs{display:flex;gap:4px}
.tab{
  padding:7px 18px;border:none;background:none;
  color:var(--muted);font-size:.82rem;font-weight:500;
  cursor:pointer;border-radius:var(--rsm);
  transition:var(--tr);font-family:inherit;
}
.tab:hover{background:var(--card2);color:var(--text)}
.tab.active{background:rgba(192,132,252,0.15);color:var(--accent);border:1px solid rgba(192,132,252,0.3)}
.nav-right{display:flex;align-items:center;gap:10px}
.date-chip{font-size:.72rem;color:var(--muted);background:var(--card);
  padding:5px 12px;border-radius:99px;border:1px solid var(--border)}

/* ── PAGE ── */
.page{display:none;padding:28px 32px;max-width:1100px;margin:0 auto}
.page.active{display:block}

/* ── HERO ── */
.hero{
  background:linear-gradient(135deg,#1a0f2e 0%,#0f0717 60%,#1a0a1e 100%);
  border:1px solid var(--border);border-radius:20px;
  padding:36px 36px 32px;margin-bottom:28px;
  position:relative;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;top:-60px;right:-60px;
  width:220px;height:220px;border-radius:50%;
  background:radial-gradient(circle,rgba(192,132,252,0.15),transparent 70%);
  pointer-events:none;
}
.hero-label{font-size:.68rem;font-weight:600;color:var(--accent);
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px}
.hero-title{font-size:1.6rem;font-weight:700;letter-spacing:-.03em;line-height:1.2;margin-bottom:6px}
.hero-sub{font-size:.85rem;color:var(--muted);margin-bottom:24px}
.hero-actions{display:flex;gap:10px;flex-wrap:wrap}
.btn-glow{
  padding:10px 22px;border:none;border-radius:var(--rsm);
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#0f0717;font-weight:700;font-size:.85rem;
  cursor:pointer;font-family:inherit;
  box-shadow:0 0 20px var(--glow);transition:var(--tr);
}
.btn-glow:hover{transform:translateY(-2px);box-shadow:0 0 32px var(--glow)}
.btn-ghost{
  padding:10px 22px;border:1px solid var(--border);border-radius:var(--rsm);
  background:transparent;color:var(--muted);font-size:.85rem;font-weight:500;
  cursor:pointer;font-family:inherit;transition:var(--tr);
}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent)}

/* ── STATS ROW ── */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}
.stat-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--r);padding:18px 20px;
  display:flex;align-items:center;gap:14px;
}
.stat-icon{
  width:42px;height:42px;border-radius:12px;
  display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;
}
.stat-val{font-size:1.4rem;font-weight:700;letter-spacing:-.02em;line-height:1}
.stat-lbl{font-size:.72rem;color:var(--muted);margin-top:2px;font-weight:400}

/* ── CATEGORY PILLS ── */
.cat-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
.cat-pill{
  padding:5px 13px;border-radius:99px;font-size:.74rem;font-weight:500;
  border:1px solid var(--border);background:transparent;color:var(--muted);
  cursor:pointer;transition:var(--tr);font-family:inherit;
}
.cat-pill:hover{color:var(--text);border-color:var(--accent)}
.cat-pill.on{background:rgba(192,132,252,0.15);color:var(--accent);border-color:rgba(192,132,252,0.4)}

/* ── EVENT GRID ── */
.evt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
.evt-tile{
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--r);padding:18px;
  display:flex;flex-direction:column;gap:10px;
  transition:var(--tr);position:relative;overflow:hidden;
}
.evt-tile::after{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:var(--tile-color,var(--accent));border-radius:var(--r) var(--r) 0 0;
}
.evt-tile:hover{transform:translateY(-3px);border-color:rgba(192,132,252,0.35);
  box-shadow:0 8px 32px rgba(0,0,0,0.5)}
.tile-top{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}
.tile-title{font-weight:600;font-size:.92rem;letter-spacing:-.01em;line-height:1.3}
.tile-del{background:none;border:none;color:var(--dim);cursor:pointer;
  font-size:.9rem;border-radius:6px;padding:3px 5px;transition:var(--tr);flex-shrink:0}
.tile-del:hover{color:var(--danger);background:rgba(244,63,94,0.12)}
.tile-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.tile-time{font-size:.78rem;font-weight:600;color:var(--accent);
  background:rgba(192,132,252,0.1);padding:3px 9px;border-radius:99px;border:1px solid rgba(192,132,252,0.2)}
.tile-date{font-size:.72rem;color:var(--muted)}
.tile-badge{font-size:.62rem;font-weight:600;padding:2px 8px;border-radius:99px;
  text-transform:uppercase;letter-spacing:.05em;border:1px solid}
.tile-desc{font-size:.76rem;color:var(--muted);line-height:1.5}

/* ── EMPTY / LOADING ── */
.empty-state{
  grid-column:1/-1;text-align:center;padding:64px 20px;
  display:flex;flex-direction:column;align-items:center;gap:14px;
}
.empty-glyph{font-size:3rem;opacity:.3}
.empty-title{font-size:1rem;font-weight:600;color:var(--muted)}
.empty-sub{font-size:.8rem;color:var(--dim);max-width:280px;line-height:1.6}
.spinner{width:30px;height:30px;border:2.5px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .65s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── FORM PAGE ── */
.form-wrap{max-width:520px}
.form-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:20px;padding:30px;
}
.form-head{display:flex;align-items:center;gap:12px;margin-bottom:26px}
.form-icon{width:40px;height:40px;border-radius:12px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;
  box-shadow:0 0 14px var(--glow)}
.form-title{font-size:1rem;font-weight:700}
.form-sub{font-size:.72rem;color:var(--muted);margin-top:1px}
.fld{margin-bottom:16px}
.fld label{display:block;font-size:.7rem;font-weight:600;color:var(--muted);
  letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px}
.fld input,.fld select,.fld textarea{
  width:100%;padding:10px 13px;
  background:#120a22;border:1.5px solid var(--border);
  border-radius:var(--rsm);font-size:.86rem;color:var(--text);
  outline:none;font-family:inherit;transition:border-color var(--tr),box-shadow var(--tr);
  -webkit-appearance:none;
}
.fld input::placeholder,.fld textarea::placeholder{color:var(--dim)}
.fld input:focus,.fld select:focus,.fld textarea:focus{
  border-color:var(--accent);box-shadow:0 0 0 3px var(--glow)}
.fld textarea{min-height:68px;resize:vertical}
.fld select{
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='11' viewBox='0 0 24 24' fill='none' stroke='%238b6faa' stroke-width='2.5'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;padding-right:34px}
.fld select option{background:#1a0f2e}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.submit-btn{
  width:100%;padding:11px;margin-top:6px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#0f0717;border:none;border-radius:var(--rsm);
  font-size:.9rem;font-weight:700;cursor:pointer;font-family:inherit;
  box-shadow:0 0 18px var(--glow);transition:var(--tr);
}
.submit-btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 0 28px var(--glow)}
.submit-btn:disabled{opacity:.3;cursor:not-allowed;transform:none}
.form-msg{margin-top:10px;font-size:.78rem;text-align:center;min-height:14px;color:var(--muted)}
.form-msg.ok{color:var(--ok)}.form-msg.err{color:var(--danger)}

/* ── ASK PAGE ── */
.ask-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:20px;padding:30px;max-width:600px;
}
.ai-resp{
  margin-top:18px;background:#120a22;border:1px solid var(--border);
  border-radius:var(--rsm);padding:16px;
  font-size:.86rem;line-height:1.8;white-space:pre-wrap;
  display:none;color:var(--text);position:relative;
}
.ai-resp::before{
  content:'AI Response';position:absolute;top:-10px;left:14px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  color:#0f0717;font-size:.58rem;font-weight:700;
  padding:2px 9px;border-radius:99px;letter-spacing:.06em;
}
.ask-err{font-size:.76rem;color:var(--danger);margin-top:6px;min-height:12px}
.hint{font-size:.68rem;color:var(--dim);text-align:right;margin-top:5px}
</style>
</head>
<body>
"""

_UI2 = """
<nav>
  <div class="brand">
    <div class="brand-dot">&#128336;</div>
    <span class="brand-name">MyPlanner</span>
  </div>
  <div class="nav-tabs">
    <button class="tab active" onclick="goTab('home')" id="tab-home">&#127968; Dashboard</button>
    <button class="tab" onclick="goTab('add')" id="tab-add">&#43; New Event</button>
    <button class="tab" onclick="goTab('ask')" id="tab-ask">&#129302; Ask AI</button>
  </div>
  <div class="nav-right">
    <span class="date-chip" id="today-chip"></span>
  </div>
</nav>

<!-- HOME PAGE -->
<div class="page active" id="page-home">
  <div class="hero">
    <div class="hero-label">Your Personal Scheduler</div>
    <div class="hero-title">What's on your<br/>agenda today?</div>
    <div class="hero-sub">Add events, ask AI, and stay on top of your schedule.</div>
    <div class="hero-actions">
      <button class="btn-glow" onclick="loadEvents()">&#128197; Load Events</button>
      <button class="btn-ghost" onclick="goTab('add')">&#43; Add Event</button>
    </div>
  </div>

  <div class="stats">
    <div class="stat-card">
      <div class="stat-icon" style="background:rgba(192,132,252,0.15)">&#128197;</div>
      <div><div class="stat-val" id="s-total">—</div><div class="stat-lbl">Total Events</div></div>
    </div>
    <div class="stat-card">
      <div class="stat-icon" style="background:rgba(251,191,36,0.15)">&#9201;</div>
      <div><div class="stat-val" id="s-today">—</div><div class="stat-lbl">Today</div></div>
    </div>
    <div class="stat-card">
      <div class="stat-icon" style="background:rgba(45,212,191,0.15)">&#128640;</div>
      <div><div class="stat-val" id="s-upcoming">—</div><div class="stat-lbl">Upcoming</div></div>
    </div>
  </div>

  <div class="cat-row" id="cat-row">
    <button class="cat-pill on" onclick="filterCat('all')" id="cp-all">All</button>
    <button class="cat-pill" onclick="filterCat('education')" id="cp-education">Education</button>
    <button class="cat-pill" onclick="filterCat('work')" id="cp-work">Work</button>
    <button class="cat-pill" onclick="filterCat('health')" id="cp-health">Health</button>
    <button class="cat-pill" onclick="filterCat('family')" id="cp-family">Family</button>
    <button class="cat-pill" onclick="filterCat('social')" id="cp-social">Social</button>
    <button class="cat-pill" onclick="filterCat('personal')" id="cp-personal">Personal</button>
    <button class="cat-pill" onclick="filterCat('shopping')" id="cp-shopping">Shopping</button>
    <button class="cat-pill" onclick="filterCat('entertainment')" id="cp-entertainment">Entertainment</button>
    <button class="cat-pill" onclick="filterCat('travel')" id="cp-travel">Travel</button>
    <button class="cat-pill" onclick="filterCat('finance')" id="cp-finance">Finance</button>
    <button class="cat-pill" onclick="filterCat('home')" id="cp-home">Home</button>
    <button class="cat-pill" onclick="filterCat('deadline')" id="cp-deadline">Deadline</button>
    <button class="cat-pill" onclick="filterCat('recurring')" id="cp-recurring">Recurring</button>
    <button class="cat-pill" onclick="filterCat('appointment')" id="cp-appointment">Appointment</button>
    <button class="cat-pill" onclick="filterCat('emergency')" id="cp-emergency">Emergency</button>
    <button class="cat-pill" onclick="filterCat('meeting')" id="cp-meeting">Meeting</button>
    <button class="cat-pill" onclick="filterCat('task')" id="cp-task">Task</button>
    <button class="cat-pill" onclick="filterCat('workshop')" id="cp-workshop">Workshop</button>
    <button class="cat-pill" onclick="filterCat('other')" id="cp-other">Other</button>
  </div>

  <div class="evt-grid" id="evt-grid">
    <div class="empty-state">
      <div class="empty-glyph">&#128197;</div>
      <div class="empty-title">No events yet</div>
      <div class="empty-sub">Click "Load Events" above or add a new event to get started.</div>
    </div>
  </div>
</div>

<!-- ADD EVENT PAGE -->
<div class="page" id="page-add">
  <div class="form-wrap">
    <div class="form-card">
      <div class="form-head">
        <div class="form-icon">&#9998;</div>
        <div><div class="form-title">Manage Event</div><div class="form-sub">Create, update or remove a schedule item</div></div>
      </div>
      <div class="fld"><label>Action</label>
        <select id="f-action">
          <option value="add">Add new event</option>
          <option value="update">Update existing event</option>
          <option value="delete">Delete event</option>
        </select>
      </div>
      <div class="row2">
        <div class="fld"><label>Date</label><input type="date" id="f-date"/></div>
        <div class="fld"><label>Time</label><input type="time" id="f-time" value="09:00"/></div>
      </div>
      <div class="fld"><label>Title</label>
        <input type="text" id="f-title" placeholder="e.g. Doctor visit, Team sync..."/>
      </div>
      <div class="fld"><label>Category</label>
        <select id="f-cat">
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
      </div>
      <div class="fld"><label>Description <span style="opacity:.5;font-weight:400;text-transform:none">(optional)</span></label>
        <input type="text" id="f-desc" placeholder="Extra details..."/>
      </div>
      <button class="submit-btn" id="save-btn" onclick="saveEvent()">Save Event</button>
      <div class="form-msg" id="save-msg"></div>
    </div>
  </div>
</div>

<!-- ASK AI PAGE -->
<div class="page" id="page-ask">
  <div class="ask-card">
    <div class="form-head">
      <div class="form-icon">&#129302;</div>
      <div><div class="form-title">Ask the AI</div><div class="form-sub">Ask anything about your schedule in plain English</div></div>
    </div>
    <div class="fld"><label>Your question</label>
      <textarea id="q-txt" rows="3" placeholder="e.g. What do I have this week?&#10;Am I free tomorrow afternoon?&#10;Schedule a meeting Friday at 3pm"></textarea>
      <div class="hint">Ctrl + Enter to send</div>
    </div>
    <button class="submit-btn" id="ask-btn" onclick="askAI()">Ask AI</button>
    <div class="ai-resp" id="ai-resp"></div>
    <div class="ask-err" id="ask-err"></div>
  </div>
</div>
"""

_UI3 = """
<script>
const COLORS = {
  education:'#38bdf8', work:'#818cf8', health:'#4ade80', family:'#fbbf24',
  social:'#f472b6', personal:'#c084fc', shopping:'#fb923c', entertainment:'#e879f9',
  travel:'#f43f5e', finance:'#2dd4bf', home:'#a3e635', deadline:'#f87171',
  recurring:'#22d3ee', appointment:'#d946ef', emergency:'#dc2626',
  meeting:'#60a5fa', task:'#facc15', workshop:'#34d399', other:'#94a3b8',
};

let allEvents = [], currentCat = 'all', loaded = false;
const todayStr = new Date().toISOString().split('T')[0];
const tomorrowStr = new Date(Date.now()+86400000).toISOString().split('T')[0];

document.getElementById('today-chip').textContent =
  new Date().toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric'});
document.getElementById('f-date').value = todayStr;

function goTab(t) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById('page-'+t).classList.add('active');
  document.getElementById('tab-'+t).classList.add('active');
}

function filterCat(cat) {
  currentCat = cat;
  document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('on'));
  document.getElementById('cp-'+cat).classList.add('on');
  if (loaded) renderGrid();
}

async function loadEvents() {
  const grid = document.getElementById('evt-grid');
  grid.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
  try {
    const url = currentCat==='all' ? '/events' : '/events?category='+currentCat;
    const r = await fetch(url);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    allEvents = d.events || [];
    loaded = true;
    updateStats();
    renderGrid();
  } catch(e) {
    grid.innerHTML = '<div class="empty-state"><div class="empty-glyph">&#9888;</div><div class="empty-title">Could not load</div><div class="empty-sub">'+e.message+'</div></div>';
  }
}

function updateStats() {
  document.getElementById('s-total').textContent = allEvents.length;
  document.getElementById('s-today').textContent = allEvents.filter(e=>e.date===todayStr).length;
  document.getElementById('s-upcoming').textContent = allEvents.filter(e=>e.date>todayStr).length;
}

function fmtDate(ds) {
  if (ds===todayStr) return 'Today';
  if (ds===tomorrowStr) return 'Tomorrow';
  return new Date(ds+'T12:00:00').toLocaleDateString('en-US',{month:'short',day:'numeric'});
}

function fmt12(t) {
  if (!t) return '';
  const [h,m] = t.split(':').map(Number);
  return (h%12||12)+':'+(m<10?'0':'')+m+(h>=12?' PM':' AM');
}

function renderGrid() {
  const grid = document.getElementById('evt-grid');
  const list = currentCat==='all' ? allEvents : allEvents.filter(e=>e.category===currentCat);
  if (!list.length) {
    grid.innerHTML = '<div class="empty-state"><div class="empty-glyph">&#128197;</div><div class="empty-title">No events'+(currentCat!=='all'?' in this category':'')+'</div><div class="empty-sub">Add one using the New Event tab.</div></div>';
    return;
  }
  grid.innerHTML = list.map(e => {
    const c = COLORS[e.category]||'#c084fc';
    const bg = c+'18', border = c+'44';
    const lbl = (e.category||'personal');
    const labelStr = lbl.charAt(0).toUpperCase()+lbl.slice(1);
    const safeT = (e.title||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const safeD = (e.description||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const safeJs = (e.title||'').replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    return `<div class="evt-tile" style="--tile-color:${c}">
      <div class="tile-top">
        <div class="tile-title">${safeT||'(no title)'}</div>
        <button class="tile-del" onclick="delEvent('${e.id}','${safeJs}')" title="Delete">&#128465;</button>
      </div>
      <div class="tile-meta">
        ${e.time?`<span class="tile-time">${fmt12(e.time)}</span>`:''}
        <span class="tile-date">${fmtDate(e.date)}</span>
      </div>
      ${safeD?`<div class="tile-desc">${safeD}</div>`:''}
      <div><span class="tile-badge" style="background:${bg};color:${c};border-color:${border}">${labelStr}</span></div>
    </div>`;
  }).join('');
}

async function delEvent(id, title) {
  if (!confirm('Delete "'+title+'"?')) return;
  try {
    const r = await fetch('/events/'+encodeURIComponent(id),{method:'DELETE'});
    if (!r.ok) { const d=await r.json(); throw new Error(d.detail); }
    allEvents = allEvents.filter(e=>e.id!==id);
    updateStats(); renderGrid();
  } catch(e) { alert('Delete failed: '+e.message); }
}

async function saveEvent() {
  const action = document.getElementById('f-action').value;
  const date   = document.getElementById('f-date').value;
  const time   = document.getElementById('f-time').value||'09:00';
  const title  = document.getElementById('f-title').value.trim();
  const cat    = document.getElementById('f-cat').value;
  const desc   = document.getElementById('f-desc').value.trim();
  const btn    = document.getElementById('save-btn');
  const msg    = document.getElementById('save-msg');
  if (!date||!title){msg.className='form-msg err';msg.textContent='Date and title are required.';return;}
  btn.disabled=true; msg.className='form-msg'; msg.textContent='Saving...';
  const text = action==='delete'
    ? `Delete the event titled "${title}" on ${date}`
    : action==='update'
    ? `Update event "${title}" on ${date}: time ${time}, category ${cat}. ${desc}`
    : `Add a ${cat} event titled "${title}" on ${date} at ${time}. ${desc}`;
  try {
    const r = await fetch('/agent/invoke',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:text})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    msg.className='form-msg ok';
    msg.textContent = action==='delete'?'Deleted!':action==='update'?'Updated!':'Event saved!';
    if (action==='add'){document.getElementById('f-title').value='';document.getElementById('f-desc').value='';}
  } catch(e){msg.className='form-msg err';msg.textContent='Error: '+e.message;}
  finally{btn.disabled=false;}
}

async function askAI() {
  const q = document.getElementById('q-txt').value.trim();
  const btn = document.getElementById('ask-btn');
  const resp = document.getElementById('ai-resp');
  const err = document.getElementById('ask-err');
  if (!q) return;
  btn.disabled=true; btn.textContent='Thinking...';
  resp.style.display='none'; err.textContent='';
  try {
    const r = await fetch('/agent/invoke',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:q})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    resp.style.display='block'; resp.textContent=d.output;
  } catch(e){err.textContent='Error: '+e.message;}
  finally{btn.disabled=false; btn.textContent='Ask AI';}
}

document.getElementById('q-txt').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')askAI();});
</script>
</body>
</html>
"""


@app.get("/agent/playground", response_class=HTMLResponse)
def playground():
    return _UI + _UI2 + _UI3


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
