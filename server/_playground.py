"""Interactive playground HTML served at GET /."""

PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>r/science Mod Bot — Playground</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#dae0e6;color:#1c1c1c;min-height:100vh}
    a{color:#0079d3}

    header{background:#ff4500;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:12px}
    header h1{font-size:1.15rem;font-weight:700}
    header p{font-size:.82rem;opacity:.9;margin-top:2px}

    .wrap{max-width:820px;margin:20px auto;padding:0 14px}

    .card{background:#fff;border:1px solid #ccc;border-radius:4px;padding:16px;margin-bottom:14px}

    /* Setup row */
    .setup{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
    .setup label{font-weight:600;font-size:.88rem;white-space:nowrap}
    .lvl-btns{display:flex;gap:6px}
    .lvl-btn{padding:5px 14px;border:2px solid #ddd;background:#fff;border-radius:4px;cursor:pointer;font-size:.82rem;transition:.12s}
    .lvl-btn.on{border-color:#ff4500;color:#ff4500;font-weight:700}
    .go-btn{background:#ff4500;color:#fff;border:none;padding:7px 18px;border-radius:4px;cursor:pointer;font-weight:700;font-size:.88rem;margin-left:auto}
    .go-btn:hover{background:#e03d00}
    .go-btn:disabled{background:#aaa;cursor:not-allowed}

    /* Progress */
    .prog-row{display:flex;align-items:center;gap:14px}
    .bar-wrap{flex:1;background:#eee;border-radius:99px;height:7px}
    .bar{background:#ff4500;height:7px;border-radius:99px;transition:width .3s;width:0%}
    .stat{font-size:.78rem;color:#555;white-space:nowrap}

    /* Post */
    .post-meta{display:flex;gap:10px;align-items:center;font-size:.78rem;color:#666;flex-wrap:wrap;margin-bottom:8px}
    .sub{font-weight:700;color:#ff4500}
    .flair{background:#0079d3;color:#fff;padding:1px 8px;border-radius:99px;font-size:.72rem}
    .post-title{font-size:1.05rem;font-weight:700;margin-bottom:8px;line-height:1.35}
    .post-body{font-size:.88rem;line-height:1.55;color:#333;white-space:pre-wrap;margin-bottom:12px;max-height:180px;overflow-y:auto;border-left:3px solid #eee;padding-left:10px}

    .reports{background:#fff3cd;border:1px solid #ffc107;border-radius:4px;padding:7px 11px;font-size:.78rem;color:#856404;margin-bottom:10px}

    .author-box{background:#f6f7f8;border-radius:4px;padding:9px 12px;font-size:.8rem;margin-bottom:10px}
    .author-name{font-weight:700;margin-bottom:5px}
    .author-stats{display:flex;gap:14px;flex-wrap:wrap;color:#444}
    .badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.68rem;margin-left:5px;vertical-align:middle}
    .badge.red{background:#ff585b;color:#fff}
    .badge.green{background:#46d160;color:#fff}

    .rules-toggle{font-size:.78rem;color:#0079d3;cursor:pointer;background:none;border:none;padding:0;margin-bottom:6px}
    .rules-list{font-size:.78rem;margin-bottom:12px;padding-left:16px}
    .rules-list li{margin-bottom:4px;color:#444;line-height:1.4}
    .rules-list li strong{color:#1c1c1c}

    /* Thread context */
    .thread-ctx{background:#f0f4ff;border-left:3px solid #0079d3;padding:8px 11px;font-size:.78rem;color:#333;margin-bottom:10px;border-radius:0 4px 4px 0}
    .thread-ctx p{margin-bottom:3px;opacity:.7;font-weight:600}

    /* Actions */
    .act-label{font-weight:600;font-size:.85rem;margin-bottom:9px}
    .act-btns{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}
    .act-btn{padding:7px 15px;border:2px solid;border-radius:4px;cursor:pointer;font-size:.82rem;font-weight:600;background:#fff;transition:.12s}
    .act-btn:disabled{opacity:.35;cursor:not-allowed}
    .act-btn.approve{border-color:#46d160;color:#46d160}
    .act-btn.approve:hover:not(:disabled){background:#46d160;color:#fff}
    .act-btn.remove{border-color:#ff585b;color:#ff585b}
    .act-btn.remove:hover:not(:disabled){background:#ff585b;color:#fff}
    .act-btn.warn{border-color:#f6a629;color:#c17d00}
    .act-btn.warn:hover:not(:disabled){background:#f6a629;color:#fff}
    .act-btn.temp_ban{border-color:#8b5cf6;color:#8b5cf6}
    .act-btn.temp_ban:hover:not(:disabled){background:#8b5cf6;color:#fff}
    .act-btn.perma_ban{border-color:#dc2626;color:#dc2626}
    .act-btn.perma_ban:hover:not(:disabled){background:#dc2626;color:#fff}
    .act-btn.escalate_to_senior_mod{border-color:#0079d3;color:#0079d3}
    .act-btn.escalate_to_senior_mod:hover:not(:disabled){background:#0079d3;color:#fff}

    .extras{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px;font-size:.82rem}
    .extras label{color:#555}
    .extras select,.extras input[type=number]{padding:4px 8px;border:1px solid #ccc;border-radius:4px;font-size:.82rem}

    /* Feedback */
    .fb{padding:11px 14px;border-radius:4px;font-size:.88rem;line-height:1.5;margin-top:12px}
    .fb.correct{background:#dcfce7;border:1px solid #86efac}
    .fb.incorrect{background:#fee2e2;border:1px solid #fca5a5}
    .fb.info{background:#e8f0fe;border:1px solid #93c5fd}
    .fb.done{background:#fef3c7;border:1px solid #fcd34d;font-weight:500}

    .hidden{display:none!important}

    footer{text-align:center;color:#999;font-size:.72rem;padding:20px 0}
  </style>
</head>
<body>
<header>
  <div>
    <h1>r/science Mod Bot &mdash; Playground</h1>
    <p>Interactive RL environment &mdash; moderate posts across three difficulty levels</p>
  </div>
</header>

<div class="wrap">

  <!-- Setup -->
  <div class="card setup">
    <label>Task level:</label>
    <div class="lvl-btns">
      <button class="lvl-btn on" data-l="1" onclick="setLevel(1)">1 &mdash; Spam</button>
      <button class="lvl-btn"   data-l="2" onclick="setLevel(2)">2 &mdash; Rules</button>
      <button class="lvl-btn"   data-l="3" onclick="setLevel(3)">3 &mdash; Context</button>
    </div>
    <button class="go-btn" id="goBtn" onclick="startEpisode()">Start Episode</button>
  </div>

  <!-- Progress -->
  <div class="card hidden" id="progCard">
    <div class="prog-row">
      <div class="bar-wrap"><div class="bar" id="bar"></div></div>
      <span class="stat" id="statPosts">0 / 0</span>
      <span class="stat" id="statScore">Score: 0.00</span>
      <span class="stat" id="statAcc">Accuracy: &mdash;</span>
    </div>
  </div>

  <!-- Post + Actions -->
  <div class="card hidden" id="postCard">
    <div class="post-meta">
      <span class="sub" id="pSub"></span>
      <span id="pScore"></span>
      <span id="pComments"></span>
      <span class="flair hidden" id="pFlair"></span>
    </div>
    <div class="post-title" id="pTitle"></div>
    <div class="post-body"  id="pBody"></div>

    <div class="reports hidden" id="pReports"></div>

    <div class="thread-ctx hidden" id="pThread">
      <p>Thread context</p>
      <div id="pThreadBody"></div>
    </div>

    <div class="author-box">
      <div class="author-name" id="aName"></div>
      <div class="author-stats" id="aStats"></div>
    </div>

    <button class="rules-toggle" onclick="toggleRules()">&#9654; Show subreddit rules</button>
    <ul class="rules-list hidden" id="rulesList"></ul>

    <hr style="border:none;border-top:1px solid #eee;margin:12px 0">

    <div class="act-label">Your decision:</div>
    <div class="act-btns" id="actBtns"></div>
    <div class="extras"   id="extras"></div>

    <div class="fb hidden" id="fb"></div>
  </div>

  <footer>
    OpenEnv &middot; Reddit Mod Bot RL Environment &middot; <a href="/docs">API Docs</a>
  </footer>
</div>

<script>
const ALLOWED = {
  1: ['approve','remove'],
  2: ['approve','remove','warn'],
  3: ['approve','remove','warn','temp_ban','perma_ban','escalate_to_senior_mod']
};
const LABELS = {
  approve:'Approve', remove:'Remove', warn:'Warn',
  temp_ban:'Temp Ban', perma_ban:'Perma Ban', escalate_to_senior_mod:'Escalate'
};

let level = 1, total = 0, correct = 0, currentRules = [];

function setLevel(n) {
  level = n;
  document.querySelectorAll('.lvl-btn').forEach(b => b.classList.toggle('on', +b.dataset.l === n));
}

async function startEpisode() {
  const btn = document.getElementById('goBtn');
  btn.disabled = true;
  correct = 0;

  try {
    const res = await fetch('/pg/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_level: level})
    });
    if (!res.ok) { alert('Reset failed: ' + res.status); return; }
    const obs = await res.json();

    total = obs.posts_remaining;
    document.getElementById('progCard').classList.remove('hidden');
    renderObs(obs, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Restart';
  }
}

async function submitAction(actionType) {
  setActionsDisabled(true);

  const payload = {action_type: actionType};
  const rc = document.getElementById('xRule');
  if (rc && rc.value) payload.rule_cited = parseInt(rc.value);
  const bd = document.getElementById('xBanDays');
  if (bd && actionType === 'temp_ban') payload.ban_duration_days = parseInt(bd.value) || 7;

  try {
    const res = await fetch('/pg/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!res.ok) { alert('Step failed: ' + res.status); setActionsDisabled(false); return; }
    const obs = await res.json();
    if (obs.action_was_correct) correct++;
    renderObs(obs, false);
  } catch(e) {
    alert('Error: ' + e.message);
    setActionsDisabled(false);
  }
}

function renderObs(obs, isReset) {
  updateProgress(obs);

  if (obs.done || !obs.current_post) {
    showFeedback(obs.feedback, 'done');
    document.getElementById('actBtns').innerHTML = '';
    document.getElementById('extras').innerHTML = '';
    return;
  }

  renderPost(obs.current_post);
  renderActions(obs.task_level, obs.current_post.subreddit?.rules || []);

  if (isReset) {
    showFeedback(obs.feedback, 'info');
  } else if (obs.action_was_correct !== null) {
    showFeedback(obs.feedback, obs.action_was_correct ? 'correct' : 'incorrect');
  }
}

function renderPost(post) {
  document.getElementById('postCard').classList.remove('hidden');

  document.getElementById('pSub').textContent = post.subreddit?.name || 'r/science';
  document.getElementById('pScore').textContent = '\\u2191 ' + post.score;
  document.getElementById('pComments').textContent = '\\uD83D\\uDCAC ' + post.num_comments;

  const fl = document.getElementById('pFlair');
  if (post.flair) { fl.textContent = post.flair; fl.classList.remove('hidden'); }
  else fl.classList.add('hidden');

  document.getElementById('pTitle').textContent = post.title || '';
  document.getElementById('pBody').textContent = post.body || '';

  // Reports
  const rb = document.getElementById('pReports');
  if (post.reports?.length) {
    rb.classList.remove('hidden');
    rb.innerHTML = '&#9873; Reports: ' + post.reports.map(r =>
      `<strong>${r.reason}</strong> (&times;${r.count})`).join(', ');
  } else rb.classList.add('hidden');

  // Thread context
  const tc = document.getElementById('pThread');
  if (post.thread_context?.length) {
    tc.classList.remove('hidden');
    document.getElementById('pThreadBody').innerHTML = post.thread_context
      .map(c => `<p style="margin-bottom:4px;padding-left:8px;border-left:2px solid #ccc">${esc(c)}</p>`).join('');
  } else tc.classList.add('hidden');

  // Author
  const a = post.author;
  const badges = (a.is_repeat_offender ? '<span class="badge red">Repeat Offender</span>' : '') +
                 (a.is_approved_contributor ? '<span class="badge green">Approved</span>' : '');
  document.getElementById('aName').innerHTML = `<strong>u/${esc(a.username)}</strong>${badges}`;
  document.getElementById('aStats').innerHTML =
    `<span>Karma: <strong>${a.karma.toLocaleString()}</strong></span>` +
    `<span>Age: <strong>${a.account_age_days}d</strong></span>` +
    `<span>Warnings: <strong>${a.prior_warnings}</strong></span>` +
    `<span>Removals: <strong>${a.prior_removals}</strong></span>`;

  // Rules
  currentRules = post.subreddit?.rules || [];
  const rl = document.getElementById('rulesList');
  rl.innerHTML = currentRules.map(r =>
    `<li><strong>Rule ${r.rule_number}: ${esc(r.title)}</strong> &mdash; ${esc(r.description)}</li>`
  ).join('');
  rl.classList.add('hidden');
  document.querySelector('.rules-toggle').textContent = '\\u25B6 Show subreddit rules';
}

function renderActions(lv, rules) {
  const btns = document.getElementById('actBtns');
  const ex   = document.getElementById('extras');
  btns.innerHTML = '';
  ex.innerHTML = '';

  const allowed = ALLOWED[lv] || ALLOWED[1];

  // Rule citation select (levels 2+)
  if (lv >= 2 && rules.length) {
    const opts = rules.map(r => `<option value="${r.rule_number}">Rule ${r.rule_number}: ${esc(r.title)}</option>`).join('');
    ex.innerHTML += `<label>Rule cited:</label><select id="xRule"><option value="">— none —</option>${opts}</select>`;
  }

  // Ban duration (level 3)
  if (allowed.includes('temp_ban')) {
    ex.innerHTML += `<label>Ban duration:</label><input type="number" id="xBanDays" value="7" min="1" max="365" style="width:65px"> days`;
  }

  allowed.forEach(act => {
    const b = document.createElement('button');
    b.className = 'act-btn ' + act;
    b.textContent = LABELS[act];
    b.onclick = () => submitAction(act);
    btns.appendChild(b);
  });
}

function updateProgress(obs) {
  const reviewed = total - obs.posts_remaining;
  const pct = total > 0 ? (reviewed / total * 100) : 0;
  document.getElementById('bar').style.width = pct + '%';
  document.getElementById('statPosts').textContent = `${reviewed} / ${total}`;
  document.getElementById('statScore').textContent = `Score: ${(obs.cumulative_reward || 0).toFixed(2)}`;
  document.getElementById('statAcc').textContent = reviewed > 0
    ? `Accuracy: ${Math.round(correct / reviewed * 100)}%`
    : 'Accuracy: \u2014';
}

function showFeedback(text, cls) {
  const fb = document.getElementById('fb');
  fb.className = 'fb ' + cls;
  fb.textContent = text;
  fb.classList.remove('hidden');
  fb.scrollIntoView({behavior:'smooth', block:'nearest'});
}

function setActionsDisabled(v) {
  document.querySelectorAll('.act-btn').forEach(b => b.disabled = v);
}

function toggleRules() {
  const rl = document.getElementById('rulesList');
  const btn = document.querySelector('.rules-toggle');
  const hidden = rl.classList.toggle('hidden');
  btn.textContent = (hidden ? '\\u25B6' : '\\u25BC') + ' Show subreddit rules';
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""
