function fillSearch(text) {
  document.getElementById('queryInput').value = text;
  search();
}

function getSelectedSources() {
  const sources = [];
  document.querySelectorAll('.source-filter:checked').forEach(checkbox => {
    sources.push(checkbox.value);
  });
  console.log('📍 Selected sources:', sources);
  return sources;
}

let allJobs = [];
let shownCount = 0;
const BATCH = 5;

function renderJobs(jobs, startIndex = 0) {
  const list = document.getElementById('jobsList');
  const batch = jobs.slice(startIndex, startIndex + BATCH);

  batch.forEach((job, i) => {
    const isLinkedIn = job.source === 'linkedin';
    const badgeStyle = isLinkedIn
      ? 'border:1px solid rgba(96,165,250,0.4);color:#60a5fa;background:rgba(96,165,250,0.1);padding:3px 10px;border-radius:999px;font-size:11px'
      : 'border:1px solid rgba(45,212,191,0.4);color:#2dd4bf;background:rgba(45,212,191,0.1);padding:3px 10px;border-radius:999px;font-size:11px';

    const matchBadge = job.match
      ? `<span class="mono text-xs font-bold" style="color:${job.match >= 85 ? '#2dd4bf' : job.match >= 70 ? '#f59e0b' : '#94a3b8'}">${job.match}% match</span>`
      : '';

    const card = document.createElement('div');
    card.className = 'fade-up bg-[#161f30] border border-[#1e2d42] hover:border-teal-400/30 rounded-2xl p-6 mb-4 transition-all duration-300 hover:-translate-y-1';
    card.style.animationDelay = `${i * 0.08}s`;

    card.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
        <div class="mono text-teal-400/60 text-xs">// ${String(startIndex + i + 1).padStart(2,'0')}</div>
        <div style="display:flex;gap:8px;align-items:center">
          ${matchBadge}
          <span class="mono text-xs" style="${badgeStyle}">${isLinkedIn ? '🔗 LinkedIn' : '🟢 Wuzzuf'}</span>
        </div>
      </div>
      <div class="font-bold text-lg text-[#f1ede6] mb-1">${job.title || 'N/A'}</div>
      <div class="text-teal-400 font-semibold text-sm mb-4">🏢 ${job.company || 'N/A'}</div>
      <div class="flex flex-wrap gap-2 mb-5">
        <span class="bg-white/[0.04] border border-[#1e2d42] rounded-full px-3 py-1 text-xs text-slate-400" style="direction:ltr">📍 ${job.location || 'N/A'}</span>
        <span class="bg-white/[0.04] border border-[#1e2d42] rounded-full px-3 py-1 text-xs text-slate-400" style="direction:ltr">⏱ ${job.experience || 'N/A'}</span>
        <span class="bg-white/[0.04] border border-[#1e2d42] rounded-full px-3 py-1 text-xs text-slate-400">📋 ${job.job_type || 'N/A'}</span>
        <span class="bg-white/[0.04] border border-[#1e2d42] rounded-full px-3 py-1 text-xs text-slate-400">🎯 ${job.level || 'N/A'}</span>
      </div>
      <a href="${job.url || '#'}" target="_blank"
         class="inline-flex items-center gap-2 border border-teal-400 text-teal-400 hover:bg-teal-400 hover:text-[#0b1120] rounded-xl px-5 py-2 text-sm font-bold transition-all duration-200">
        تقدم للوظيفة ←
      </a>
    `;
    list.appendChild(card);
  });

  shownCount = startIndex + batch.length;

  const oldBtn = document.getElementById('loadMoreBtn');
  if (oldBtn) oldBtn.remove();

  if (shownCount < jobs.length) {
    const remaining = jobs.length - shownCount;
    const btn = document.createElement('button');
    btn.id = 'loadMoreBtn';
    btn.className = 'w-full mt-2 mb-6 py-3 border border-[#1e2d42] hover:border-teal-400/50 text-slate-400 hover:text-teal-400 rounded-2xl text-sm mono transition-all duration-200';
    btn.innerHTML = `عرض ${Math.min(BATCH, remaining)} أكتر ← (${remaining} متبقي)`;
    btn.onclick = () => renderJobs(jobs, shownCount);
    document.getElementById('jobsList').appendChild(btn);
  }
}

async function search() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const sources = getSelectedSources();
  
  if (sources.length === 0) {
    alert('Please select at least one website to search from');
    return;
  }

  console.log('🔍 Searching with query:', query);
  console.log('📍 Selected sources:', sources);

  allJobs = [];
  shownCount = 0;
  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('aiAnswer').classList.add('hidden');
  document.getElementById('jobsSection').classList.add('hidden');
  document.getElementById('jobsList').innerHTML = '';
  document.getElementById('aiText').innerHTML = '';

  try {
    const res = await fetch('/search/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, sources })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    document.getElementById('loading').classList.add('hidden');
    document.getElementById('aiAnswer').classList.remove('hidden');

    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    document.getElementById('aiText').appendChild(cursor);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const lines = decoder.decode(value).split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') { cursor.remove(); break; }

        try {
          const parsed = JSON.parse(data);

          // ── Jobs: renderJobs مرة واحدة بس ──
          if (parsed.type === 'jobs' && parsed.jobs.length > 0) {
            allJobs = parsed.jobs;
            document.getElementById('jobsSection').classList.remove('hidden');
            renderJobs(allJobs, 0);  // ← هنا بس
          }

          // ── Streaming text ──
          if (parsed.type === 'text') {
            document.getElementById('aiText').insertBefore(
              document.createTextNode(parsed.content), cursor
            );
          }

        } catch(e) {}
      }
    }

  } catch(err) {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('aiAnswer').classList.remove('hidden');
    document.getElementById('aiText').innerHTML =
      `<span style="color:#f87171">حصل خطأ: ${err.message}</span>`;
  }
}

async function uploadCV(input) {
  const file = input.files[0];
  if (!file) return;

  allJobs = [];
  shownCount = 0;

  const status = document.getElementById('cvStatus');
  status.classList.remove('hidden');
  status.textContent = '⏳ جاري تحليل الـ CV...';

  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('aiAnswer').classList.add('hidden');
  document.getElementById('jobsSection').classList.add('hidden');
  document.getElementById('jobsList').innerHTML = '';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res  = await fetch('/cv-match', { method: 'POST', body: formData });
    const data = await res.json();

    document.getElementById('loading').classList.add('hidden');

    if (data.error) { status.textContent = '❌ ' + data.error; return; }

    status.textContent = `✅ تم تحليل الـ CV — ${data.matches.length} وظايف مناسبة`;

    document.getElementById('aiText').textContent = data.summary;
    document.getElementById('aiAnswer').classList.remove('hidden');

    allJobs = data.matches;
    document.getElementById('jobsSection').classList.remove('hidden');
    renderJobs(allJobs, 0);  // ← هنا بس

  } catch(err) {
    document.getElementById('loading').classList.add('hidden');
    status.textContent = '❌ حصل خطأ: ' + err.message;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('queryInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') search();
  });
});