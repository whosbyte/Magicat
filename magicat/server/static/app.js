const form = document.getElementById("submit-form");
const progressCard = document.getElementById("progress-card");
const progressList = document.getElementById("progress");
const results = document.getElementById("results");
const submitBtn = document.getElementById("submit");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  progressList.innerHTML = "";
  results.hidden = true;
  progressCard.hidden = false;

  const url = document.getElementById("url").value.trim();
  const fileInput = document.getElementById("file");
  let resp;
  if (fileInput.files.length > 0) {
    const data = new FormData();
    data.append("file", fileInput.files[0]);
    resp = await fetch("/api/jobs", { method: "POST", body: data });
  } else if (url) {
    resp = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
  } else {
    alert("Paste a URL or choose a file.");
    submitBtn.disabled = false;
    return;
  }
  if (!resp.ok) {
    alert("Submit failed: " + resp.status);
    submitBtn.disabled = false;
    return;
  }
  const { job_id } = await resp.json();
  watch(job_id);
});

function watch(jobId) {
  const items = {};
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (msg) => {
    const { stage, state } = JSON.parse(msg.data);
    if (stage === "job") {
      source.close();
      finish(jobId, state);
      return;
    }
    if (!items[stage]) {
      items[stage] = document.createElement("li");
      progressList.appendChild(items[stage]);
    }
    items[stage].textContent = stage;
    items[stage].className = state;
  };
  source.onerror = () => { source.close(); finish(jobId, "done"); };
}

async function finish(jobId, state) {
  submitBtn.disabled = false;
  const job = await (await fetch(`/api/jobs/${jobId}`)).json();
  if (job.status !== "done") {
    alert("Job " + job.status + (job.error ? ": " + job.error : ""));
    return;
  }
  results.hidden = false;
  document.getElementById("preview").src =
    `/api/jobs/${jobId}/artifacts/preview.mp4`;
  const report = job.report || {};
  const music = report.music || {};
  const captions = report.captions || {};
  const esc = (s) => String(s ?? "-").replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  let html = `<p>${esc((report.shots || {}).count)} shots</p>`;
  html += music.detected
    ? `<p>Music: <strong>${esc(music.title)}</strong> by ${esc(music.artist)}
       (via ${esc(music.identified_by)})</p>`
    : "<p>No music detected.</p>";
  html += captions.count
    ? `<p>${captions.count} caption(s), font: ${esc((captions.fonts || []).join(", ") || "uncertain")}</p>`
    : "<p>No captions.</p>";
  const sources = report.sources || {};
  if (sources.searched && (sources.shots || []).length) {
    const links = sources.shots.flatMap((s) => s.candidates || [])
      .slice(0, 5)
      .map((c) => `<a href="${esc(c.url)}" target="_blank">${esc(c.title || c.url)}</a>`)
      .join(" · ");
    html += `<p>Source candidates: ${links}</p>`;
  }
  document.getElementById("summary").innerHTML = html;
  document.getElementById("downloads").innerHTML =
    `<a href="/api/jobs/${jobId}/artifacts/preview.mp4" download>Preview MP4</a>
     <a href="/api/jobs/${jobId}/artifacts/premiere_resolve.zip" download>Premiere/Resolve project</a>
     <a href="/api/jobs/${jobId}/artifacts/report.html" target="_blank">Report</a>
     <a href="/api/jobs/${jobId}/artifacts/manifest.json" download>Manifest</a>`;
}
