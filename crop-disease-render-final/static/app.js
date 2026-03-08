const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCamBtn = document.getElementById('startCam');
const snapBtn = document.getElementById('snap');
const toggleBtn = document.getElementById('toggle');
const intervalInput = document.getElementById('interval');
const modelSel = document.getElementById('model');
const statusEl = document.getElementById('status');
const topLabelEl = document.getElementById('topLabel');
const confEl = document.getElementById('conf');
const topList = document.getElementById('topList');

let stream = null;
let running = false;
let busy = false;

function setStatus(s){ statusEl.textContent = s; }

async function startCamera(){
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
  video.srcObject = stream;
  await video.play();
  setStatus('Camera active');
}

function frameToBlob(){
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
}

function renderResult(res){
  topLabelEl.textContent = res.top_label || '—';
  confEl.textContent = `${((res.confidence || 0) * 100).toFixed(2)}%`;
  topList.innerHTML = '';
  (res.top_predictions || []).slice(0,3).forEach(item => {
    const li = document.createElement('li');
    li.textContent = `${item.label} — ${(item.score * 100).toFixed(2)}%`;
    topList.appendChild(li);
  });
}

async function predictOnce(){
  if (busy) return;
  if (!stream) await startCamera();
  busy = true;
  setStatus('Predicting...');
  try {
    const blob = await frameToBlob();
    const form = new FormData();
    form.append('image', blob, 'frame.jpg');
    form.append('model_name', modelSel.value);
    const res = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');
    renderResult(data);
    setStatus('Prediction updated');
  } catch (e) {
    setStatus(e.message || 'Prediction failed');
  } finally {
    busy = false;
  }
}

async function loop(){
  while (running) {
    const start = performance.now();
    await predictOnce();
    const wait = Math.max(0, Number(intervalInput.value || 1500) - (performance.now() - start));
    await new Promise(r => setTimeout(r, wait));
  }
}

startCamBtn.onclick = () => startCamera().catch(e => setStatus(e.message));
snapBtn.onclick = () => predictOnce();
toggleBtn.onclick = async () => {
  running = !running;
  toggleBtn.textContent = running ? 'Stop realtime' : 'Start realtime';
  if (running) {
    await startCamera().catch(() => {});
    loop();
  }
};
