const statusEl = document.querySelector("#status");
const cameraSelect = document.querySelector("#camera-select");
const frameSelect = document.querySelector("#frame-select");
const frameUpload = document.querySelector("#frame-upload");
const captureButton = document.querySelector("#capture-button");
const capturesEl = document.querySelector("#captures");
const performanceEl = document.querySelector("#performance");

const controls = {
  processing_profile: document.querySelector("#processing-profile"),
  skin_smoothing: document.querySelector("#skin"),
  purikura_intensity: document.querySelector("#purikura"),
  eye_enlarge: document.querySelector("#eye-enlarge"),
  face_slim: document.querySelector("#face-slim"),
  doll_intensity: document.querySelector("#doll-intensity"),
  background_high_key: document.querySelector("#background-high-key"),
  debug_overlay: document.querySelector("#debug-overlay"),
};

function setStatus(message) {
  statusEl.textContent = message;
}

function formatMs(value) {
  return `${Number(value || 0).toFixed(0)}ms`;
}

function renderPerformance(performance) {
  const metrics = [
    ["Profile", performance.profile],
    ["Process", formatMs(performance.processing_ms)],
    ["Encode", formatMs(performance.encode_ms)],
    ["FPS", Number(performance.effective_fps || 0).toFixed(1)],
    ["Frame age", formatMs(performance.frame_age_ms)],
    ["Landmark age", formatMs(performance.landmark_age_ms)],
    ["Mask age", formatMs(performance.mask_age_ms)],
    ["Publish gap", formatMs(performance.publish_interval_ms)],
    ["Stall", formatMs(performance.preview_stall_ms)],
    ["Lag frames", String(performance.publish_lag_frames || 0)],
    ["Dropped", String(performance.dropped_frames || 0)],
    ["Discarded", String(performance.discarded_processed_frames || 0)],
  ];
  performanceEl.replaceChildren(
    ...metrics.map(([label, value]) => {
      const item = document.createElement("div");
      item.className = "metric";
      const name = document.createElement("span");
      name.textContent = label;
      const metricValue = document.createElement("strong");
      metricValue.textContent = value;
      item.append(name, metricValue);
      return item;
    }),
  );
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadCameras() {
  const cameras = await requestJson("/api/cameras");
  cameraSelect.replaceChildren();
  for (const camera of cameras) {
    const option = document.createElement("option");
    option.value = String(camera.id);
    option.textContent = `${camera.name}${camera.available ? "" : " (unavailable)"}`;
    option.disabled = !camera.available;
    cameraSelect.append(option);
  }
}

async function loadEffects() {
  const settings = await requestJson("/api/effects");
  for (const [key, input] of Object.entries(controls)) {
    if (input.type === "checkbox") {
      input.checked = settings[key];
    } else {
      input.value = settings[key];
    }
  }
}

async function saveEffects() {
  const body = {
    processing_profile: controls.processing_profile.value,
    skin_smoothing: Number(controls.skin_smoothing.value),
    purikura_intensity: Number(controls.purikura_intensity.value),
    eye_enlarge: Number(controls.eye_enlarge.value),
    face_slim: Number(controls.face_slim.value),
    doll_intensity: Number(controls.doll_intensity.value),
    background_high_key: Number(controls.background_high_key.value),
    debug_overlay: controls.debug_overlay.value,
  };
  await requestJson("/api/effects", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function loadFrames() {
  const frames = await requestJson("/api/frames");
  const current = frameSelect.value;
  frameSelect.replaceChildren(new Option("No frame", ""));
  for (const frame of frames) {
    frameSelect.append(new Option(frame.name, String(frame.id)));
  }
  frameSelect.value = current;
}

async function loadCaptures() {
  const captures = await requestJson("/api/captures");
  capturesEl.replaceChildren();
  for (const capture of captures) {
    const item = document.createElement("a");
    item.className = "capture-thumb";
    item.href = `/api/captures/${capture.id}/image`;
    item.target = "_blank";

    const image = document.createElement("img");
    image.src = `/api/captures/${capture.id}/image?ts=${Date.now()}`;
    image.alt = `Capture ${capture.id}`;

    const label = document.createElement("span");
    label.textContent = new Date(capture.created_at).toLocaleString();

    item.append(image, label);
    capturesEl.append(item);
  }
}

async function loadPerformance() {
  const performance = await requestJson("/api/performance");
  renderPerformance(performance);
}

for (const input of Object.values(controls)) {
  const eventName = input.tagName === "SELECT" ? "change" : "input";
  input.addEventListener(eventName, async () => {
    try {
      await saveEffects();
      setStatus("Effects updated");
    } catch (error) {
      setStatus(error.message);
    }
  });
}

cameraSelect.addEventListener("change", async () => {
  try {
    await requestJson("/api/camera", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: Number(cameraSelect.value) }),
    });
    setStatus(`Camera ${cameraSelect.value} selected`);
  } catch (error) {
    setStatus(error.message);
  }
});

frameUpload.addEventListener("change", async () => {
  const file = frameUpload.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  try {
    await requestJson("/api/frames", { method: "POST", body: formData });
    await loadFrames();
    setStatus("Frame uploaded");
  } catch (error) {
    setStatus(error.message);
  }
});

frameSelect.addEventListener("change", async () => {
  try {
    await requestJson("/api/frame/current", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame_id: frameSelect.value ? Number(frameSelect.value) : null }),
    });
    setStatus(frameSelect.value ? "Frame selected" : "Frame cleared");
  } catch (error) {
    setStatus(error.message);
  }
});

captureButton.addEventListener("click", async () => {
  captureButton.disabled = true;
  try {
    const result = await requestJson("/api/captures", { method: "POST" });
    setStatus(`Captured #${result.id}`);
    await loadCaptures();
  } catch (error) {
    setStatus(error.message);
  } finally {
    captureButton.disabled = false;
  }
});

try {
  await Promise.all([loadCameras(), loadEffects(), loadFrames(), loadCaptures(), loadPerformance()]);
  window.setInterval(() => {
    loadPerformance().catch(() => {});
  }, 1000);
  setStatus("Ready");
} catch (error) {
  setStatus(error.message);
}
