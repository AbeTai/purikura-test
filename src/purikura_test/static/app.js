const statusEl = document.querySelector("#status");
const cameraSelect = document.querySelector("#camera-select");
const frameSelect = document.querySelector("#frame-select");
const frameUpload = document.querySelector("#frame-upload");
const captureButton = document.querySelector("#capture-button");
const capturesEl = document.querySelector("#captures");

const controls = {
  processing_profile: document.querySelector("#processing-profile"),
  skin_smoothing: document.querySelector("#skin"),
  purikura_intensity: document.querySelector("#purikura"),
  skin_whitening: document.querySelector("#skin-whitening"),
  eye_enlarge: document.querySelector("#eye-enlarge"),
  face_slim: document.querySelector("#face-slim"),
  eye_sparkle: document.querySelector("#eye-sparkle"),
  lip_tint: document.querySelector("#lip-tint"),
  blush: document.querySelector("#blush"),
  brightness: document.querySelector("#brightness"),
  contrast: document.querySelector("#contrast"),
  saturation: document.querySelector("#saturation"),
  doll_intensity: document.querySelector("#doll-intensity"),
  porcelain_skin: document.querySelector("#porcelain-skin"),
  eye_roundness: document.querySelector("#eye-roundness"),
  eye_liner: document.querySelector("#eye-liner"),
  lash_emphasis: document.querySelector("#lash-emphasis"),
  lower_eyelid: document.querySelector("#lower-eyelid"),
  iris_gloss: document.querySelector("#iris-gloss"),
  cheek_gradient: document.querySelector("#cheek-gradient"),
  lip_gloss: document.querySelector("#lip-gloss"),
  hair_silk: document.querySelector("#hair-silk"),
  background_high_key: document.querySelector("#background-high-key"),
  soft_glow: document.querySelector("#soft-glow"),
  debug_overlay: document.querySelector("#debug-overlay"),
};

function setStatus(message) {
  statusEl.textContent = message;
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
    skin_whitening: Number(controls.skin_whitening.value),
    eye_enlarge: Number(controls.eye_enlarge.value),
    face_slim: Number(controls.face_slim.value),
    eye_sparkle: Number(controls.eye_sparkle.value),
    lip_tint: Number(controls.lip_tint.value),
    blush: Number(controls.blush.value),
    brightness: Number(controls.brightness.value),
    contrast: Number(controls.contrast.value),
    saturation: Number(controls.saturation.value),
    doll_intensity: Number(controls.doll_intensity.value),
    porcelain_skin: Number(controls.porcelain_skin.value),
    eye_roundness: Number(controls.eye_roundness.value),
    eye_liner: Number(controls.eye_liner.value),
    lash_emphasis: Number(controls.lash_emphasis.value),
    lower_eyelid: Number(controls.lower_eyelid.value),
    iris_gloss: Number(controls.iris_gloss.value),
    cheek_gradient: Number(controls.cheek_gradient.value),
    lip_gloss: Number(controls.lip_gloss.value),
    hair_silk: Number(controls.hair_silk.value),
    background_high_key: Number(controls.background_high_key.value),
    soft_glow: Number(controls.soft_glow.value),
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
  await Promise.all([loadCameras(), loadEffects(), loadFrames(), loadCaptures()]);
  setStatus("Ready");
} catch (error) {
  setStatus(error.message);
}
