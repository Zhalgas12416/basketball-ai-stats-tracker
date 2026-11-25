const state = {
  gameId: null,
  result: null,
  health: null,
  quality: null,
  pollTimer: null,
  overlayFrameRequest: null,
  corrections: [],
  hardFrames: [],
  activeHardFrameIndex: null,
  draftCorrection: null,
  isDrawingCorrection: false,
};

const elements = {
  uploadForm: document.querySelector("#uploadForm"),
  videoInput: document.querySelector("#videoInput"),
  modelStatus: document.querySelector("#modelStatus"),
  modelDetails: document.querySelector("#modelDetails"),
  statusText: document.querySelector("#statusText"),
  playersCount: document.querySelector("#playersCount"),
  eventsCount: document.querySelector("#eventsCount"),
  tracksCount: document.querySelector("#tracksCount"),
  detectionsCount: document.querySelector("#detectionsCount"),
  durationText: document.querySelector("#durationText"),
  detectorSummary: document.querySelector("#detectorSummary"),
  warningList: document.querySelector("#warningList"),
  qualitySummary: document.querySelector("#qualitySummary"),
  qualityClassCards: document.querySelector("#qualityClassCards"),
  skippedFramesText: document.querySelector("#skippedFramesText"),
  lowConfidenceList: document.querySelector("#lowConfidenceList"),
  overlayVideo: document.querySelector("#overlayVideo"),
  videoStage: document.querySelector("#videoStage"),
  overlayCanvas: document.querySelector("#overlayCanvas"),
  overlayEmpty: document.querySelector("#overlayEmpty"),
  overlayStatus: document.querySelector("#overlayStatus"),
  overlaySource: document.querySelector("#overlaySource"),
  showOverlay: document.querySelector("#showOverlay"),
  labelMode: document.querySelector("#labelMode"),
  labelClass: document.querySelector("#labelClass"),
  sampleHardFrames: document.querySelector("#sampleHardFrames"),
  undoCorrection: document.querySelector("#undoCorrection"),
  exportCorrections: document.querySelector("#exportCorrections"),
  buildCorrectionDataset: document.querySelector("#buildCorrectionDataset"),
  correctionStatus: document.querySelector("#correctionStatus"),
  hardFrameStatus: document.querySelector("#hardFrameStatus"),
  hardFrameList: document.querySelector("#hardFrameList"),
  statsBody: document.querySelector("#statsBody"),
  tracksBody: document.querySelector("#tracksBody"),
  eventList: document.querySelector("#eventList"),
  eventFilter: document.querySelector("#eventFilter"),
  exportJson: document.querySelector("#exportJson"),
};

loadHealth();

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = elements.videoInput.files[0];
  if (!file) {
    setStatus("Choose a video file");
    return;
  }

  const formData = new FormData();
  formData.append("video", file);

  setStatus("Uploading");
  elements.exportJson.disabled = true;
  clearResult();

  const response = await fetch("/api/games", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    setStatus("Upload failed");
    return;
  }

  const payload = await response.json();
  state.gameId = payload.job.game_id;
  setStatus(payload.job.status);
  startPolling();
});

elements.eventFilter.addEventListener("change", () => {
  renderEvents(state.result?.events ?? []);
});

elements.overlayVideo.addEventListener("loadedmetadata", () => {
  resizeOverlayCanvas();
  renderOverlay();
});
elements.overlayVideo.addEventListener("error", () => {
  elements.overlayStatus.textContent = "The browser could not play this video file. Try exporting the clip as MP4/H.264.";
});
elements.overlayVideo.addEventListener("timeupdate", renderOverlay);
elements.overlayVideo.addEventListener("seeked", renderOverlay);
elements.overlayVideo.addEventListener("play", startOverlayLoop);
elements.overlayVideo.addEventListener("pause", stopOverlayLoop);
elements.overlayVideo.addEventListener("ended", stopOverlayLoop);
elements.overlaySource.addEventListener("change", renderOverlay);
elements.showOverlay.addEventListener("change", renderOverlay);
elements.labelMode.addEventListener("change", () => {
  elements.videoStage.classList.toggle("labeling", elements.labelMode.checked);
  updateCorrectionStatus();
  renderOverlay();
});
elements.overlayCanvas.addEventListener("pointerdown", startCorrectionDraft);
elements.overlayCanvas.addEventListener("pointermove", updateCorrectionDraft);
elements.overlayCanvas.addEventListener("pointerup", finishCorrectionDraft);
elements.overlayCanvas.addEventListener("pointercancel", cancelCorrectionDraft);
elements.sampleHardFrames.addEventListener("click", sampleHardFrames);
elements.undoCorrection.addEventListener("click", undoLastCorrection);
elements.exportCorrections.addEventListener("click", exportCorrections);
elements.buildCorrectionDataset.addEventListener("click", buildCorrectionDataset);
window.addEventListener("resize", () => {
  resizeOverlayCanvas();
  renderOverlay();
});

elements.exportJson.addEventListener("click", () => {
  if (!state.result) {
    return;
  }
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.result.game_id}-stats.json`;
  link.click();
  URL.revokeObjectURL(url);
});

function startPolling() {
  window.clearInterval(state.pollTimer);
  state.pollTimer = window.setInterval(fetchGame, 1200);
  fetchGame();
}

async function fetchGame() {
  if (!state.gameId) {
    return;
  }

  const response = await fetch(`/api/games/${state.gameId}`);
  if (!response.ok) {
    setStatus("Analysis unavailable");
    window.clearInterval(state.pollTimer);
    return;
  }

  const payload = await response.json();
  setStatus(payload.job.status);

  if (payload.job.status === "completed") {
    window.clearInterval(state.pollTimer);
    state.result = payload.result;
    renderResult(payload.result);
  }

  if (payload.job.status === "failed") {
    window.clearInterval(state.pollTimer);
    setStatus(payload.job.error || "Analysis failed");
  }
}

function renderResult(result) {
  elements.playersCount.textContent = String(result.players.length);
  elements.eventsCount.textContent = String(result.events.length);
  elements.tracksCount.textContent = String(result.vision?.track_count ?? 0);
  elements.detectionsCount.textContent = String(result.vision?.detection_count ?? 0);
  elements.durationText.textContent = result.duration_timecode;
  elements.exportJson.disabled = false;
  renderDetectorInfo(result.vision?.detector, result.vision, result.warnings ?? []);
  setupOverlay(result);
  renderStats(result.stats);
  renderEvents(result.events);
  renderTracks(result.vision?.tracks ?? []);
  loadQualityReport();
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("Health check failed");
    }
    state.health = await response.json();
    renderDetectorInfo(
      state.health?.analyzer?.detector,
      null,
      state.health?.analyzer?.warnings ?? [],
    );
  } catch {
    elements.modelStatus.textContent = "Detector unknown";
    elements.modelDetails.textContent = "Could not read /api/health.";
    elements.detectorSummary.textContent = "Upload still works, but detector status is unavailable.";
  }
}

function renderStats(stats) {
  elements.statsBody.innerHTML = "";

  if (!stats.length) {
    elements.statsBody.innerHTML = '<tr><td colspan="10" class="empty-cell">No player stats detected.</td></tr>';
    return;
  }

  for (const row of stats) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="player-cell">${escapeHtml(row.player.label)}</td>
      <td>${row.points}</td>
      <td>${row.assists}</td>
      <td>${row.rebounds}</td>
      <td>${row.steals}</td>
      <td>${row.turnovers}</td>
      <td>${row.blocks}</td>
      <td>${row.field_goals_made}/${row.field_goals_attempted}</td>
      <td>${row.three_points_made}/${row.three_points_attempted}</td>
      <td>${row.free_throws_made}/${row.free_throws_attempted}</td>
    `;
    elements.statsBody.appendChild(tr);
  }
}

function renderTracks(tracks) {
  elements.tracksBody.innerHTML = "";

  if (!tracks.length) {
    elements.tracksBody.innerHTML = '<tr><td colspan="5" class="empty-cell">No tracks detected.</td></tr>';
    return;
  }

  for (const track of tracks) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="player-cell">${escapeHtml(track.track_id)}</td>
      <td>${prettyEvent(track.label)}</td>
      <td>${formatSeconds(track.first_seen_seconds)}-${formatSeconds(track.last_seen_seconds)}</td>
      <td>${track.observation_count}</td>
      <td>${Math.round(track.max_confidence * 100)}%</td>
    `;
    elements.tracksBody.appendChild(tr);
  }
}

function renderEvents(events) {
  const filter = elements.eventFilter.value;
  const filteredEvents = filter === "all" ? events : events.filter((event) => event.event_type === filter);
  elements.eventList.innerHTML = "";

  if (!filteredEvents.length) {
    const empty = document.createElement("li");
    empty.innerHTML = '<span class="timecode">0:00</span><div><p class="event-title">No events</p><p class="event-meta">Change the filter or upload a game video.</p></div>';
    elements.eventList.appendChild(empty);
    return;
  }

  const playerLookup = new Map((state.result?.players ?? []).map((player) => [player.player_id, player.label]));
  for (const event of filteredEvents) {
    const playerLabel = playerLookup.get(event.player_id) || event.player_id;
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="timecode">${escapeHtml(event.timecode)}</span>
      <div>
        <p class="event-title">${prettyEvent(event.event_type)} by ${escapeHtml(playerLabel)}</p>
        <p class="event-meta">${escapeHtml(event.notes || "")}</p>
      </div>
    `;
    elements.eventList.appendChild(li);
  }
}

function clearResult() {
  state.result = null;
  state.quality = null;
  state.corrections = [];
  state.hardFrames = [];
  state.activeHardFrameIndex = null;
  state.draftCorrection = null;
  state.isDrawingCorrection = false;
  stopOverlayLoop();
  elements.overlayVideo.removeAttribute("src");
  elements.overlayVideo.load();
  elements.overlayEmpty.classList.remove("hidden");
  elements.videoStage.classList.remove("labeling");
  elements.labelMode.checked = false;
  elements.undoCorrection.disabled = true;
  elements.exportCorrections.disabled = true;
  elements.sampleHardFrames.disabled = true;
  elements.buildCorrectionDataset.disabled = true;
  clearOverlayCanvas();
  elements.playersCount.textContent = "0";
  elements.eventsCount.textContent = "0";
  elements.tracksCount.textContent = "0";
  elements.detectionsCount.textContent = "0";
  elements.durationText.textContent = "0:00";
  elements.detectorSummary.textContent = "Analysis running. Detector tracks will appear when processing completes.";
  elements.qualitySummary.textContent = "Analysis running. Quality report will appear when processing completes.";
  elements.qualityClassCards.innerHTML = "";
  elements.skippedFramesText.textContent = "No sampled frames yet.";
  elements.lowConfidenceList.innerHTML = "";
  elements.overlayStatus.textContent = "Analysis running. Overlay will load when processing completes.";
  elements.correctionStatus.textContent = "Turn on Label, choose a class, then drag on the video to save a correction.";
  elements.hardFrameStatus.textContent = "Run hard-frame sampling after analysis to jump to weak detector moments.";
  elements.hardFrameList.innerHTML = "";
  renderWarnings([]);
  elements.statsBody.innerHTML = '<tr><td colspan="10" class="empty-cell">Analysis running.</td></tr>';
  elements.tracksBody.innerHTML = '<tr><td colspan="5" class="empty-cell">Analysis running.</td></tr>';
  elements.eventList.innerHTML = "";
}

async function loadQualityReport() {
  if (!state.gameId) {
    return;
  }

  try {
    const response = await fetch(`/api/games/${state.gameId}/quality`);
    if (!response.ok) {
      throw new Error("Quality report unavailable");
    }
    const payload = await response.json();
    state.quality = payload.quality;
    renderQualityReport(payload.quality);
  } catch {
    const fallback = buildQualityReport(state.result?.vision || {});
    state.quality = fallback;
    renderQualityReport(fallback);
  }
}

function renderQualityReport(report) {
  if (!report) {
    return;
  }

  const emptyRate = Math.round(Number(report.empty_sampled_frame_rate || 0) * 100);
  elements.qualitySummary.textContent = `${report.detection_count} detections across ${report.detected_frame_count}/${report.sampled_frame_count} sampled frames · ${emptyRate}% empty sampled frames`;
  elements.qualityClassCards.innerHTML = "";

  for (const label of report.target_classes || ["player", "ball", "rim", "backboard", "referee"]) {
    const count = Number(report.class_counts?.[label] || 0);
    const low = Number(report.low_confidence_by_class?.[label] || 0);
    const average = report.average_confidence_by_class?.[label];
    const card = document.createElement("article");
    card.className = `quality-card quality-${label}`;
    card.innerHTML = `
      <span>${prettyEvent(label)}</span>
      <strong>${count}</strong>
      <small>${low} low-conf · avg ${average == null ? "n/a" : `${Math.round(Number(average) * 100)}%`}</small>
    `;
    elements.qualityClassCards.appendChild(card);
  }

  const skippedNumbers = report.skipped_frame_numbers || [];
  const skippedPreview = skippedNumbers.length ? ` Examples: ${skippedNumbers.slice(0, 18).join(", ")}${skippedNumbers.length > 18 ? ", ..." : ""}` : "";
  elements.skippedFramesText.textContent = `${report.empty_sampled_frame_count} sampled frames had no detections. ${report.skipped_frame_note || ""}${skippedPreview}`;
  renderLowConfidence(report.low_confidence_detections || []);
}

function renderLowConfidence(detections) {
  elements.lowConfidenceList.innerHTML = "";
  if (!detections.length) {
    const item = document.createElement("li");
    item.textContent = "No low-confidence detections at the current threshold.";
    elements.lowConfidenceList.appendChild(item);
    return;
  }

  for (const detection of detections.slice(0, 8)) {
    const item = document.createElement("li");
    const frame = detection.frame_number == null ? "frame ?" : `frame ${detection.frame_number}`;
    item.innerHTML = `
      <span>${formatSeconds(detection.timestamp_seconds)}</span>
      <strong>${prettyEvent(detection.class_name)}</strong>
      <small>${frame} · ${Math.round(Number(detection.confidence || 0) * 100)}%</small>
    `;
    elements.lowConfidenceList.appendChild(item);
  }
}

function buildQualityReport(vision) {
  const targetClasses = ["player", "ball", "rim", "backboard", "referee"];
  const classCounts = Object.fromEntries(targetClasses.map((label) => [label, 0]));
  const lowConfidenceByClass = Object.fromEntries(targetClasses.map((label) => [label, 0]));
  const confidenceValues = Object.fromEntries(targetClasses.map((label) => [label, []]));
  const detectedFrames = new Set();
  const lowConfidenceDetections = [];
  const threshold = 0.3;

  for (const detection of vision.detections || []) {
    const label = detection.label;
    const confidence = Number(detection.confidence || 0);
    if (Number.isFinite(Number(detection.frame_index))) {
      detectedFrames.add(Number(detection.frame_index));
    }
    if (label in classCounts) {
      classCounts[label] += 1;
      confidenceValues[label].push(confidence);
      if (confidence < threshold) {
        lowConfidenceByClass[label] += 1;
        lowConfidenceDetections.push({
          frame_number: detection.frame_index,
          timestamp_seconds: detection.time_seconds,
          class_name: label,
          confidence,
          bbox: detection.bbox,
        });
      }
    }
  }

  const sampledFrameCount = Number(vision.sampled_frame_count || 0);
  const emptySampledFrameCount = Math.max(0, sampledFrameCount - detectedFrames.size);
  return {
    target_classes: targetClasses,
    low_confidence_threshold: threshold,
    detection_count: (vision.detections || []).length,
    sampled_frame_count: sampledFrameCount,
    detected_frame_count: detectedFrames.size,
    empty_sampled_frame_count: emptySampledFrameCount,
    empty_sampled_frame_rate: sampledFrameCount ? emptySampledFrameCount / sampledFrameCount : 0,
    skipped_frame_numbers: [],
    skipped_frame_note: "Skipped count is exact for sampled frames; exact skipped frame numbers need sampled-frame logging in the detector.",
    class_counts: classCounts,
    low_confidence_count: Object.values(lowConfidenceByClass).reduce((total, count) => total + count, 0),
    low_confidence_by_class: lowConfidenceByClass,
    average_confidence_by_class: Object.fromEntries(
      targetClasses.map((label) => {
        const values = confidenceValues[label];
        return [label, values.length ? values.reduce((total, value) => total + value, 0) / values.length : null];
      }),
    ),
    low_confidence_detections: lowConfidenceDetections.sort((a, b) => a.confidence - b.confidence),
  };
}

function setStatus(text) {
  elements.statusText.textContent = text;
}

function setupOverlay(result) {
  if (!state.gameId || !result?.vision) {
    return;
  }
  stopOverlayLoop();
  elements.overlayVideo.src = `/api/games/${state.gameId}/video`;
  elements.overlayEmpty.classList.add("hidden");
  elements.overlayStatus.textContent = "Video loaded. Press play or scrub to inspect detection boxes.";
  elements.sampleHardFrames.disabled = false;
  elements.overlayVideo.load();
  loadCorrections();
  resizeOverlayCanvas();
  renderOverlay();
}

async function loadCorrections() {
  if (!state.gameId) {
    return;
  }

  try {
    const response = await fetch(`/api/games/${state.gameId}/corrections`);
    if (!response.ok) {
      throw new Error("Correction fetch failed");
    }
    const payload = await response.json();
    state.corrections = payload.corrections || [];
  } catch {
    state.corrections = [];
    elements.correctionStatus.textContent = "Could not load saved corrections for this upload.";
  }
  updateCorrectionStatus();
  renderHardFrames();
  renderOverlay();
}

async function sampleHardFrames() {
  if (!state.gameId || !state.result?.vision) {
    return;
  }

  elements.sampleHardFrames.disabled = true;
  elements.hardFrameStatus.textContent = "Finding weak detector moments...";
  elements.hardFrameList.innerHTML = "";
  try {
    const response = await fetch(`/api/games/${state.gameId}/hard-frames`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classes: ["ball", "referee", "rim", "backboard"],
        max_frames: 40,
        sample_interval_seconds: 0.5,
        min_spacing_seconds: 1.0,
        low_confidence_threshold: 0.35,
      }),
    });
    if (!response.ok) {
      throw new Error("Hard-frame sampling failed");
    }
    const payload = await response.json();
    state.hardFrames = payload.frames || [];
    state.activeHardFrameIndex = state.hardFrames.length ? 0 : null;
    renderHardFrames();
    if (state.hardFrames.length) {
      jumpToHardFrame(0);
    }
  } catch {
    elements.hardFrameStatus.textContent = "Could not sample hard frames for this game.";
  } finally {
    elements.sampleHardFrames.disabled = false;
  }
}

function renderHardFrames() {
  const frames = state.hardFrames || [];
  elements.hardFrameList.innerHTML = "";
  if (!frames.length) {
    elements.hardFrameStatus.textContent = state.gameId
      ? "No hard-frame queue yet. Click Find hard frames after analysis."
      : "Run hard-frame sampling after analysis to jump to weak detector moments.";
    return;
  }

  elements.hardFrameStatus.textContent = `${frames.length} hard frame${frames.length === 1 ? "" : "s"} queued. Click one, turn on Label, and draw missed boxes.`;
  frames.forEach((frame, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === state.activeHardFrameIndex ? "active" : "";
    button.addEventListener("click", () => jumpToHardFrame(index));
    const savedCount = correctionsNearTime(Number(frame.timestamp_seconds), 0.55).length;
    const fixedText = savedCount ? `<small>${savedCount} saved fix${savedCount === 1 ? "" : "es"}</small>` : "<small>needs labels</small>";
    button.innerHTML = `
      <span>${formatSeconds(frame.timestamp_seconds)}</span>
      <strong>${escapeHtml((frame.reasons || []).slice(0, 3).map(prettyEvent).join(", "))}</strong>
      ${fixedText}
    `;
    item.appendChild(button);
    elements.hardFrameList.appendChild(item);
  });
}

function jumpToHardFrame(index) {
  const frame = state.hardFrames[index];
  if (!frame) {
    return;
  }
  state.activeHardFrameIndex = index;
  elements.overlayVideo.pause();
  elements.overlayVideo.currentTime = Math.max(0, Number(frame.timestamp_seconds || 0));
  elements.labelMode.checked = true;
  elements.videoStage.classList.add("labeling");
  elements.overlaySource.value = "detections";
  updateCorrectionStatus(`Hard frame ${index + 1}/${state.hardFrames.length}: draw missing ball/referee/rim/backboard boxes.`);
  renderHardFrames();
  renderOverlay();
}

function startOverlayLoop() {
  stopOverlayLoop();
  const tick = () => {
    renderOverlay();
    state.overlayFrameRequest = window.requestAnimationFrame(tick);
  };
  state.overlayFrameRequest = window.requestAnimationFrame(tick);
}

function stopOverlayLoop() {
  if (state.overlayFrameRequest) {
    window.cancelAnimationFrame(state.overlayFrameRequest);
    state.overlayFrameRequest = null;
  }
}

function resizeOverlayCanvas() {
  const canvas = elements.overlayCanvas;
  const video = elements.overlayVideo;
  const rect = video.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * pixelRatio));
  const height = Math.max(1, Math.round(rect.height * pixelRatio));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function renderOverlay() {
  resizeOverlayCanvas();
  const canvas = elements.overlayCanvas;
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.result?.vision || !elements.overlayVideo.src) {
    return;
  }

  const currentTime = elements.overlayVideo.currentTime || 0;
  const boxes = elements.showOverlay.checked ? (elements.overlaySource.value === "detections"
    ? activeDetections(state.result.vision, currentTime)
    : activeTrackPoints(state.result.vision, currentTime)) : [];
  const correctionBoxes = activeCorrections(currentTime);
  drawBoxes(context, boxes, state.result.vision);
  drawBoxes(context, correctionBoxes, state.result.vision, { dashed: true, titlePrefix: "fix" });
  if (state.draftCorrection) {
    drawBoxes(context, [state.draftCorrection], state.result.vision, { dashed: true, titlePrefix: "new" });
  }
  updateOverlayStatus(boxes, correctionBoxes, currentTime);
}

function clearOverlayCanvas() {
  const context = elements.overlayCanvas.getContext("2d");
  if (context) {
    context.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
  }
}

function activeTrackPoints(vision, currentTime) {
  const tolerance = overlayTolerance(vision);
  const boxes = [];
  for (const track of vision.tracks || []) {
    const point = nearestPoint(track.points || [], currentTime, tolerance.track);
    if (!point) {
      continue;
    }
    boxes.push({
      label: track.label,
      title: track.track_id,
      confidence: point.confidence,
      bbox: point.bbox,
    });
  }
  return boxes;
}

function activeDetections(vision, currentTime) {
  const tolerance = overlayTolerance(vision);
  const detections = vision.detections || [];
  const nearestTime = nearestDetectionTime(detections, currentTime, tolerance.detection);
  if (nearestTime === null) {
    return [];
  }
  return detections
    .filter((detection) => Math.abs(Number(detection.time_seconds) - nearestTime) <= 0.001)
    .slice(0, 120)
    .map((detection) => ({
      label: detection.label,
      title: prettyEvent(detection.label),
      confidence: detection.confidence,
      bbox: detection.bbox,
    }));
}

function activeCorrections(currentTime) {
  const tolerance = overlayTolerance(state.result?.vision || {});
  return (state.corrections || [])
    .filter((correction) => Math.abs(Number(correction.video_time_seconds) - currentTime) <= tolerance.correction)
    .map((correction) => ({
      label: correction.label,
      title: correction.label,
      confidence: 1,
      bbox: correction.bbox,
    }));
}

function nearestDetectionTime(detections, currentTime, tolerance) {
  let bestTime = null;
  let bestDelta = Infinity;
  for (const detection of detections) {
    const time = Number(detection.time_seconds);
    const delta = Math.abs(time - currentTime);
    if (delta < bestDelta) {
      bestTime = time;
      bestDelta = delta;
    }
  }
  return bestDelta <= tolerance ? bestTime : null;
}

function nearestPoint(points, currentTime, tolerance) {
  let best = null;
  let bestDelta = Infinity;
  for (const point of points) {
    const delta = Math.abs(Number(point.time_seconds) - currentTime);
    if (delta < bestDelta) {
      best = point;
      bestDelta = delta;
    }
  }
  return bestDelta <= tolerance ? best : null;
}

function overlayTolerance(vision) {
  const detector = vision.detector || {};
  const primaryFps = Number(detector.sample_fps || detector.primary?.sample_fps || 0);
  const fallbackFps = Number(detector.fallback?.sample_fps || 0);
  const sampleFps = Math.max(primaryFps, fallbackFps, 4);
  const base = Math.max(0.22, 1.8 / sampleFps);
  return {
    track: base,
    detection: Math.max(base, 0.45),
    correction: Math.max(base, 0.45),
  };
}

function drawBoxes(context, boxes, vision, options = {}) {
  const canvas = elements.overlayCanvas;
  const videoWidth = Number(vision.width || elements.overlayVideo.videoWidth || 1);
  const videoHeight = Number(vision.height || elements.overlayVideo.videoHeight || 1);
  const scaleX = canvas.width / videoWidth;
  const scaleY = canvas.height / videoHeight;
  context.lineWidth = Math.max(2, canvas.width / 700);
  context.font = `${Math.max(12, Math.round(canvas.width / 74))}px Avenir Next, sans-serif`;
  context.textBaseline = "top";

  for (const box of boxes) {
    const color = overlayColor(box.label);
    const x = Number(box.bbox.x) * scaleX;
    const y = Number(box.bbox.y) * scaleY;
    const width = Number(box.bbox.width) * scaleX;
    const height = Number(box.bbox.height) * scaleY;
    const title = options.titlePrefix
      ? `${options.titlePrefix}: ${prettyEvent(box.title || box.label)}`
      : `${box.title} ${Math.round(Number(box.confidence || 0) * 100)}%`;

    context.strokeStyle = color;
    context.fillStyle = color;
    context.setLineDash(options.dashed ? [10, 7] : []);
    context.strokeRect(x, y, width, height);
    context.setLineDash([]);

    const textWidth = context.measureText(title).width + 12;
    const labelHeight = Math.max(20, canvas.width / 48);
    const labelY = Math.max(0, y - labelHeight);
    context.fillRect(x, labelY, textWidth, labelHeight);
    context.fillStyle = "#12140f";
    context.fillText(title, x + 6, labelY + 4);
  }
}

function startCorrectionDraft(event) {
  if (!canLabelCorrections()) {
    return;
  }
  elements.overlayCanvas.setPointerCapture(event.pointerId);
  const point = canvasPointToVideo(event);
  state.isDrawingCorrection = true;
  state.draftCorrection = {
    label: elements.labelClass.value,
    title: elements.labelClass.value,
    confidence: 1,
    bbox: { x: point.x, y: point.y, width: 0, height: 0 },
    start: point,
  };
  renderOverlay();
}

function updateCorrectionDraft(event) {
  if (!state.isDrawingCorrection || !state.draftCorrection) {
    return;
  }
  const point = canvasPointToVideo(event);
  state.draftCorrection.bbox = normalizeBbox(state.draftCorrection.start, point);
  renderOverlay();
}

async function finishCorrectionDraft(event) {
  if (!state.isDrawingCorrection || !state.draftCorrection) {
    return;
  }
  updateCorrectionDraft(event);
  const draft = state.draftCorrection;
  state.isDrawingCorrection = false;
  state.draftCorrection = null;

  if (draft.bbox.width <= 4 || draft.bbox.height <= 4) {
    updateCorrectionStatus("Draw a larger box to save a correction.");
    renderOverlay();
    return;
  }

  await saveCorrection({
    label: draft.label,
    bbox: draft.bbox,
    video_time_seconds: elements.overlayVideo.currentTime || 0,
    notes: activeHardFrameNote(),
  });
}

function cancelCorrectionDraft() {
  state.isDrawingCorrection = false;
  state.draftCorrection = null;
  renderOverlay();
}

async function saveCorrection(payload) {
  if (!state.gameId) {
    return;
  }
  elements.correctionStatus.textContent = "Saving correction...";
  try {
    const response = await fetch(`/api/games/${state.gameId}/corrections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error("Correction save failed");
    }
    const saved = await response.json();
    state.corrections.push(saved.correction);
    updateCorrectionStatus();
    renderHardFrames();
  } catch {
    elements.correctionStatus.textContent = "Correction could not be saved. Try drawing it again.";
  }
  renderOverlay();
}

async function undoLastCorrection() {
  const correction = state.corrections.at(-1);
  if (!state.gameId || !correction) {
    return;
  }

  try {
    const response = await fetch(`/api/games/${state.gameId}/corrections/${correction.correction_id}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      throw new Error("Correction delete failed");
    }
    state.corrections.pop();
  } catch {
    elements.correctionStatus.textContent = "Could not undo the last correction.";
  }
  updateCorrectionStatus();
  renderHardFrames();
  renderOverlay();
}

async function exportCorrections() {
  if (!state.gameId) {
    return;
  }

  let payload = {
    game_id: state.gameId,
    schema: "basketball-detection-corrections-v1",
    corrections: state.corrections,
  };

  try {
    const response = await fetch(`/api/games/${state.gameId}/corrections/export`);
    if (response.ok) {
      payload = await response.json();
    }
  } catch {}

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.gameId}-corrections.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function buildCorrectionDataset() {
  if (!state.gameId || !state.corrections.length) {
    return;
  }

  elements.buildCorrectionDataset.disabled = true;
  elements.correctionStatus.textContent = "Building YOLO dataset from saved corrections...";
  try {
    const response = await fetch(`/api/games/${state.gameId}/corrections/export-yolo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output_dataset: "datasets/basketball-corrections", split: "train" }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Dataset export failed");
    }
    const summary = payload.summary || {};
    elements.correctionStatus.textContent = `YOLO dataset built: ${summary.exported_images || 0} images, ${summary.exported_boxes || 0} boxes. Next: merge/retrain.`;
  } catch (error) {
    elements.correctionStatus.textContent = `Could not build YOLO dataset: ${error.message || "try again"}.`;
  }
  updateCorrectionStatus(elements.correctionStatus.textContent);
}

function canLabelCorrections() {
  return Boolean(
    elements.labelMode.checked
      && state.result?.vision
      && state.gameId
      && elements.overlayVideo.src,
  );
}

function canvasPointToVideo(event) {
  const rect = elements.overlayCanvas.getBoundingClientRect();
  const vision = state.result?.vision || {};
  const videoWidth = Number(vision.width || elements.overlayVideo.videoWidth || 1);
  const videoHeight = Number(vision.height || elements.overlayVideo.videoHeight || 1);
  return {
    x: clamp(((event.clientX - rect.left) / rect.width) * videoWidth, 0, videoWidth),
    y: clamp(((event.clientY - rect.top) / rect.height) * videoHeight, 0, videoHeight),
  };
}

function normalizeBbox(start, end) {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  return {
    x,
    y,
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}

function updateCorrectionStatus(message = null) {
  elements.undoCorrection.disabled = !state.corrections.length;
  elements.exportCorrections.disabled = !state.corrections.length;
  elements.buildCorrectionDataset.disabled = !state.corrections.length;
  if (message) {
    elements.correctionStatus.textContent = message;
    return;
  }
  if (!state.gameId) {
    elements.correctionStatus.textContent = "Turn on Label, choose a class, then drag on the video to save a correction.";
    return;
  }
  const mode = elements.labelMode.checked ? "Label mode on: drag on the video to add a missed box." : "Turn on Label to draw missed boxes.";
  elements.correctionStatus.textContent = `${state.corrections.length} correction${state.corrections.length === 1 ? "" : "s"} saved. ${mode}`;
}

function activeHardFrameNote() {
  const frame = state.activeHardFrameIndex == null ? null : state.hardFrames[state.activeHardFrameIndex];
  if (!frame) {
    return null;
  }
  return `hard_frame frame=${frame.frame_number} reasons=${(frame.reasons || []).join("|")}`;
}

function correctionsNearTime(timestampSeconds, toleranceSeconds) {
  return (state.corrections || []).filter((correction) => (
    Math.abs(Number(correction.video_time_seconds) - timestampSeconds) <= toleranceSeconds
  ));
}

function updateOverlayStatus(boxes, correctionBoxes, currentTime) {
  const summary = boxes.reduce((counts, box) => {
    counts[box.label] = (counts[box.label] || 0) + 1;
    return counts;
  }, {});
  const summaryText = formatSummary(summary);
  const correctionText = correctionBoxes.length ? ` · ${correctionBoxes.length} manual fix${correctionBoxes.length === 1 ? "" : "es"}` : "";
  const totalDetections = state.result?.vision?.detection_count || 0;
  const noBoxesText = totalDetections
    ? `no model boxes near this moment; try Raw detections or scrub a little`
    : "no model detections were produced for this video";
  elements.overlayStatus.textContent = `${formatSeconds(currentTime)} · ${summaryText || noBoxesText}${correctionText}`;
}

function overlayColor(label) {
  const colors = {
    player: "#51d88a",
    ball: "#ff9f1c",
    rim: "#ff4d6d",
    referee: "#4dabf7",
    backboard: "#c77dff",
  };
  return colors[label] || "#fffdf7";
}

function prettyEvent(value) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function renderDetectorInfo(detector, vision = null, warnings = []) {
  if (!detector) {
    elements.modelStatus.textContent = "Detector unknown";
    elements.modelDetails.textContent = "No detector metadata returned yet.";
    elements.detectorSummary.textContent = "Upload a game video to create detection tracks.";
    renderWarnings(warnings);
    return;
  }

  elements.modelStatus.textContent = detectorTitle(detector.engine);
  elements.modelDetails.textContent = detectorDetails(detector);
  elements.detectorSummary.textContent = vision ? visionSummary(vision) : startupSummary(detector);
  renderWarnings(warnings);
}

function detectorTitle(engine) {
  if (engine === "hybrid_yolo_opencv") {
    return "Hybrid active";
  }
  if (engine === "yolo") {
    return "YOLO active";
  }
  if (engine === "opencv") {
    return "OpenCV fallback";
  }
  if (engine === "demo") {
    return "Demo mode";
  }
  return "Detector unknown";
}

function detectorDetails(detector) {
  const parts = [];
  if (detector.model_path) {
    parts.push(fileName(detector.model_path));
  }
  if (Number.isFinite(Number(detector.confidence))) {
    parts.push(`conf ${Math.round(Number(detector.confidence) * 100)}%`);
  }
  if (Number.isFinite(Number(detector.sample_fps))) {
    parts.push(`${Number(detector.sample_fps).toFixed(1)} sample fps`);
  }
  if (Number.isFinite(Number(detector.imgsz))) {
    parts.push(`${detector.imgsz}px`);
  }
  if (Number.isFinite(Number(detector.supplemented_player_tracks))) {
    parts.push(`${detector.supplemented_player_tracks} player supplements`);
  }
  return parts.length ? parts.join(" · ") : detector.name || "No model details";
}

function startupSummary(detector) {
  if (detector.engine === "hybrid_yolo_opencv") {
    const classNames = Object.values(detector.class_names || {}).join(", ");
    return `Ready to run YOLO detections${classNames ? ` for ${classNames}` : ""}, with OpenCV player-track supplement if needed.`;
  }
  if (detector.engine === "yolo") {
    const classNames = Object.values(detector.class_names || {}).join(", ");
    return `Ready to run trained YOLO detections${classNames ? ` for ${classNames}` : ""}.`;
  }
  if (detector.engine === "opencv") {
    return "Ready to run heuristic OpenCV detection. Add trained weights for YOLO mode.";
  }
  return "Upload a video to start analysis.";
}

function visionSummary(vision) {
  const pieces = [
    `Sampled ${vision.sampled_frame_count ?? 0} frames`,
    `${vision.detection_count ?? 0} detections`,
    `${vision.track_count ?? 0} tracks`,
  ];
  const detectionSummary = formatSummary(vision.detection_summary);
  const trackSummary = formatSummary(vision.track_summary);
  if (detectionSummary) {
    pieces.push(`detections: ${detectionSummary}`);
  }
  if (trackSummary) {
    pieces.push(`tracks: ${trackSummary}`);
  }
  return pieces.join(" · ");
}

function renderWarnings(warnings) {
  elements.warningList.innerHTML = "";
  const visibleWarnings = warnings.filter(Boolean).slice(0, 4);
  if (!visibleWarnings.length) {
    const item = document.createElement("li");
    item.textContent = "Detector ready.";
    elements.warningList.appendChild(item);
    return;
  }
  for (const warning of visibleWarnings) {
    const item = document.createElement("li");
    item.textContent = warning;
    elements.warningList.appendChild(item);
  }
}

function formatSummary(summary) {
  if (!summary) {
    return "";
  }
  return Object.entries(summary)
    .map(([label, count]) => `${prettyEvent(label)} ${count}`)
    .join(", ");
}

function fileName(path) {
  return String(path).split(/[\\/]/).pop();
}

function formatSeconds(seconds) {
  const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
