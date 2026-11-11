const PROFILE_KEY = "barehands.gesture-profile.v1";

export const DEFAULT_PROFILE = Object.freeze({
  version: 1,
  name: "Default",
  minQuality: 0.58,
  dropoutGraceMs: 180,
  pinchIntentMs: 55,
  pinchReleaseMs: 0,
  fastPinchReleaseMs: 70,
  pinchProbationMs: 135,
  pinchContrast: 0.18,
  pinchBackFloor: 1.30,
  pinchProfileAspect: 2.0,
  pinchProfileThumb: 0.95,
  pinchGapFrontal: 0.32,
  pinchGapProfile: 0.38,
  pinchReleaseGap: 0.55,
  pinchFastReleaseGap: 0.70,
  clawArmMs: 465,
  clawMouthEnter: 0.80,
  clawMouthHold: 0.68,
  clawIndexEnter: 0.60,
  clawMiddleEnter: 0.35,
  clawRingEnter: 0.55,
  clawMeanEnter: 0.30,
  clawPinkyEnter: 0.10,
  clapHandUp: 0.85,
  clapApart: 0.18,
  clapWristClose: 0.11,
  clapKnuckleClose: 0.09,
  pointerMinCutoff: 1.2,
  pointerBeta: 0.018,
  pointerDerivativeCutoff: 1.0,
  fastMotionViewportPerSec: 0.42,
  ghostBirthViewportPerSec: 0.47,
  ghostRecoveryViewportPerSec: 0.26,
  throwViewportPerSec: 0.68,
  tapTravelViewport: 0.024,
  pinchClosingVelocityThresh: 0.45,
  intentLockDurationMs: 140,
  intentLockDeadzone: 0.028,
  multiCamMinQuality: 0.50
});

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const dist2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const dist3 = (a, b) => Math.hypot(
  a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));

function sanitizeProfile(profile) {
  const clean = { ...DEFAULT_PROFILE, version: 1 };
  for (const [key, fallback] of Object.entries(DEFAULT_PROFILE)) {
    if (typeof fallback !== "number") continue;
    const value = Number(profile?.[key]);
    clean[key] = Number.isFinite(value) ? value : fallback;
  }
  clean.name = typeof profile?.name === "string"
    ? profile.name.slice(0, 80) : DEFAULT_PROFILE.name;
  clean.minQuality = clamp(clean.minQuality, 0.3, 0.95);
  clean.dropoutGraceMs = clamp(clean.dropoutGraceMs, 50, 500);
  clean.pinchIntentMs = clamp(clean.pinchIntentMs, 0, 300);
  clean.pinchReleaseMs = clamp(clean.pinchReleaseMs, 0, 200);
  clean.fastPinchReleaseMs = clamp(clean.fastPinchReleaseMs, 0, 250);
  clean.pinchProbationMs = clamp(clean.pinchProbationMs, 50, 500);
  clean.pinchContrast = clamp(clean.pinchContrast, 0.08, 0.40);
  clean.pinchBackFloor = clamp(clean.pinchBackFloor, 1.05, 1.65);
  clean.pinchGapFrontal = clamp(clean.pinchGapFrontal, 0.18, 0.38);
  clean.pinchGapProfile = clamp(clean.pinchGapProfile, 0.22, 0.45);
  clean.clawArmMs = clamp(clean.clawArmMs, 200, 1200);
  clean.clawMouthEnter = clamp(clean.clawMouthEnter, 0.60, 1.10);
  clean.clawMouthHold = clamp(
    clean.clawMouthHold, 0.45, clean.clawMouthEnter - 0.03);
  clean.clapHandUp = clamp(clean.clapHandUp, 0.55, 1.3);
  clean.pointerMinCutoff = clamp(clean.pointerMinCutoff, 0.1, 10);
  clean.pointerBeta = clamp(clean.pointerBeta, 0, 0.2);
  clean.pointerDerivativeCutoff = clamp(
    clean.pointerDerivativeCutoff, 0.1, 10);
  return clean;
}

export function loadGestureProfile(storage = globalThis.localStorage) {
  try {
    const saved = JSON.parse(storage.getItem(PROFILE_KEY) || "null");
    return saved && saved.version === 1
      ? sanitizeProfile(saved)
      : { ...DEFAULT_PROFILE };
  } catch {
    return { ...DEFAULT_PROFILE };
  }
}

export function saveGestureProfile(profile, storage = globalThis.localStorage) {
  const clean = sanitizeProfile(profile);
  storage.setItem(PROFILE_KEY, JSON.stringify(clean));
  return clean;
}

export function resetGestureProfile(storage = globalThis.localStorage) {
  storage.removeItem(PROFILE_KEY);
  return { ...DEFAULT_PROFILE };
}

class LowPassFilter {
  constructor() {
    this.value = 0;
    this.ready = false;
  }

  filter(value, alpha) {
    if (!this.ready) {
      this.value = value;
      this.ready = true;
    } else {
      this.value = alpha * value + (1 - alpha) * this.value;
    }
    return this.value;
  }
}

class OneEuroAxis {
  constructor(minCutoff, beta, derivativeCutoff) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.value = new LowPassFilter();
    this.derivative = new LowPassFilter();
    this.lastRaw = 0;
    this.lastTime = 0;
  }

  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(raw, nowMs) {
    if (!this.lastTime) {
      this.lastRaw = raw;
      this.lastTime = nowMs;
      return this.value.filter(raw, 1);
    }
    const dt = clamp((nowMs - this.lastTime) / 1000, 1 / 240, 0.1);
    const derivative = (raw - this.lastRaw) / dt;
    const smoothDerivative = this.derivative.filter(
      derivative, this.alpha(this.derivativeCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(smoothDerivative);
    this.lastRaw = raw;
    this.lastTime = nowMs;
    return this.value.filter(raw, this.alpha(cutoff, dt));
  }
}

export class AdaptivePointFilter {
  constructor(profile = DEFAULT_PROFILE) {
    this.x = new OneEuroAxis(
      profile.pointerMinCutoff, profile.pointerBeta,
      profile.pointerDerivativeCutoff);
    this.y = new OneEuroAxis(
      profile.pointerMinCutoff, profile.pointerBeta,
      profile.pointerDerivativeCutoff);
  }

  update(x, y, nowMs) {
    return { x: this.x.filter(x, nowMs), y: this.y.filter(y, nowMs) };
  }
}

export class TemporalGate {
  constructor(enterMs, exitMs = 0) {
    this.enterMs = enterMs;
    this.exitMs = exitMs;
    this.active = false;
    this.trueSince = 0;
    this.falseSince = 0;
  }

  update(value, nowMs) {
    if (value) {
      this.falseSince = 0;
      if (!this.trueSince) this.trueSince = nowMs;
      if (!this.active && nowMs - this.trueSince >= this.enterMs)
        this.active = true;
    } else {
      this.trueSince = 0;
      if (!this.falseSince) this.falseSince = nowMs;
      if (this.active && nowMs - this.falseSince >= this.exitMs)
        this.active = false;
    }
    return this.active;
  }
}

function handednessName(entry) {
  const category = Array.isArray(entry) ? entry[0] : entry;
  return category?.categoryName || category?.displayName || "Unknown";
}

export class StableHandTracker {
  constructor({ graceMs = DEFAULT_PROFILE.dropoutGraceMs } = {}) {
    this.graceMs = graceMs;
    this.nextId = 1;
    this.tracks = new Map();
  }

  update(detections, nowMs) {
    const unmatched = new Set(this.tracks.keys());
    const result = [];
    const ordered = [...detections].sort((a, b) =>
      (handednessName(a.handedness) > handednessName(b.handedness) ? 1 : -1));

    for (const detection of ordered) {
      const wrist = detection.landmarks[0];
      const hand = handednessName(detection.handedness);
      let bestId = null;
      let bestCost = Infinity;
      for (const id of unmatched) {
        const track = this.tracks.get(id);
        const handPenalty = track.handedness !== "Unknown" &&
          hand !== "Unknown" && track.handedness !== hand ? 0.35 : 0;
        const elapsed = Math.max(1, nowMs - track.lastSeen);
        const predicted = {
          x: track.wrist.x + track.velocity.x * elapsed,
          y: track.wrist.y + track.velocity.y * elapsed
        };
        const cost = dist2(wrist, predicted) + handPenalty;
        const travelBudget = Math.min(0.20, 0.07 + elapsed * 0.0007);
        if (cost < bestCost && cost < travelBudget) {
          bestCost = cost;
          bestId = id;
        }
      }

      if (bestId == null) bestId = this.nextId++;
      const previous = this.tracks.get(bestId);
      const elapsed = previous ? Math.max(1, nowMs - previous.lastSeen) : 1;
      const velocity = previous
        ? {
            x: (wrist.x - previous.wrist.x) / elapsed,
            y: (wrist.y - previous.wrist.y) / elapsed
          }
        : { x: 0, y: 0 };
      this.tracks.set(bestId, {
        id: bestId,
        wrist: { x: wrist.x, y: wrist.y },
        velocity,
        handedness: hand,
        lastSeen: nowMs,
        landmarks: detection.landmarks
      });
      unmatched.delete(bestId);
      result.push({ ...detection, id: bestId, handedness: hand });
    }

    for (const [id, track] of this.tracks)
      if (nowMs - track.lastSeen > this.graceMs) this.tracks.delete(id);
    return result;
  }

  activeIds() {
    return new Set(this.tracks.keys());
  }
}

const fingerArch = (landmarks, tip, mcp) => {
  const wrist = landmarks[0];
  const base = dist2(landmarks[mcp], wrist);
  return base > 0 ? dist2(landmarks[tip], wrist) / base : 9;
};

const segment = (landmarks, a, b) => {
  const length = dist3(landmarks[a], landmarks[b]) || 1;
  return {
    x: (landmarks[b].x - landmarks[a].x) / length,
    y: (landmarks[b].y - landmarks[a].y) / length,
    z: ((landmarks[b].z || 0) - (landmarks[a].z || 0)) / length
  };
};

const fingerCurl = (landmarks, mcp, pip, dip, tip) => {
  const proximal = segment(landmarks, mcp, pip);
  const distal = segment(landmarks, dip, tip);
  return proximal.x * distal.x + proximal.y * distal.y +
    proximal.z * distal.z;
};

export function assessLandmarkQuality(landmarks, previous = null) {
  if (!Array.isArray(landmarks) || landmarks.length !== 21) return 0;
  if (landmarks.some(point =>
    !Number.isFinite(point.x) || !Number.isFinite(point.y))) return 0;
  const span = dist2(landmarks[0], landmarks[9]);
  const palmWidth = dist2(landmarks[5], landmarks[17]);
  if (span < 0.015 || palmWidth < 0.01) return 0;
  const aspect = span / palmWidth;
  let score = 1;
  if (aspect < 0.65 || aspect > 6) score -= 0.55;
  const outside = landmarks.filter(point =>
    point.x < -0.08 || point.x > 1.08 || point.y < -0.08 || point.y > 1.08).length;
  score -= outside * 0.025;
  if (previous) {
    const meanJump = landmarks.reduce((sum, point, i) =>
      sum + dist3(point, previous[i]), 0) / landmarks.length;
    if (meanJump > 0.15) score -= Math.min(0.4, (meanJump - 0.15) * 2);
  }
  return clamp(score, 0, 1);
}

export function extractHandFeatures(landmarks, previous = null) {
  // Shape thresholds in stage.html were fitted in image-space. Keep the
  // shared feature math in that same space; curls alone use z because their
  // angular fold test was explicitly fitted in 3D.
  const span = dist2(landmarks[0], landmarks[9]) || 1;
  const palmWidth = dist2(landmarks[5], landmarks[17]) || 1;
  const f8 = fingerArch(landmarks, 8, 5);
  const f12 = fingerArch(landmarks, 12, 9);
  const f16 = fingerArch(landmarks, 16, 13);
  const f20 = fingerArch(landmarks, 20, 17);
  const backMean = (f12 + f16 + f20) / 3;
  const tipGap = (a, b) => dist2(landmarks[a], landmarks[b]);
  return {
    ratio: dist2(landmarks[4], landmarks[8]) / span,
    f8, f12, f16, f20,
    backMean,
    contrast: backMean - f8,
    thumbOffset: dist2(landmarks[4], landmarks[13]) / span,
    aspect: span / palmWidth,
    c8: fingerCurl(landmarks, 5, 6, 7, 8),
    c12: fingerCurl(landmarks, 9, 10, 11, 12),
    c16: fingerCurl(landmarks, 13, 14, 15, 16),
    c20: fingerCurl(landmarks, 17, 18, 19, 20),
    spread: (tipGap(8, 12) + tipGap(12, 16) + tipGap(16, 20)) /
      (3 * span),
    handUp: (landmarks[0].y - landmarks[9].y) / span,
    quality: assessLandmarkQuality(landmarks, previous)
  };
}

export function classifyRuleBased(features, profile = DEFAULT_PROFILE) {
  const frontalPinch =
    features.contrast > profile.pinchContrast &&
    features.backMean > profile.pinchBackFloor;
  const profilePinch =
    features.aspect < profile.pinchProfileAspect &&
    features.thumbOffset > profile.pinchProfileThumb;
  const pinchCeiling = features.aspect < profile.pinchProfileAspect
    ? profile.pinchGapProfile : profile.pinchGapFrontal;
  const pinch = features.quality >= profile.minQuality &&
    features.ratio < pinchCeiling && (frontalPinch || profilePinch);
  const clawMean = (features.c8 + features.c12 + features.c16) / 3;
  const claw = features.quality >= profile.minQuality &&
    features.ratio > profile.clawMouthEnter &&
    features.ratio < 1.45 &&
    features.c8 < profile.clawIndexEnter &&
    features.c12 < profile.clawMiddleEnter &&
    features.c16 < profile.clawRingEnter &&
    clawMean < profile.clawMeanEnter &&
    features.c20 > profile.clawPinkyEnter &&
    features.aspect > 1.05;
  const open = features.ratio > 0.7 &&
    [features.f8, features.f12, features.f16, features.f20]
      .filter(value => value > 1.35).length >= 3;
  return { pinch, claw, open };
}

const numericFeatureNames = Object.freeze([
  "ratio", "f8", "f12", "f16", "f20", "backMean", "contrast",
  "thumbOffset", "aspect", "c8", "c12", "c16", "c20", "spread",
  "handUp", "quality"
]);

export function featureVector(features) {
  return numericFeatureNames.map(name => Number(features[name]) || 0);
}

export function featureNames() {
  return [...numericFeatureNames];
}

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.round((sorted.length - 1) * fraction)];
}

function canyonThreshold(positive, negative, direction, fallback) {
  if (!positive.length || !negative.length) return fallback;
  const positiveEdge = percentile(positive, direction === "high" ? 0.10 : 0.90);
  const negativeEdge = percentile(negative, direction === "high" ? 0.90 : 0.10);
  const separated = direction === "high"
    ? positiveEdge > negativeEdge : positiveEdge < negativeEdge;
  return separated ? (positiveEdge + negativeEdge) / 2 : fallback;
}

export function deriveProfile(samples, base = DEFAULT_PROFILE) {
  const byLabel = label => samples
    .filter(sample => sample.label === label && sample.features)
    .map(sample => sample.features);
  const pinch = byLabel("pinch");
  const fistAndRelaxed = [...byLabel("fist"), ...byLabel("relaxed")];
  const claw = byLabel("claw");
  const clawImpostors = [...byLabel("relaxed"), ...byLabel("point")];
  return {
    ...base,
    version: 1,
    name: `Calibrated ${new Date().toLocaleDateString()}`,
    pinchContrast: canyonThreshold(
      pinch.map(v => v.contrast), fistAndRelaxed.map(v => v.contrast),
      "high", base.pinchContrast),
    pinchBackFloor: canyonThreshold(
      pinch.map(v => v.backMean), byLabel("fist").map(v => v.backMean),
      "high", base.pinchBackFloor),
    pinchGapFrontal: canyonThreshold(
      pinch.map(v => v.ratio), fistAndRelaxed.map(v => v.ratio),
      "low", base.pinchGapFrontal),
    clawMouthEnter: canyonThreshold(
      claw.map(v => v.ratio), clawImpostors.map(v => v.ratio),
      "high", base.clawMouthEnter),
    clawMiddleEnter: canyonThreshold(
      claw.map(v => v.c12), clawImpostors.map(v => v.c12),
      "low", base.clawMiddleEnter),
    clawMeanEnter: canyonThreshold(
      claw.map(v => (v.c8 + v.c12 + v.c16) / 3),
      clawImpostors.map(v => (v.c8 + v.c12 + v.c16) / 3),
      "low", base.clawMeanEnter)
  };
}

export class LandmarkRecorder {
  constructor() {
    this.frames = [];
    this.startedAt = 0;
  }

  start(nowMs = performance.now()) {
    this.frames = [];
    this.startedAt = nowMs;
  }

  add({ label, landmarks, handedness = "Unknown", nowMs = performance.now() }) {
    if (!this.startedAt) this.start(nowMs);
    this.frames.push({
      t: Math.round(nowMs - this.startedAt),
      label,
      handedness,
      landmarks: landmarks.map(point => ({
        x: point.x, y: point.y, z: point.z || 0
      })),
      features: extractHandFeatures(landmarks)
    });
  }

  export(metadata = {}) {
    return {
      schema: "barehands-landmarks/v1",
      createdAt: new Date().toISOString(),
      featureNames: featureNames(),
      metadata,
      frames: this.frames
    };
  }
}

export function replayDataset(dataset, profile = DEFAULT_PROFILE) {
  if (dataset?.schema !== "barehands-landmarks/v1" ||
      !Array.isArray(dataset.frames))
    throw new Error("Not a barehands landmark dataset");
  const totals = {};
  for (const frame of dataset.frames) {
    const features = frame.features || extractHandFeatures(frame.landmarks);
    const predicted = classifyRuleBased(features, profile);
    const expected = frame.label;
    if (!totals[expected])
      totals[expected] = { frames: 0, matched: 0, pinch: 0, claw: 0, open: 0 };
    const row = totals[expected];
    row.frames++;
    if (predicted[expected] === true ||
        (!["pinch", "claw", "open"].includes(expected) &&
         !predicted.pinch && !predicted.claw && !predicted.open)) row.matched++;
    if (predicted.pinch) row.pinch++;
    if (predicted.claw) row.claw++;
    if (predicted.open) row.open++;
  }
  return totals;
}

export class DynamicPinchGate {
  constructor(profile = DEFAULT_PROFILE) {
    this.profile = profile;
    this.ratioHistory = [];
    this.isPinched = false;
    this.intentLocked = false;
    this.lockStartedT = 0;
    this.lockedAnchor = null;
  }

  update({ ratio, x, y, nowMs = performance.now() }) {
    this.ratioHistory.push({ t: nowMs, ratio });
    while (this.ratioHistory.length && nowMs - this.ratioHistory[0].t > 300) {
      this.ratioHistory.shift();
    }

    let closingRate = 0;
    if (this.ratioHistory.length >= 2) {
      const oldest = this.ratioHistory[0];
      const dt = Math.max(0.001, (nowMs - oldest.t) / 1000);
      closingRate = (ratio - oldest.ratio) / dt;
    }

    const isClosingFast = closingRate < -this.profile.pinchClosingVelocityThresh;

    if (!this.isPinched && isClosingFast && ratio < 0.45) {
      if (!this.intentLocked) {
        this.intentLocked = true;
        this.lockStartedT = nowMs;
        this.lockedAnchor = { x, y };
      }
    }

    if (this.intentLocked) {
      const travel = this.lockedAnchor ? dist2({ x, y }, this.lockedAnchor) : 0;
      if (nowMs - this.lockStartedT > this.profile.intentLockDurationMs ||
          travel > this.profile.intentLockDeadzone) {
        this.intentLocked = false;
      }
    }

    const outputPos = (this.intentLocked && this.lockedAnchor) ? this.lockedAnchor : { x, y };

    return {
      outputPos,
      closingRate,
      intentLocked: this.intentLocked,
      isClosingFast
    };
  }
}

export class MultiCameraFusion {
  constructor({ minQuality = DEFAULT_PROFILE.multiCamMinQuality } = {}) {
    this.minQuality = minQuality;
    this.cameraOffsets = new Map();
  }

  setCameraOffset(cameraId, offset = { dx: 0, dy: 0, dz: 0 }) {
    this.cameraOffsets.set(cameraId, offset);
  }

  fuse(cameraFrames) {
    const candidates = [];
    for (const frame of cameraFrames) {
      const offset = this.cameraOffsets.get(frame.cameraId) || { dx: 0, dy: 0, dz: 0 };
      for (const det of (frame.detections || [])) {
        const quality = assessLandmarkQuality(det.landmarks);
        if (quality < this.minQuality) continue;
        const hand = handednessName(det.handedness);
        candidates.push({
          cameraId: frame.cameraId,
          handedness: hand,
          landmarks: det.landmarks.map(p => ({
            x: p.x + offset.dx,
            y: p.y + offset.dy,
            z: (p.z || 0) + offset.dz
          })),
          quality
        });
      }
    }

    if (!candidates.length) return [];

    const byHand = new Map();
    for (const cand of candidates) {
      const list = byHand.get(cand.handedness) || [];
      list.push(cand);
      byHand.set(cand.handedness, list);
    }

    const fusedHands = [];
    for (const [handedness, views] of byHand) {
      if (views.length === 1) {
        fusedHands.push({
          handedness,
          landmarks: views[0].landmarks,
          quality: views[0].quality,
          sourceCameraCount: 1
        });
        continue;
      }

      const totalWeight = views.reduce((sum, v) => sum + v.quality, 0);
      const fusedLandmarks = [];
      for (let i = 0; i < 21; i++) {
        let fx = 0, fy = 0, fz = 0;
        for (const v of views) {
          const pt = v.landmarks[i];
          fx += pt.x * v.quality;
          fy += pt.y * v.quality;
          fz += (pt.z || 0) * v.quality;
        }
        fusedLandmarks.push({
          x: fx / totalWeight,
          y: fy / totalWeight,
          z: fz / totalWeight
        });
      }

      fusedHands.push({
        handedness,
        landmarks: fusedLandmarks,
        quality: Math.min(1.0, (totalWeight / views.length) + 0.15),
        sourceCameraCount: views.length
      });
    }

    return fusedHands;
  }
}

