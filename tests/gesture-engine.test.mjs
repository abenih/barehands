import assert from "node:assert/strict";
import {
  AdaptivePointFilter, DEFAULT_PROFILE, DynamicPinchGate, MultiCameraFusion,
  StableHandTracker, TemporalGate, deriveProfile, replayDataset
} from "../gesture-engine.mjs";

const hand = (x, label = "Left") => ({
  landmarks: Array.from({ length: 21 }, (_, index) => ({
    x: x + index * 0.001, y: 0.4 + index * 0.001, z: 0
  })),
  handedness: [{ categoryName: label }]
});

const tracker = new StableHandTracker({ graceMs: 180 });
const first = tracker.update([hand(0.2), hand(0.7, "Right")], 0);
const crossed = tracker.update([hand(0.68, "Right"), hand(0.22)], 33);
assert.equal(first[0].id, crossed.find(row => row.handedness === "Left").id);
assert.equal(first[1].id, crossed.find(row => row.handedness === "Right").id);

tracker.update([], 100);
assert.equal(tracker.activeIds().size, 2);
tracker.update([], 250);
assert.equal(tracker.activeIds().size, 0);

const reuseTracker = new StableHandTracker({ graceMs: 180 });
const original = reuseTracker.update([hand(0.1)], 10)[0];
const impossibleJump = reuseTracker.update([hand(0.55)], 110)[0];
assert.notEqual(original.id, impossibleJump.id);

const gate = new TemporalGate(60, 40);
assert.equal(gate.update(true, 10), false);
assert.equal(gate.update(true, 69), false);
assert.equal(gate.update(true, 70), true);
assert.equal(gate.update(false, 90), true);
assert.equal(gate.update(false, 130), false);

const filter = new AdaptivePointFilter(DEFAULT_PROFILE);
assert.deepEqual(filter.update(10, 20, 1), { x: 10, y: 20 });
const moved = filter.update(20, 30, 34);
assert(moved.x > 10 && moved.x < 20);

const sample = features => ({ label: "pinch", features });
const profile = deriveProfile([
  sample({ contrast: 0.4, backMean: 1.6, ratio: 0.2 }),
  { label: "fist", features: { contrast: 0.05, backMean: 1.0, ratio: 0.5 } },
  { label: "relaxed", features: { contrast: 0.08, backMean: 1.5, ratio: 0.8 } }
]);
assert(profile.pinchContrast > 0.08 && profile.pinchContrast < 0.4);

assert.throws(() => replayDataset({ schema: "wrong", frames: [] }));

// Test DynamicPinchGate
const pinchGate = new DynamicPinchGate(DEFAULT_PROFILE);
const p1 = pinchGate.update({ ratio: 0.8, x: 0.5, y: 0.5, nowMs: 0 });
assert.equal(p1.intentLocked, false);
// Rapid closing towards pinch
const p2 = pinchGate.update({ ratio: 0.35, x: 0.505, y: 0.502, nowMs: 100 });
assert.equal(p2.intentLocked, true);
assert.deepEqual(p2.outputPos, { x: 0.505, y: 0.502 });
// Small jitter during pinch is locked to anchor
const p3 = pinchGate.update({ ratio: 0.25, x: 0.508, y: 0.504, nowMs: 140 });
assert.equal(p3.intentLocked, true);
assert.deepEqual(p3.outputPos, { x: 0.505, y: 0.502 }); // Position locked to prevent click-drift

// Test MultiCameraFusion
const fusion = new MultiCameraFusion();
const realisticHand = (x, label = "Right") => ({
  landmarks: Array.from({ length: 21 }, (_, index) => ({
    x: x + index * 0.01, y: 0.4 + index * 0.01, z: 0
  })),
  handedness: [{ categoryName: label }]
});
const cam1Hand = realisticHand(0.3, "Right");
const cam2Hand = realisticHand(0.32, "Right");
const fused = fusion.fuse([
  { cameraId: "cam0", detections: [cam1Hand] },
  { cameraId: "cam1", detections: [cam2Hand] }
]);
assert.equal(fused.length, 1);
assert.equal(fused[0].sourceCameraCount, 2);
assert.equal(fused[0].handedness, "Right");

console.log("gesture engine tests passed");
