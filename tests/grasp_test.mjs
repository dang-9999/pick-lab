// Headless check: scripted top-down pick in MuJoCo must lift the cube above 15 cm.
// Usage: node grasp_test.mjs <mujoco pkg dir> <scene source: .xml or .html with <script id="mjcf">>
import { readFileSync } from 'node:fs';
import assert from 'node:assert';
const [pkg, src] = process.argv.slice(2);
const loadMujoco = (await import(`${pkg}/mujoco.js`)).default;
const mj = await loadMujoco({ wasmBinary: readFileSync(`${pkg}/mujoco.wasm`) });
let xml = readFileSync(src, 'utf8');
if (src.endsWith('.html')) xml = xml.match(/<script type="text\/xml" id="mjcf">([\s\S]*?)<\/script>/)[1];
const model = mj.MjModel.from_xml_string(xml);
const data = new mj.MjData(model);

const L0 = 0.33, L1 = 0.32, L2 = 0.30, TCP = 0.12;
function ik(x, y, z, roll) { // MuJoCo frame, Z up
  const r = Math.hypot(x, y), dz = z + TCP - L0, d = Math.hypot(r, dz);
  const q3 = Math.acos(Math.min(1, Math.max(-1, (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2))));
  const q2 = Math.atan2(r, dz) - Math.atan2(L2 * Math.sin(q3), L1 + L2 * Math.cos(q3));
  return [Math.atan2(y, x), q2, q3, Math.PI - q2 - q3, roll];
}
const site = n => { const i = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, n); return [...data.site_xpos.slice(3 * i, 3 * i + 3)]; };
const cubeBody = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'cube');
const cubePos = () => [...data.xpos.slice(3 * cubeBody, 3 * cubeBody + 3)];

const cx = 0.38, cy = 0.12;
const qa = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, 'cube')];
data.qpos.set([cx, cy, 0.025, 1, 0, 0, 0], qa);
let q = ik(cx, cy, 0.2, 0); data.qpos.set(q, 0); data.ctrl.set([...q, 0.04, 0.04]); mj.mj_forward(model, data);

function go(z, grip, secs) {
  q = ik(cx, cy, z, 0); data.ctrl.set([...q, grip, grip]);
  for (let i = 0; i < secs / model.opt.timestep; i++) mj.mj_step(model, data);
}
go(0.2, 0.04, 0.5);
let err = Math.hypot(...site('tcp').map((v, i) => v - [cx, cy, 0.2][i]));
console.log('tracking error at hover (m):', err.toFixed(4));
go(0.03, 0.04, 1.0);
console.log('tcp before close', site('tcp').map(v => v.toFixed(3)), 'cube', cubePos().map(v => v.toFixed(3)));
go(0.03, 0, 0.6);
console.log('finger opening while closed:', data.qpos[5].toFixed(4), data.qpos[6].toFixed(4));
go(0.25, 0, 1.5);
const c = cubePos();
console.log('cube after lift', c.map(v => v.toFixed(3)));
assert(err < 0.01, 'arm should track hover target within 1 cm');
assert(c[2] > 0.15, 'cube should be lifted above 15 cm');
// drop: open and let it fall
go(0.25, 0.04, 1.0);
console.log('cube after release', cubePos().map(v => v.toFixed(3)));
assert(cubePos()[2] < 0.03, 'released cube should land on the floor');
console.log('OK');
