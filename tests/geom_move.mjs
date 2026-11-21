import { readFileSync } from 'node:fs';
const pkg = process.argv[2];
const mj = await (await import(`${pkg}/mujoco.js`)).default({ wasmBinary: readFileSync(`${pkg}/mujoco.wasm`) });
const m = mj.MjModel.from_xml_string(`<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/><geom name="wall" type="box" size=".16 .01 .06" pos="0 3 .06"/>
<body name="c" pos="0.3 0 0.3"><freejoint/><geom type="box" size=".025 .025 .025" mass=".05"/></body></worldbody></mujoco>`);
const d = new mj.MjData(m);
const g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM.value, 'wall');
m.geom_pos.set([0.3, 0, 0.06], 3 * g); m.geom_quat.set([Math.cos(0.4), 0, 0, Math.sin(0.4)], 4 * g);
mj.mj_forward(m, d);
console.log('wall xpos', [...d.geom_xpos.slice(3 * g, 3 * g + 3)]);
for (let i = 0; i < 1000; i++) mj.mj_step(m, d);
console.log('cube z after falling onto moved wall (expect ~0.145):', d.qpos[2].toFixed(3));
