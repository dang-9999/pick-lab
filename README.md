# Pick Lab

건설 현장 작업 로봇을 키보드로 조종해 벽돌을 다루는 브라우저 물리 게임입니다. 모든 플레이는 로봇 학습용 시연 데이터(관절 궤적과 손목 카메라 영상)로 저장되고, LeRobot 데이터셋으로 변환할 수 있습니다.

- 물리 엔진: [MuJoCo](https://github.com/google-deepmind/mujoco) 4.0.1 WASM (`@mujoco/mujoco`, Apache-2.0)
- 렌더링: three.js r128
- 로봇: 5관절 팔과 2지 그리퍼 (MJCF가 `game/index.html` 안에 포함)

## 스테이지

| # | 이름 | 목표 | ★★★ / ★★ |
|---|---|---|---|
| 1 | 자재 들기 | 벽돌을 15cm 위로 들어 1초 유지 | 12초 / 25초 |
| 2 | 자재 운반 | 벽돌을 초록 원 안에 내려놓기 | 15초 / 30초 |
| 3 | 정밀 배치 | 오차 1.5cm 이내로 작은 원에 넣기 | 20초 / 40초 |
| 4 | 바리케이드 넘기 | 바리케이드 너머의 원으로 옮기기 | 25초 / 45초 |
| 5 | 적재 | 벽돌과 파란 블록 중 하나를 다른 하나 위에 올리기 | 25초 / 45초 |

조작은 그리퍼 위치 모드(W/A/S/D, E/Q, Z/C, Space)와 관절 직접 모드(Q/A ~ T/G) 두 가지이고, `M`으로 전환합니다.

## 구성

```
game/index.html          게임 전체 (HTML 하나, claude.ai 아티팩트로 배포)
game/mujoco.wasm         MuJoCo 엔진 바이너리 (페이지가 같은 폴더에서 불러옴)
pick_lab_to_lerobot.py   에피소드 → LeRobot 데이터셋 변환
tests/grasp_test.mjs     MuJoCo 헤드리스 집기 테스트
tests/geom_move.mjs      실행 중 벽 위치 변경 테스트
```

## 로컬에서 실행

```bash
python3 -m http.server 8765 --directory game
```

브라우저에서 `http://localhost:8765`를 엽니다. 로컬에서는 기록이 저장되지 않습니다(저장은 claude.ai 아티팩트의 DB 기능을 사용).

## 데이터 포맷

에피소드 하나가 문서 하나이며, 10Hz 열 단위 배열로 저장됩니다.

- 좌표계: MuJoCo 월드 기준 (Z축이 위, 미터, 라디안)
- 숫자는 1000을 곱한 정수 (`scale: 1000`)
- 관측: 관절 각도 `q1~q5`, 집게 개폐 `grip`, 그리퍼 끝 위치 `tcp_*`, 큐브 위치와 쿼터니언
- 행동: 서보 목표값 `ctrl1~5`, `ctrl_grip`
- 사람 입력: `key0~5`, 조작 모드 `mode`
- 손목 카메라: 128×128 WebM, 10fps, i번째 프레임이 i번째 행과 대응

## LeRobot 변환

```bash
python3 -m venv .venv && .venv/bin/pip install lerobot
.venv/bin/python pick_lab_to_lerobot.py --raw raw --out dataset --repo-id local/pick-lab
```

`raw/episodes/*.json`과 `raw/videos/<asset id>.webm`을 읽어 LeRobot v3.0 데이터셋을 만듭니다. 성공한 판만 넣으려면 `--only-success`, Hugging Face에 올리려면 `--repo-id 계정/pick-lab --push`를 붙입니다.

| LeRobot 키 | 형태 | 내용 |
|---|---|---|
| `observation.state` | float32 [6] | 관절 5개 + 집게 |
| `action` | float32 [6] | 서보 목표값 |
| `observation.images.wrist` | video 128×128 | 손목 카메라 |
| `observation.ee_pos` | float32 [3] | 그리퍼 끝 위치 |
| `observation.environment_state` | float32 [17] | 큐브 2개 자세 + 벽 위치 |
| `observation.goal` | float32 [4] | 목표 원 중심, 허용 오차, 목표 원 유무 |
| `teleop.keys`, `teleop.mode` | float32 | 사람 키 입력, 조작 모드 |
| `next.success`, `next.done` | bool | 마지막 프레임의 성공 여부 |

이 Mac처럼 시스템 FFmpeg가 없으면 데이터셋을 읽을 때 `video_backend="pyav"`를 넘깁니다.
