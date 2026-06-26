# 압사위험 감지 시스템 (Crowd Management System)

CCTV 영상에서 군중 밀도/흐름을 분석해 압사 위험을 감지하는 시스템.
P2PNet(인원 카운팅) + Optical Flow(흐름) + Risk Engine(Green Guide/kNN/Weidmann).

## 시연 영상

▶ **[YouTube 시연 영상 보기](https://youtu.be/MwY7M_0E2vY)**

> 샘플 영상(`cctv_crowd_dashboard/data/sibuya_test.mp4`, 22MB)은 저장소에 포함되어 바로 실행 가능.

## 빠른 시작 (Windows)

전제: **Python 3.10~3.12** 설치 ([python.org](https://www.python.org/downloads/), 설치 시 "Add to PATH" 체크).

```
1) git clone <이 저장소>
2) setup.bat          더블클릭  → 가상환경 + 의존성 설치 (최초 1회, GPU 자동 감지)
3) 영상 준비          cctv_crowd_dashboard\data\ 에 .mp4 넣기 (예: sibuya_test.mp4)
4) run_setup_roi.bat  --video data\내영상.mp4   
    → 관심영역(ROI) 마우스로 그리기
    → 호모그래피 기준점 4개 마우스로 그리기
    → 실세게 좌표 4개(ex. 0, 0) 입력(임의로)
5) run.bat            서버 실행
6) 브라우저           http://127.0.0.1:8000
```

## 구성

| 파일 | 역할 |
|---|---|
| `setup.bat` | venv 생성 + PyTorch(CUDA 12.6, GPU 없으면 CPU) + 의존성 설치 |
| `run.bat` | 웹 대시보드 서버 실행 (포트 8000) |
| `run_setup_roi.bat` | ROI 설정 GUI (마우스로 영역 지정 → json 저장) |
| `cctv_crowd_dashboard/models/best_mae.pth` | P2PNet 학습 가중치 (저장소 포함) |

## 주의

- ROI json 없으면 분석 안 됨 → `run_setup_roi.bat` 먼저 실행.
- GPU: NVIDIA + 드라이버 있으면 자동 사용(`cuda:0`), 없으면 자동 CPU.
- 최초 `setup.bat`는 PyTorch 다운로드로 수 분~수십 분(네트워크 따라) 걸림.

## 수동 실행 (bat 없이)

```bash
python -m venv venv
venv\Scripts\activate
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cd cctv_crowd_dashboard
python -m analyzer.setup_roi --video data\내영상.mp4   # ROI 설정
python -m server.main                                  # 서버
# 또는 분석 단독:
python -m analyzer.analyzer --video data\내영상.mp4 --show
```
