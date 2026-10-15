"""
설정 모음. 환경/현장에 맞게 여기 값만 바꾸면 됩니다.
"""
import os

# ── 모델 경로 ─────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
YUNET_PATH   = os.path.join(MODELS_DIR, "yunet.onnx")     # 얼굴 검출
FERPLUS_PATH = os.path.join(MODELS_DIR, "ferplus.onnx")   # (구) FER+ — 미사용
# 감정 분류: HSEmotion 모델명. 첫 실행 시 ~/.hsemotion/ 에 자동 다운로드(또는 미리 배치).
HSEMOTION_MODEL = "enet_b0_8_best_afew"

# ── 카메라 ────────────────────────────────────────────────
CAM_SIZE     = (1280, 720)   # 캡처 해상도
# 발화 1회 동안 얼굴 프레임 몇 장을 뜰지 (다수결용). 캡처 자체는 가벼움.
FRAMES_PER_UTTERANCE = 3
# 카메라를 거꾸로/옆으로 달았으면 여기서 뒤집는다. 감정 모델은 똑바로 선 얼굴을 기대한다.
CAM_HFLIP = os.getenv("CAM_HFLIP", "false").lower() == "true"
CAM_VFLIP = os.getenv("CAM_VFLIP", "false").lower() == "true"

# ── 얼굴 트래킹 (Pixy2 팬틸트) ─────────────────────────────
# Pi 카메라는 Pixy2 와 같은 틸트 플레이트에 붙어 있다고 가정한다.
# 모터가 돌면 카메라도 같이 돌기 때문에, 카메라 화면 속 얼굴을 가운데로 맞추면 된다.
TRACK_ENABLED = os.getenv("TRACK_ENABLED", "true").lower() == "true"
# face = Pi 카메라 + YuNet 얼굴 검출(기본) / pixy = Pixy2 색 시그니처(피부색·마커)
TRACK_SOURCE  = os.getenv("TRACK_SOURCE", "face").lower()
PIXY_SIGNATURES = (1, 2, 3)        # TRACK_SOURCE=pixy 일 때 따라갈 시그니처 번호
TRACK_DETECT_WIDTH = 320           # 트래킹용 검출은 축소 프레임에서 (빠르게)
TRACK_FPS = 15                     # 제어 루프 목표 주기

# 제어 게인. 오차는 화면 반폭 기준 -1 ~ 1, 출력은 서보 위치(0~1000) 증분.
# Pixy2 공식 pan_tilt_demo 의 게인을 같은 단위로 환산한 값에서 시작했다.
PAN_KP,  PAN_KD  = 60.0, 30.0
TILT_KP, TILT_KD = 45.0, 25.0
TRACK_DEADBAND = 0.06              # 가운데 근처 이만큼은 무시 → 떨림·모터 소음 감소
TRACK_MAX_STEP = 40                # 한 번에 움직일 최대 서보 위치
# 모터가 얼굴 반대로 도망가면 해당 축만 -1 로 바꾼다. (설치 방향에 따라 다름)
PAN_DIR  = int(os.getenv("PAN_DIR", "1"))
TILT_DIR = int(os.getenv("TILT_DIR", "1"))

# 얼굴을 놓쳤을 때: 잠깐은 그 자리에서 기다리고, 그 뒤엔 천천히 정면(홈)으로.
TRACK_LOST_HOLD_SEC = 2.0
TRACK_HOME = (500, 500)            # (pan, tilt) 홈 자세 — 어르신이 주로 앉는 쪽으로 맞춤
TRACK_HOME_STEP = 4                # 홈으로 돌아갈 때 프레임당 이동량

# ── 얼굴 검출(YuNet) ──────────────────────────────────────
FACE_SCORE_THRESHOLD = 0.5
FACE_NMS_THRESHOLD   = 0.3

# ── 감정 라벨 ─────────────────────────────────────────────
# [A] HSEmotion 라벨. [B] 모델이 있으면 그쪽 라벨이 우선.

# [A] 지금 — HSEmotion (AffectNet 8감정). 데모용, FER+보다 정확.
EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]
EMOTION_KO = {"Anger": "분노", "Contempt": "못마땅함", "Disgust": "불쾌", "Fear": "두려움",
              "Happiness": "기쁨", "Neutral": "중립", "Sadness": "슬픔", "Surprise": "놀람"}

# [B] 직접 학습한 5감정 모델 (emotion/train/). 아래 파일이 있으면 vision.py 가
#     [A] 대신 자동으로 이 모델을 쓴다. 라벨은 ONNX 메타데이터에 같이 들어 있고,
#     아래 목록은 메타데이터가 없을 때만 쓰는 예비값. 한글 라벨 그대로 백엔드로 간다.
CUSTOM_EMOTION_PATH   = os.path.join(MODELS_DIR, "emotion_ko5.onnx")
CUSTOM_EMOTION_LABELS = ["기쁨", "슬픔", "분노", "불안", "중립"]
