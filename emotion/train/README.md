# 얼굴 감정 모델 학습 (5감정)

지금 인형의 HSEmotion(AffectNet, 서양인 위주 8감정)을 출발점으로 삼아
AI Hub 한국인 얼굴로 다시 학습한 **기쁨 / 슬픔 / 분노 / 불안 / 중립** 5감정 모델을 만든다.

- 라벨 합치기 (`labels.py`): 상처→슬픔, 당황→불안. 나머지는 그대로.
- 백엔드 `emotion_codes.py` 가 5개 한글 라벨을 이미 다 알아들어서 백엔드·앱은 고칠 게 없다.
- 구조가 같은 EfficientNet-B0 이라 파이에서의 속도도 지금과 같다.
- 얼굴 crop·전처리는 인형과 같은 `emotion/face_crop.py` 를 쓴다. 학습 때와 현장의 얼굴 모양이 같아야 정확도가 실제로 오른다.

## 0. 준비

```bash
cd emotion/train
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p ../../models
curl -L -o ../../models/yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
# 기준선 비교용 (지금 인형 모델의 ONNX)
curl -L -o ../../models/enet_b0_8_best_afew.onnx \
  https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx
```

데이터: [AI Hub — 한국인 감정인식을 위한 복합 영상](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=82)
을 신청·다운로드해서 Training / Validation 압축을 한 폴더 아래에 푼다.

## 1. 얼굴 자르기 (한 번만)

```bash
# 처음엔 작게 돌려서 라벨이 제대로 읽히는지부터 확인
.venv/bin/python prepare_aihub.py --src ~/data/aihub82 --out data/crops --max-per-class 300
# 괜찮으면 전부 (끊겨도 다시 실행하면 이어서 함)
.venv/bin/python prepare_aihub.py --src ~/data/aihub82 --out data/crops
```

마지막에 split·감정별 장수가 찍힌다. 어느 감정이 0장이면 라벨을 못 읽은 것이니 JSON 구조를 확인할 것.
작업자 3명 중 2명 이상이 같은 감정이라고 한 사진만 쓴다(`--min-agree`).

## 2. 학습

```bash
.venv/bin/python train.py --data data/crops --epochs 3 --limit 5000   # 한 바퀴 확인
.venv/bin/python train.py --data data/crops                           # 본 학습
```

epoch 마다 val 정확도·macro-F1 이 찍히고, 가장 좋았던 것이 `models/emotion_ko5.onnx` 로 나온다.

## 3. 평가 — 지금 모델보다 나아졌는지

```bash
.venv/bin/python evaluate.py --data data/crops/test
```

AI Hub Validation(학습에 한 번도 안 쓴 사람들)으로 HSEmotion 과 새 모델을 같은 5감정 기준으로 비교한다.
정확도보다 **macro-F1** 을 볼 것 — 중립만 찍어도 정확도는 높게 나온다.

AI Hub 는 대부분 젊은~중년 배우의 연기 표정이라 어르신 표정과는 차이가 있다.
보호자 동의를 받아 어르신 사진을 감정별 폴더(`기쁨/ 슬픔/ …`)에 몇십 장씩 모아 두면
원본 그대로 인형과 같은 방식으로 잘라 평가할 수 있다. 실제 현장 정확도는 이 숫자가 가장 정확하다.

```bash
.venv/bin/python evaluate.py --data ~/elder_faces --detect
```

인형 카메라로 직접 찍을 때는 `emotion/capture_faces.py` 를 쓴다(main.py 는 끄고).
번호(1 기쁨 · 2 슬픔 · 3 분노 · 4 불안 · 5 중립)를 누르면 인형과 같은 카메라 설정으로 찍어
`~/인형사진/<조건>/<감정>/<이름>_<번호>.jpg` 에 저장한다. 얼굴을 못 찾은 사진은 저장하지 않는다.

```bash
python emotion/capture_faces.py --who 수아                 # 조건 '기본'
python emotion/capture_faces.py --who 수아 --cond 어두움 --delay 3
.venv/bin/python evaluate.py --data ~/인형사진/기본 --detect
```

## 4. 인형에 넣기

`models/emotion_ko5.onnx` 는 저장소에 같이 올라가 있어 파이에서 `git pull` 하면 끝.
(다시 학습했으면 이 파일을 커밋한다. 원본·crop 사진은 절대 올리지 않는다 — 아래 출처 참고)
`vision.py` 가 파일이 있으면 자동으로 새 모델을, 없으면 HSEmotion 을 쓴다
(시작할 때 `🙂 감정 모델: …` 로그로 확인). 되돌리려면 파일만 지우면 된다.

## 결과 (2026-10, Training `_01` 8.4만 장, 12 epoch 중 6번째)

같은 5감정 기준 macro-F1. test 는 AI Hub Validation 중 **학습에 한 번도 안 나온 사람**(747명, 32,642장)만 썼다.
AI Hub Validation 은 사진 단위로 나뉘어 있어 학습 사람과 250명이 겹치기 때문이다.

| | val | test (처음 보는 사람) |
|---|---|---|
| HSEmotion (기존) | 0.728 | 0.722 |
| emotion_ko5 | 0.842 | 0.831 |

참고로 라벨 작업자 1명이 나머지 2명의 다수결과 맞는 비율이 84% 라, 이 데이터로는 이 근처가 천장이다.

## 데이터 출처

이 모델은 과학기술정보통신부와 한국지능정보사회진흥원의 지원으로 구축된
AI Hub [「한국인 감정인식을 위한 복합 영상」](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=82)
데이터(aihub.or.kr)를 이용해 학습했습니다.
원본 데이터와 그로부터 잘라낸 얼굴 사진은 [AI Hub 데이터 이용정책](https://aihub.or.kr/intrcn/guid/usagepolicy.do)에
따라 저장소에 포함하지 않습니다(`emotion/train/data/` 는 `.gitignore`).
