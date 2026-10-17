"""
main_fish.py
------------
예전 실행 파일. 이제 main.py 가 Fish Audio 로 말하는 게 기본이라 main.py 와 같다.

    python main.py        # 또는 python main_fish.py

CosyVoice 로 돌리려면 .env 에 TTS_ENGINE=cosyvoice 를 넣는다.
"""

import os

from dotenv import load_dotenv

# main 을 import 하기 전에 정해야 한다(main.py 가 import 시점에 TTS_ENGINE 을 읽음).
# .env 를 먼저 읽어서, .env 에 FISH_REFERENCE_ID 가 있으면 그 값이 우선한다.
load_dotenv()
os.environ["TTS_ENGINE"] = "fish"
os.environ.setdefault("FISH_REFERENCE_ID", "c2045f5c866b4222a4a2c30b5b051f7e")
# 모델을 안 정하면 유료 기본 모델(s2.1-pro)로 가서 API 크레딧이 없으면 402 가 난다.
os.environ.setdefault("FISH_MODEL", "s2.1-pro-free")

import main  # noqa: E402

if __name__ == "__main__":
    main.main()
