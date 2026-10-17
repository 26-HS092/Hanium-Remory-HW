#!/bin/sh
# 부팅 때 얼굴 화면(chromium 키오스크)을 띄운다. ~/.config/autostart/mori-face.desktop 이 부른다.
# 로그인 직후 바로 켜면 데스크톱이 덜 떠서 하얀 화면에서 멈추는 일이 있어 잠깐 기다린다.
sleep 10
exec chromium --ozone-platform=wayland --kiosk --noerrdialogs --disable-infobars \
  --disable-session-crashed-bubble --password-store=basic \
  file:///home/han/project1/face/mori_face.html
