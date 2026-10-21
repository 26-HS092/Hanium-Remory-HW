#!/bin/sh
# 얼굴 화면에서 마우스 포인터를 아예 안 보이게 한다. 한 번만 돌리면 되고, 다시 로그인(또는
# 재부팅)하면 적용된다.
#
# 터치 화면이 마우스 흉내(rc.xml 의 mouseEmulation)를 내서, 화면을 누를 때마다 labwc 가
# 포인터를 그린다. CSS(cursor:none)는 chromium 안에서만 먹으니, labwc 가 쓰는 커서
# 테마 자체를 투명한 것으로 바꾼다. 되돌리려면 ~/.config/labwc/environment 의
# XCURSOR_THEME 줄을 지운다.
set -e

THEME=mori-none
DIR="$HOME/.icons/$THEME/cursors"
mkdir -p "$DIR"

# 1x1 투명 픽셀 하나짜리 Xcursor 파일.
python3 - "$DIR/default" <<'EOF'
import struct, sys
image = struct.pack("<9I", 36, 0xfffd0002, 1, 1, 1, 1, 0, 0, 0) + b"\0\0\0\0"
header = b"Xcur" + struct.pack("<3I", 16, 0x10000, 1)
toc = struct.pack("<3I", 0xfffd0002, 1, 16 + 12)
open(sys.argv[1], "wb").write(header + toc + image)
EOF

# 프로그램마다 부르는 커서 이름이 달라서, 흔한 이름은 모두 투명 커서를 가리키게 한다.
for name in left_ptr arrow top_left_arrow pointer hand1 hand2 text xterm wait watch \
    progress left_ptr_watch crosshair cross move grab grabbing not-allowed help \
    context-menu cell all-scroll col-resize row-resize ew-resize ns-resize \
    nesw-resize nwse-resize n-resize s-resize e-resize w-resize ne-resize \
    nw-resize se-resize sw-resize sb_h_double_arrow sb_v_double_arrow \
    top_side bottom_side left_side right_side top_left_corner top_right_corner \
    bottom_left_corner bottom_right_corner fleur; do
  ln -sf default "$DIR/$name"
done
printf '[Icon Theme]\nName=%s\n' "$THEME" > "$HOME/.icons/$THEME/index.theme"

ENV="$HOME/.config/labwc/environment"
mkdir -p "$(dirname "$ENV")"
touch "$ENV"
sed -i '/^XCURSOR_THEME=/d' "$ENV"
echo "XCURSOR_THEME=$THEME" >> "$ENV"
echo "✅ 투명 커서 테마 적용 — 다시 로그인하거나 재부팅하면 포인터가 사라집니다."
