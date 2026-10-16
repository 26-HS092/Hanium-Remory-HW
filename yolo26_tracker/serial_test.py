import time
import serial

PORT = "/dev/ttyACM0"

with serial.Serial(PORT, 115200, timeout=3) as arduino:
    # USB 연결 시 아두이노가 재시작하므로 기다림
    time.sleep(2)
    arduino.reset_input_buffer()

    command = "84,108"
    arduino.write((command + "\n").encode())
    arduino.flush()

    print("보낸 명령:", command)

    reply = arduino.readline().decode(
        errors="replace"
    ).strip()

    if reply == f"RECEIVED: {command}":
        print("통신 성공:", reply)
    else:
        print("예상과 다른 응답:", repr(reply))