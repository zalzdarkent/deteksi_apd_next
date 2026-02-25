import cv2

rtsp_url = "rtsp://admin:abcd1234@172.16.20.53:554/Streaming/Channels/101"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Cannot open the RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot grab frame from RTSP stream.")
        break

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()