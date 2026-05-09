# save as test_cam.py and run: python test_cam.py
import cv2

for idx in range(0, 5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    print(f"Index {idx} via DSHOW: {'OK' if ok else 'FAIL'}")
    cap.release()

print("---- MSMF ----")
for idx in range(0, 5):
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    ok = cap.isOpened()
    print(f"Index {idx} via MSMF: {'OK' if ok else 'FAIL'}")
    cap.release()