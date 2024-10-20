import cv2
from cvzone.PoseModule import PoseDetector

VIDEO_PATH = r"D:\praca_wlasna\mocapSolutions\anim\points\mocapVid.mp4"
TEXT_FILE_PATH = r"D:\praca_wlasna\mocapSolutions\mediapipe_guide\animationFile2.txt"

# add text parameters
VIDEO_TEXT_POS = (50, 50)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_TEXT_SCALE = 1
VIDEO_TEXT_COLOR = (0,255,255)
VIDEO_TEXT_THICK = 2

# save video parameters
TITLE = r"D:\praca_wlasna\mocapSolutions\anim\points\mocapWithKeypoints.avi"
FPS = 25
SIZE = (720, 486)


video = cv2.VideoCapture(VIDEO_PATH)
# Video writer to store video in TITLE
result = cv2.VideoWriter(TITLE, cv2.VideoWriter_fourcc(*'MJPG'), 25, SIZE)

detector = PoseDetector()
posList = []
frame_counter = 0
while True:
    frame_counter += 1
    success, img = video.read()
    if success != 1:
        print("Video has ended")
        break
    img = detector.findPose(img)
    cv2.putText(img, str(frame_counter), VIDEO_TEXT_POS, FONT, VIDEO_TEXT_SCALE, VIDEO_TEXT_COLOR, VIDEO_TEXT_THICK , cv2.LINE_4)
    lmList, bboxInfo = detector.findPosition(img)

    if bboxInfo:
        lmString = ''
        for lm in lmList:
            lmString += f'{lm[0]},{img.shape[0] - lm[1]},{lm[2]},'
        posList.append(lmString)

    result.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

video.release()
result.release()
print("Do you want to save animationData? Press s. Else press any other key?")
input1 = input()
if (input1 == 's'):
    with open(TEXT_FILE_PATH, 'w') as f:
        f.writelines(["%s\n" % item for item in posList])
        print("Video saved")