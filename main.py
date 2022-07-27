import cv2
from line_detector import LineDetector
from line_tracker import LineTracker
from line_counter import LineCounter
from utils import Utils

video = cv2.VideoCapture('IS7(D).avi')
vw = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1024, 1024))
detector = None
tracker = None
counter = None
borderY = None
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print('Cant receive frame')
        break
    if borderY is None:
        borderY = int(round(0.35 * frame.shape[0]))
    if detector is None:
        detector = LineDetector(borderY, upward = False)
    detectResult = detector.detect(frame)
    if detectResult is not None:
        groupedFittedEllipses, lines, slopes, angles = detectResult
        if tracker is None:
            tracker = LineTracker(len(lines), borderY, upward = False)
        if counter is None:
            counter = LineCounter(len(lines), borderY, upward = False) 
        lines = tracker.track(lines)
        counts = counter.count(lines) 
        #標示斜紋。
        for index, group in enumerate(lines):   
            for _, line in group.items():
                cv2.line(frame, line[0], line[1], Utils.groupColors()[index], 1)
        #畫出界線。
        cv2.line(frame, (0, borderY), (frame.shape[1], borderY), (0, 0, 0), 2)
        #標示影片時間軸。
        h, m, s = Utils.milliseconds2HMS(video.get(cv2.CAP_PROP_POS_MSEC))
        cv2.putText(frame, '{}:{}:{}'.format(h, m, s), (10 , frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        #標示每條鋼纜斜紋數量。
        for index, c in enumerate(counts):
            cv2.putText(frame, str(c), (10 + 100 * index, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Utils.groupColors()[index], 2)
        vw.write(frame)
        cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break 
cv2.destroyAllWindows()
video.release()
vw.release()