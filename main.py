import cv2
from line_detector import LineDetector
from line_tracker import LineTracker
from line_counter import LineCounter
from utils import Utils
from grapher import Grapher

#讀入影片，將路徑改為影片所在路徑。
video = cv2.VideoCapture('/Users/mac/Desktop/01(D).avi') 
#斜紋偵測器。
detector = None
#斜紋追蹤器。
tracker = None
#斜紋計數器。
counter = None
#界線的位置。
borderY = None
#每組的斜紋數量、每組的累積斜紋數量、每組的斜紋補償數量、每組的累積斜紋補償數量。
counts, cumCounts, compensateCounts, cumCompensateCounts = (None, None, None, None)
#一幀一幀的讀取影片。
while True:
    ret, frame = video.read()
    #如果影片已結束或是該幀讀取異常，則終止程式。
    if not ret:
        break
    #設定界線的位置。
    if borderY is None:
        #將界線位置設為影片最上方往下 (0.35 * 影片高度) 的位置。
        borderY = int(round(0.35 * frame.shape[0]))
    #初始化斜紋偵測器。
    if detector is None:
        detector = LineDetector(borderY, upward = False)
    #獲得斜紋偵測結果。
    detectResult = detector.detect(frame)
    #如果能正常偵測，則進行後續的追蹤與計數，否則跳過這一幀。
    if detectResult is not None:
        #獲得斜紋、斜率、角度。
        lines, slopes, angles = detectResult
        #初始化斜紋追蹤器。
        if tracker is None:
            tracker = LineTracker(len(lines), borderY, upward = False)
        #初始化斜紋計數器。
        if counter is None:
            counter = LineCounter(len(lines), borderY, upward = False) 
        #獲得斜紋追蹤器正在追蹤的斜紋。
        lines = tracker.track(lines)
        #透過斜紋計數器計算每組的斜紋數量、每組的累積斜紋數量、每組的斜紋補償數量、每組的累積斜紋補償數量。
        counts, cumCounts, compensateCounts, cumCompensateCounts = counter.count(lines) 
        #分別用不同顏色標記每條鋼纜上的斜紋。
        for index, group in enumerate(lines):   
            for _, line in group.items():
                cv2.line(frame, line[0], line[1], Utils.groupColors()[index], 1)
        #畫出界線。
        cv2.line(frame, (0, borderY), (frame.shape[1], borderY), (0, 0, 0), 2)
        #標示影片時間軸。
        h, m, s = Utils.milliseconds2HMS(video.get(cv2.CAP_PROP_POS_MSEC))
        cv2.putText(frame, '{}:{}:{}'.format(h, m, s), (10 , frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        #標示每條鋼纜的斜紋數量。
        for index, c in enumerate(counts):
            cv2.putText(frame, str(c), (10 + 100 * index, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Utils.groupColors()[index], 2)
        cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break 
video.release()
#初始化圖表繪圖器。
grapher = Grapher()
#畫出斜紋累積總數、斜紋累積補償數的折線圖與斜紋總數、斜紋補償數的表格。
grapher.plotCountsGraphAndTable(counts, compensateCounts, cumCounts, cumCompensateCounts)
cv2.destroyAllWindows()






