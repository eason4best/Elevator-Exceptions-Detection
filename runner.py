import cv2
import mimetypes
from line_detector import LineDetector
from line_tracker import LineTracker
from line_counter import LineCounter
from utils import Utils
from grapher import Grapher

class Runner:
    def __init__(self, inputFile):
        self._inputFile = inputFile
            
    def run(self):
        if self._isVideo():
            self._handleVideo()
        else:
            self._handleImage()
        
    def _isVideo(self):
        mimeType = mimetypes.guess_type(self._inputFile)[0]
        if mimeType is None:
            raise Exception('Can not identify input file type.')
        fileType = mimeType.split('/')[0]
        if fileType == 'video':
            return True
        elif fileType == 'image':
            return False
        else:
            raise Exception('Input file must be image or video.')
    
    def _isUpward(self):
        return True

    def _handleVideo(self):
        #讀入影片。
        video = cv2.VideoCapture(self._inputFile)
        #影片長寬。
        frameWidth = None
        frameHeight = None
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
        #每組每一幀的斜紋。
        cumLines = []
        #一幀一幀的讀取影片。
        while True:
            #讀取該幀。
            ret, frame = video.read()
            #如果影片已結束或是該幀讀取異常，則終止程式。
            if not ret:
                break
            #獲得影片長寬。
            if frameWidth is None or frameHeight is None:
                frameWidth = frame.shape[1]
                frameHeight = frame.shape[0]
            #設定界線的位置。
            if borderY is None:
                #將界線位置設為影片最上方往下 (0.35 * 影片高度) 的位置。
                borderY = int(round(0.35 * frame.shape[0]))
            #初始化斜紋偵測器。
            if detector is None:
                detector = LineDetector(borderY, isVideo = True, upward = self._isUpward())
            #獲得斜紋偵測結果。
            detectResult = detector.detect(frame)
            #如果能正常偵測，則進行後續的追蹤與計數，否則跳過這一幀。
            if detectResult is not None:
                #獲得斜紋、斜率、角度。
                lines, slopes, angles = detectResult
                cumLines.append(lines)
                #初始化斜紋追蹤器。
                if tracker is None:
                    tracker = LineTracker(len(lines), borderY, upward = self._isUpward())
                #初始化斜紋計數器。
                if counter is None:
                    counter = LineCounter(len(lines), borderY, upward = self._isUpward()) 
                #獲得斜紋追蹤器正在追蹤的斜紋。
                trackedLines = tracker.track(lines)
                #透過斜紋計數器計算每組的斜紋數量、每組的累積斜紋數量、每組的斜紋補償數量、每組的累積斜紋補償數量。
                counts, cumCounts, compensateCounts, cumCompensateCounts = counter.count(trackedLines) 
                #分別用不同顏色標記每條鋼纜上的斜紋。
                for index, group in enumerate(trackedLines):   
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
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break 
        video.release()
        #初始化圖表繪圖器。
        grapher = Grapher()
        #畫出斜紋累積總數、斜紋累積補償數的折線圖與斜紋總數、斜紋補償數的表格。
        grapher.plotCountsGraphAndTable(counts, compensateCounts, cumCounts, cumCompensateCounts)
        #畫出斜紋斜率隨時間變化的圖。
        #範圍一（滑輪上）
        range1 = (int(round(0.35 * frameHeight)), int(round(0.55 * frameHeight)))
        #範圍二（滑輪下）
        range2 = (int(round(0.9 * frameHeight)), int(round(1.0 * frameHeight)))
        grapher.plotSlopesGraph(cumLines, [range1, range2])
        cv2.destroyAllWindows()
        
    def _handleImage(self):
        image = cv2.imread(self._inputFile)
        #斜紋偵測器。
        detector = LineDetector(isVideo = False)
        #獲得斜紋偵測結果。
        detectResult = detector.detect(image)
        #獲得斜紋、斜率、角度。
        lines, slopes, angles = detectResult
        #分別用不同顏色標記每條鋼纜上的斜紋。
        for index, group in enumerate(lines):
            for line in group:
                cv2.line(image, line[0], line[1], Utils.groupColors()[index], 1)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
                
        
        
        
        
        
        
