import cv2
import mimetypes
import copy
import xlsxwriter
import os
import string
import numpy as np
from line_detector import LineDetector
from line_tracker import LineTracker
from line_counter import LineCounter
from utils import Utils
from grapher import Grapher
from datetime import datetime

class Runner:
    def __init__(self, inputFile):
        self._inputFile = inputFile
            
    def run(self):
        if self._isVideo():
            self._handleVideo()
        else:
            self._handleImage()
    
    #判斷輸入是否為影片。
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
    
    #判斷影片為上行或下行。
    def _isUpward(self):
        #追蹤影片前十幀的斜紋，透過斜紋位置的改變得知為上行或下行。
        #讀入影片。
        video = cv2.VideoCapture(self._inputFile)
        nFrames = 10
        #斜紋偵測器。
        detector = LineDetector()
        #斜紋追蹤器。
        tracker = None
        #第一幀的斜紋追蹤結果。
        firstFrameTrackedLines = None
        #斜紋位置在Y方向上的變化量。
        yOffset = None
        while True:
            #讀取該幀。
            ret, frame = video.read()
            #如果該幀讀取異常，則終止迴圈。
            if not ret:
                break
            #獲得斜紋偵測結果。
            detectResult = detector.detect(frame)
            if detectResult is not None:
                #獲得斜紋、斜率、角度。
                lines, _, _ = detectResult
                #初始化斜紋追蹤器。
                if tracker is None:
                    tracker = LineTracker(len(lines))
                #獲得斜紋追蹤器正在追蹤的斜紋。
                trackedLines = tracker.track(lines)
                if int(video.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
                    firstFrameTrackedLines = copy.deepcopy(trackedLines)
            if int(video.get(cv2.CAP_PROP_POS_FRAMES)) == nFrames + 1:
                yOffset = Utils.getLineCentroid((trackedLines[0][0][0], trackedLines[0][0][1]))[1] - Utils.getLineCentroid((firstFrameTrackedLines[0][0][0], firstFrameTrackedLines[0][0][1]))[1]
                break
        return yOffset < 0

    #處理輸入為影片的情況。
    def _handleVideo(self):
        #讀入影片。
        video = cv2.VideoCapture(self._inputFile)
        #影片高。
        frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #影片幀數。
        frameNumber = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #鋼纜移動方向。
        upward = self._isUpward()
        #界線的位置，為影片最上方往下 (0.35 * 影片高度)。
        borderY = int(round(0.35 * frameHeight))
        #斜紋偵測器。
        detector = LineDetector(borderY, upward = upward)
        #斜紋追蹤器。
        tracker = None
        #斜紋計數器。
        counter = None
        #每組的斜紋數量、每組的累積斜紋數量、每組的斜紋補償數量、每組的累積斜紋補償數量。
        counts, cumCounts, compensateCounts, cumCompensateCounts = (None, None, None, None)
        #每組每一幀的斜紋。
        cumLines = []
        #一幀一幀的讀取影片。
        while True:
            #讀取該幀。
            ret, frame = video.read()
            #如果該幀讀取異常，則終止迴圈。
            if not ret:
                raise Exception('Can not properly read video frame.')
                break
            #獲得斜紋偵測結果。
            detectResult = detector.detect(frame)
            #如果能正常偵測，則進行後續的追蹤與計數，否則跳過這一幀。
            if detectResult is not None:
                #獲得斜紋、斜率、角度。
                groupedLines, groupedSlopes, groupedAngles = detectResult
                cumLines.append(groupedLines)
                #初始化斜紋追蹤器。
                if tracker is None:
                    tracker = LineTracker(len(groupedLines), borderY = borderY, upward = upward)
                #初始化斜紋計數器。
                if counter is None:
                    counter = LineCounter(len(groupedLines), borderY, upward = upward) 
                #獲得斜紋追蹤器正在追蹤的斜紋。
                trackedLines = tracker.track(groupedLines)
                #透過斜紋計數器計算每組的斜紋數量、每組的累積斜紋數量、每組的斜紋補償數量、每組的累積斜紋補償數量。
                counts, cumCounts, compensateCounts, cumCompensateCounts = counter.count(trackedLines) 
                #分別用不同顏色標記每條鋼纜上的斜紋。
                for index, group in enumerate(trackedLines):   
                    for _, l in group.items():
                        cv2.line(frame, l[0], l[1], Utils.groupColors()[index], 1)
                #畫出界線。
                cv2.line(frame, (0, borderY), (frame.shape[1], borderY), (0, 0, 0), 2)
                #標示影片時間軸。
                h, m, s = Utils.milliseconds2HMS(video.get(cv2.CAP_PROP_POS_MSEC))
                cv2.putText(frame, '{}:{}:{}'.format(h, m, s), (10 , frameHeight - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                #標示每條鋼纜的斜紋數量。
                for index, c in enumerate(counts):
                    cv2.putText(frame, str(c), (10 + 100 * index, frameHeight - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Utils.groupColors()[index], 2)
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', frame)
            #如果影片已結束或是按下鍵盤Q中途停止影片，則輸出最後一幀的畫面。
            if int(video.get(cv2.CAP_PROP_POS_FRAMES)) == frameNumber or cv2.waitKey(1) == ord('q'):
                folderName = datetime.now().strftime('%Y%m%d%H%M%S')
                if not os.path.exists('Result/{}'.format(folderName)):
                    os.makedirs('Result/{}'.format(folderName))
                cv2.imwrite('Result/{}/last_frame.png'.format(folderName), frame)
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
     
    #處理輸入為圖片的情況。 
    def _handleImage(self):
        image = cv2.imread(self._inputFile)
        #斜紋偵測器。
        detector = LineDetector(isVideo = False)
        #獲得斜紋偵測結果。
        detectResult = detector.detect(image)
        #獲得斜紋、斜率、角度。
        groupedLines, groupedSlopes, groupedAngles = detectResult
        #分別用不同顏色標記每條鋼纜上的斜紋。
        for index, group in enumerate(groupedLines):
            for l in group:
                cv2.line(image, l[0], l[1], Utils.groupColors()[index], 1)
        #顯示結果圖片。
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', image)
        #輸出結果圖片與Excel檔。
        folderName = datetime.now().strftime('%Y%m%d%H%M%S')
        if not os.path.exists('Result/{}'.format(folderName)):
            os.makedirs('Result/{}'.format(folderName))
        cv2.imwrite('Result/{}/result.png'.format(folderName), image)
        workbook = xlsxwriter.Workbook('Result/{}/slope.xlsx'.format(folderName))
        worksheet = workbook.add_worksheet()
        fmt = workbook.add_format()
        fmt.set_align('center')
        #寫入欄位名稱。
        uppercases = string.ascii_uppercase
        for index in range(len(groupedLines)):
            worksheet.merge_range('{}1:{}1'.format(uppercases[2 * index], uppercases[2 * index + 1]), '鋼纜{}'.format(index + 1), fmt)
            worksheet.write(1, 2 * index, '斜率', fmt)
            worksheet.write(1, 2 * index + 1, '角度', fmt)
        #寫入斜率。
        for index1, group in enumerate(groupedSlopes):
            for index2, s in enumerate(group):
                worksheet.write(index2 + 2, 2 * index1, round(s, 3), fmt)
        #寫入角度。
        for index1, group in enumerate(groupedAngles):
            for index2, a in enumerate(group):
                worksheet.write(index2 + 2, 2 * index1 + 1, round(a, 3), fmt)
        rowIndexOfAverage = max([len(group) for group in groupedLines]) + 3
        #平均斜率與角度。
        for index in range(len(groupedLines)):
            worksheet.merge_range('{}{}:{}{}'.format(uppercases[2 * index], rowIndexOfAverage, uppercases[2 * index + 1], rowIndexOfAverage), '平均', fmt)
        #寫入平均斜率。
        for index, group in enumerate(groupedSlopes):
            worksheet.write(rowIndexOfAverage, 2 * index, round(np.mean(group), 3), fmt)
        #寫入平均角度。
        for index, group in enumerate(groupedAngles):
            worksheet.write(rowIndexOfAverage, 2 * index + 1, round(np.mean(group), 3), fmt)
        workbook.close()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
                
        
        
        
        
        
        
