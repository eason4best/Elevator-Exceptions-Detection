import cv2
import numpy as np

class LineDetector:
    def __init__(self, borderY, upward = True):
        #界線的位置。
        self._borderY = borderY
        #鋼纜移動方向。
        self._upward = upward
        #用來儲存每條鋼纜上所有斜紋的平均中心點X座標。
        self._groupMeanCenterXs = []
    
    #偵測傳入的原始圖片[img]的斜紋。
    def detect(self, img):
        self._img = img
        #將原始圖片進行預處理。
        preprocessImg = self._preprocess(self._img)
        #使用Canny邊緣檢測找出斜紋的邊緣。
        canny = cv2.Canny(preprocessImg, 100, 100, apertureSize = 3)
        #找出圖片中的輪廓並對每一個輪廓都適配（Fit）一個橢圓。 
        fittedEllipses = self._findContourAndFitEllipse(canny)
        #移除屬於離群值的橢圓（面積異常小或大、角度異常小或大）。
        fittedEllipses = self._removeOutliersEllipses(fittedEllipses, lowerFactor = 2.0, upperFactor = 3.0)
        #找出橢圓長軸(以兩個端點表示)，用來表示鋼纜上的斜紋。
        fittedEllipses = self._findEllipseMajorAxes(fittedEllipses)
        #將橢圓照著長軸起點的X座標由小到大（圖片中由左到右）排序。
        fittedEllipses = sorted(fittedEllipses, key = lambda fe: fe['majorAxe']['startPoint'][0])
        #將橢圓分組，正常情況下鋼纜有幾條橢圓就有幾組。
        groupedFittedEllipses = self._groupEllipses(fittedEllipses)
        #如果分出來的組數異常，則不做後續處理，直接回傳None。
        if len(groupedFittedEllipses) != len(self._groupMeanCenterXs) and len(self._groupMeanCenterXs) > 0:
            return None
        #使用分好組的橢圓來獲得代表斜紋的直線，與該線的斜率與角度。
        self._lines, self._slopes, self._angles = self._computeSlopeAndAngle(groupedFittedEllipses)
        #如果鋼纜移動方向為向上。
        if self._upward:
            #將每組橢圓照著長軸起點旋轉後的Y座標由小到大（圖片中由上到下）排序。
            for i in range(0, len(groupedFittedEllipses)):
                sortResult = sorted(zip(groupedFittedEllipses[i], self._lines[i], self._slopes[i], self._angles[i]), key = lambda x: self._rotatePointAroundImageCenter(x[0]['majorAxe']['startPoint'], np.mean(self._angles[i]))[1])
                groupedFittedEllipses[i] = [x[0] for x in sortResult]
                self._lines[i] = [x[1] for x in sortResult]
                self._slopes[i] = [x[2] for x in sortResult]
                self._angles[i] = [x[3] for x in sortResult]
        #如果鋼纜移動方向為向下。
        else:
            #將每組橢圓照著長軸起點旋轉後的Y座標由大到小（圖片中由下到上）排序。
            for i in range(0, len(groupedFittedEllipses)):
                sortResult = sorted(zip(groupedFittedEllipses[i], self._lines[i], self._slopes[i], self._angles[i]), key = lambda x: self._rotatePointAroundImageCenter(x[0]['majorAxe']['startPoint'], np.mean(self._angles[i]))[1], reverse = True)
                groupedFittedEllipses[i] = [x[0] for x in sortResult]
                self._lines[i] = [x[1] for x in sortResult]
                self._slopes[i] = [x[2] for x in sortResult]
                self._angles[i] = [x[3] for x in sortResult]
        #將代表同一個斜紋的破碎橢圓接在一起。
        groupedFittedEllipses = self._combineSmallEllipses(groupedFittedEllipses)
        #平移所有橢圓，使同組的橢圓有相同的中心點X座標（這樣計數器在判別斜紋時會更精確）。
        groupedFittedEllipses = self._translateEllipses(groupedFittedEllipses)
        #補齊因光線、陰影等外部條件而沒有被辨識到的斜紋。只有界線之前的斜紋才需要補，因為過了界線有沒有補都不影響計數。
        groupedFittedEllipses = self._compensate(groupedFittedEllipses)
        return (self._lines, self._slopes, self._angles)
    
    #將原始圖片[img]進行預處理。
    def _preprocess(self, img):
        #將原始圖片轉為灰階，以進行後續處理。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #使用高斯模糊消除圖片中的雜訊，避免其干擾後續辨識。
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #將圖片進行二值化，突顯出要辨識的斜紋。
        _, threshold = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
        #使用 Morphological Transformations 清楚分開每條斜紋。 
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        return morph
    
    #找出圖片[img]中的輪廓並對每一個輪廓都適配（Fit）一個橢圓。
    def _findContourAndFitEllipse(self, img):
        #找出圖片中的輪廓。 
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        #存放適配輪廓的橢圓及其他相關資訊。
        fittedEllipse = []
        for c in contours:
            #輪廓至少要由5個以上的點組成，才能用橢圓適配。
            if len(c) > 5:
                #橢圓本身，裡面包含橢圓的中心點，長短軸長度以及傾斜角度。
                ellipse = cv2.fitEllipse(c)
                #利用橢圓的長軸與短軸來計算橢圓面積。
                a, b = ellipse[1]
                #短軸長度。
                minorLength = min(a,b)
                #長軸長度。
                majorLength = max(a,b)
                #計算橢圓的面積。
                area = (minorLength / 2) * (majorLength / 2) * np.pi
                #新增橢圓本身及其面積。
                fittedEllipse.append({'ellipse':ellipse, 'area':area})
        return fittedEllipse
    
    '''
    移除[ellipses]中屬於離群值的橢圓（面積異常小或大、角度異常小或大），[minArea]代表接受的最小橢圓面積，面積小於[minArea]的橢圓會直接被移除。
    [lowerFactor]與[upperFactor]用來設定移除標準的嚴格程度。[lowerFactor]與[upperFactor]越大，被移除的橢圓越少，[lowerFactor]與[upperFactor]越小，被移除的橢圓越多。
    '''
    def _removeOutliersEllipses(slef, ellipses, minArea = 80, lowerFactor = 1.0, upperFactor = 1.0):
        #使用 Median Absolute Deviation(MAD) 方法來檢測橢圓面積與角度的離群值，移除面積異常小或大、角度異常小或大的橢圓。
        #先將面積小於 minArea 的橢圓直接移除，避免其影響中位數（Median）的大小，使得正常面積的橢圓反而被當成離群值。
        ellipses = list(filter(lambda e: e['area'] > minArea, ellipses))
        areas = [e['area'] for e in ellipses]
        angles = [e['ellipse'][2] for e in ellipses]
        medianAreas = np.median(areas)
        medianAngles = np.median(angles)
        diffAreas = np.abs(areas - medianAreas)
        diffAngles = np.abs(angles - medianAngles)
        scalingFactorAreas = np.median(diffAreas)
        scalingFactorAngles = np.median(diffAngles)
        lowerAreas = medianAreas - lowerFactor * scalingFactorAreas
        upperAreas = medianAreas + upperFactor * scalingFactorAreas
        lowerAngles = medianAngles - lowerFactor * scalingFactorAngles
        upperAngles = medianAngles + upperFactor * scalingFactorAngles
        #移除面積異常小或大、角度異常小或大的橢圓。
        ellipses = list(filter(lambda e: (e['area'] < upperAreas and e['area'] > lowerAreas and e['ellipse'][2] < upperAngles and e['ellipse'][2] > lowerAngles), ellipses))
        #橢圓面積之後用不到，先移除。
        ellipses = [{k: v for k, v in e.items() if k != 'area'} for e in ellipses]
        return ellipses
    
    #找出[ellipses]橢圓的長軸(以兩個端點表示)，用來表示鋼纜上的斜紋。
    def _findEllipseMajorAxes(self, ellipses):
        for e in ellipses:
            #橢圓的中心、長短軸與角度。
            (x0, y0), (a, b), angle = e['ellipse']
            majorLength = max(a,b) / 2
            if angle > 90:
                angle = angle - 90
            else:
                angle = angle + 90
            #橢圓長軸的起點。
            startPoint = (int(round(x0 + majorLength * np.cos(np.radians(angle)))), int(round(y0 + majorLength * np.sin(np.radians(angle)))))
            #橢圓長軸的終點。
            endPoint = (int(round(x0 + majorLength * np.cos(np.radians(angle + 180)))), int(round(y0 + majorLength * np.sin(np.radians(angle + 180)))))
            #新增橢圓長軸。
            e['majorAxe'] = {'startPoint': startPoint, 'endPoint': endPoint}
        return ellipses
    
    #將橢圓[ellipses]分組，正常情況下鋼纜有幾條橢圓就有幾組。
    def _groupEllipses(self, ellipses):
        #存放已分組的橢圓，並將第一個橢圓直接先放到第一組。
        groupedEllipses = [[ellipses[0]]]
        for index, MA in enumerate([e['majorAxe'] for e in ellipses[1:]]):
            #該橢圓起點的X座標。
            currentStartPointX = MA['startPoint'][0]
            #該橢圓終點的X座標。
            currentEndPointX = MA['endPoint'][0]
            #該組平均長軸終點的X座標。
            meanGroupEndPointX = np.mean([e['majorAxe']['endPoint'][0] for e in groupedEllipses[-1]])
            #用來判斷該組是否已分完。
            inNextGroup = True
            #將該橢圓與該組所有現有橢圓的位置進行比較，判斷該橢圓是否應被分到該組。
            for previousMA in [e['majorAxe'] for e in groupedEllipses[-1]]:
                previousStartPointX = previousMA['startPoint'][0]
                previousEndPointX = previousMA['endPoint'][0]
                #如果該橢圓與該組任一個現有橢圓有交錯，且其長軸起點的X座標小於該組平均長軸終點的X座標，則將它分到該組。
                if ((currentStartPointX < previousStartPointX and currentEndPointX > previousStartPointX) or (currentEndPointX > previousStartPointX and currentStartPointX < previousEndPointX) or (currentStartPointX > previousStartPointX and currentEndPointX < previousEndPointX) or (currentStartPointX < previousStartPointX and currentEndPointX > previousEndPointX)) and (currentStartPointX < meanGroupEndPointX):
                    groupedEllipses[-1].append(ellipses[index + 1])
                    inNextGroup = False
                    break
            #該橢圓不屬於該組，代表該組已分完。
            if inNextGroup:
                #將該橢圓分到下一組。
                groupedEllipses.append([ellipses[index+1]])
        #少數破碎的橢圓會被演算法單獨分成一組，造成組數異常，所以將橢圓數量極少的組移除（這裡只將橢圓數量大於5的組留下）。
        groupedEllipses = list(filter(lambda g: len(g) > 5 , groupedEllipses))
        return groupedEllipses
    
    #透過[groupedEllipses]來獲得代表斜紋的直線，與該線的斜率與角度。
    def _computeSlopeAndAngle(self, groupedEllipses):
        #存放已分組的橢圓長軸。
        groupedLines = [[0] * len(group) for group in groupedEllipses]
        #存放已分組的斜紋斜率。
        groupedSlopes = [[0] * len(group) for group in groupedEllipses]
        #存放已分組的斜紋角度。
        groupedAngles = [[0] * len(group) for group in groupedEllipses]
        for index1, group in enumerate(groupedEllipses):
            for index2, e in enumerate(group):
                #橢圓長軸起點。
                startPoint = e['majorAxe']['startPoint']
                #橢圓長軸終點。
                endPoint = e['majorAxe']['endPoint']
                #斜紋斜率。
                slope = (startPoint[1] - endPoint[1]) / (endPoint[0] - startPoint[0])
                #斜紋角度。
                angle = np.arctan(slope) * 180 / np.pi
                groupedLines[index1][index2] = (startPoint, endPoint)
                groupedSlopes[index1][index2] = slope
                groupedAngles[index1][index2] = angle
        return (groupedLines, groupedSlopes, groupedAngles)
    
    #將[groupedEllipses]中代表同一個斜紋的破碎橢圓接在一起。
    def _combineSmallEllipses(self, groupedEllipses):
        #存放新的已分組的橢圓。
        newGroupedEllipses = []
        #存放新的已分組的橢圓長軸。
        newGroupedLines = []
        #存放新的已分組的斜紋斜率。
        newGroupedSlopes = []
        #存放新的已分組的斜紋角度。
        newGroupedAngles = []
        for index1, group in enumerate(groupedEllipses):
            newEllipses = []
            newLines = []
            newSlopes = []
            newAngles = []
            #用來判斷上個橢圓是否有跟目前這個橢圓合併。
            previousCombined = False
            for index2, e in enumerate(group[:-1]):
                #上個橢圓已跟目前這個橢圓合併，不用再將這個橢圓與其他橢圓合併。
                if previousCombined:
                    previousCombined = False
                    continue
                #目前這個橢圓的中心與長短軸長度。
                (x1, y1), (a1, b1), _ = e['ellipse']
                #下個橢圓的中心與長短軸長度。
                (x2, y2), (a2, b2), _ = group[index2 + 1]['ellipse']
                #兩個橢圓的中心相連為一垂直線，代表它們不代表同一個斜紋，不用合併。
                if x2 - x1 == 0:
                    #新增目前這個橢圓。
                    newEllipses.append(e)
                    #新增目前這個橢圓的長軸。
                    newLines.append(self._lines[index1][index2])
                    #新增目前這個橢圓的長軸斜率（斜紋斜率）。
                    newSlopes.append(self._slopes[index1][index2])
                    #新增目前這個橢圓的長軸角度（斜紋角度）。
                    newAngles.append(self._angles[index1][index2])
                    previousCombined = False
                    continue
                #兩個橢圓中心相連的直線的斜率。
                combinedSlope = (y1 - y2) / (x2 - x1)
                #使用 Median Absolute Deviation(MAD) 方法來檢測 combinedSlope 是否比該組其他橢圓的長軸斜率顯著大或小。
                groupSlopes = self._slopes[index1]
                groupSlopes.append(combinedSlope)
                medianSlopes = np.median(groupSlopes)
                diffSlopes = np.abs(groupSlopes - medianSlopes)
                scalingFactorSlopes = np.median(diffSlopes)
                lowerSlopes = medianSlopes - 2.0 * scalingFactorSlopes
                upperSlopes = medianSlopes + 2.0 * scalingFactorSlopes
                #combinedSlope 與該組其他橢圓的長軸斜率沒有顯著差異，代表兩個橢圓原本代表同一個斜紋，需將它們合併。
                if not (combinedSlope < lowerSlopes or combinedSlope > upperSlopes):
                    #合併後的新橢圓、新長軸、新斜率與新角度。
                    newEllipse, newLine, newSlope, newAngle = self._createCombinedEllipse(e, group[index2 + 1])
                    #新增合併後的橢圓。
                    newEllipses.append(newEllipse)
                    #新增合併後的橢圓長軸。
                    newLines.append(newLine)
                    #新增合併後的橢圓長軸斜率（斜紋斜率）。
                    newSlopes.append(newSlope)
                    #新增合併後的橢圓長軸角度（斜紋角度）。
                    newAngles.append(newAngle)
                    previousCombined = True
                #combinedSlope 與該組其他橢圓的長軸斜率有顯著差異，代表兩個橢圓原本代表不同斜紋，不需將它們合併。
                else:
                    #新增目前這個橢圓。
                    newEllipses.append(e)
                    #新增目前這個橢圓的長軸。
                    newLines.append(self._lines[index1][index2])
                    #新增目前這個橢圓的長軸斜率（斜紋斜率）。
                    newSlopes.append(self._slopes[index1][index2])
                    #新增目前這個橢圓的長軸角度（斜紋角度）。
                    newAngles.append(self._angles[index1][index2])
                    previousCombined = False
            newGroupedEllipses.append(newEllipses)
            newGroupedLines.append(newLines)
            newGroupedSlopes.append(newSlopes)
            newGroupedAngles.append(newAngles)
        self._lines = newGroupedLines
        self._slopes = newGroupedSlopes
        self._angles = newGroupedAngles
        return newGroupedEllipses
    
    #平移[groupedEllipses]中的所有橢圓，使同組的橢圓有相同的中心點X座標（這樣計數器在判別斜紋時會更精確）。
    def _translateEllipses(self, groupedEllipses):
        #初始情況，groupMeanCenterXs 尚未有任何值。
        if len(self._groupMeanCenterXs) == 0:
            #計算每條鋼纜上所有斜紋的平均中心點X座標。
            self._groupMeanCenterXs = [np.mean([e['ellipse'][0][0] for e in group]) for group in groupedEllipses]
        for index1, group in enumerate(groupedEllipses):
            #該組橢圓的平均中心點X座標，也就是要平移到的位置。
            groupMeanCenterX = self._groupMeanCenterXs[index1]
            for index2, e in enumerate(group):
                #該橢圓的中心點座標。
                centerX, centerY = e['ellipse'][0]
                #該橢圓的長軸斜率。
                slope = self._slopes[index1][index2]
                #長軸的直線方程式的常數項（設 y = ax + b，再將 centerX, centerY 代入 x 與 y，slope 代入 a）。
                b = centerY + slope * centerX
                #平移後的橢圓中心點 Y 座標。
                newCenterY = -slope * groupMeanCenterX + b
                #更新橢圓的中心點座標。
                e['ellipse'] = ((groupMeanCenterX, newCenterY), e['ellipse'][1], e['ellipse'][2])
                #更新橢圓的長軸。
                newStartPointX = int(round(e['majorAxe']['startPoint'][0] + (groupMeanCenterX - centerX)))
                newStartPointY = int(round(-slope * newStartPointX + b))
                newEndPointX = int(round(e['majorAxe']['endPoint'][0] + (groupMeanCenterX - centerX)))
                newEndPointY = int(round(-slope * newEndPointX + b))
                self._lines[index1][index2] = ((newStartPointX, newStartPointY), (newEndPointX, newEndPointY))
        return groupedEllipses
    '''
    透過[groupedEllipses]找出因光線、陰影等外部條件而沒有被辨識到的斜紋，並補齊斜紋。只有界線之前的斜紋才需要補，因為過了界線有沒有補都不影響計數。
    補斜紋其實就是在補橢圓。
    '''
    def _compensate(self, groupedEllipses):
        #存放新的已分組的橢圓。
        newGroupedEllipses = []
        #存放新的已分組的橢圓長軸。
        newGroupedLines = []
        #存放新的已分組的斜紋斜率。
        newGroupedSlopes = []
        #存放新的已分組的斜紋角度。
        newGroupedAngles = []
        for index1, group in enumerate(groupedEllipses):
            #使用 Median Absolute Deviation(MAD) 方法來判斷組內相鄰兩橢圓的間距是否異常大，是的話代表兩橢圓間有斜紋沒被辨識到。
            groupGaps = [np.abs(group[i + 1]['ellipse'][0][1] - e['ellipse'][0][1]) for i, e in enumerate(group[:-1])]
            medianGroupGaps = np.median(groupGaps)
            diffGroupGaps = np.abs(groupGaps - medianGroupGaps)
            scalingFactorGroupGaps = np.median(diffGroupGaps)
            upperGroupGaps = medianGroupGaps + 30.0 * scalingFactorGroupGaps
            try:
                #找出要從第幾個橢圓開始補斜紋。鋼纜為上行時，則從界線上方100像素開始往下補。鋼纜為下行時，則從界線下方100像素開始往上補。
                first = next(i for i, e in enumerate(group) if e['ellipse'][0][1] > (self._borderY - 100)) if self._upward else next(i for i, e in enumerate(group) if e['ellipse'][0][1] < (self._borderY + 100))
                newEllipses = group[:first]
                newLines = self._lines[index1][:first]
                newSlopes = self._slopes[index1][:first]
                newAngles = self._angles[index1][:first]
                for index2, e in enumerate(group[first:-1]):
                    newEllipses.append(e)
                    newLines.append(self._lines[index1][first + index2])
                    newSlopes.append(self._slopes[index1][first + index2])
                    newAngles.append(self._angles[index1][first + index2])
                    #目前這個橢圓與下一個橢圓的間距。
                    gap = np.abs(group[first + index2 + 1]['ellipse'][0][1] - e['ellipse'][0][1])
                    #如果間距過大則要補斜紋。
                    if gap > upperGroupGaps:
                        #計算需要補幾個斜紋。
                        nCompensate = int(round(gap / medianGroupGaps)) - 1
                        #從目前這個橢圓開始算，每隔多遠要補一個斜紋。
                        newGap = gap / (nCompensate + 1)
                        #補斜紋（補橢圓）。
                        for i in range(1, nCompensate + 1):
                            #補的橢圓的中心點座標。X座標為該組橢圓的平均中心點X座標，Y座標 = (目前這個橢圓的中心點Y座標) ＋ newGap * i 。
                            center = (self._groupMeanCenterXs[index1], e['ellipse'][0][1] + newGap * i) if self._upward else (self._groupMeanCenterXs[index1], e['ellipse'][0][1] - newGap * i)
                            #補的橢圓的短軸長度，為該組橢圓的平均短軸長度。
                            maLength = np.mean([min(e['ellipse'][1]) for e in group])
                            #補的橢圓的長軸長度，為該組橢圓的平均長軸長度。
                            MALength = np.mean([max(e['ellipse'][1]) for e in group])
                            #補的橢圓的斜率。
                            slope = np.mean(self._slopes[index1])
                            #補的橢圓的角度。
                            angle = np.mean(self._angles[index1])
                            newAngle = 90 - angle
                            if newAngle > 90:
                                newAngle = newAngle - 90
                            else:
                                newAngle = newAngle + 90
                            #補的橢圓的長軸起點。
                            startPoint = (int(round(center[0] + (MALength / 2) * np.cos(np.radians(newAngle)))), int(round(center[1] + (MALength / 2) * np.sin(np.radians(newAngle)))))
                            #補的橢圓的長軸終點。
                            endPoint = (int(round(center[0] + (MALength / 2) * np.cos(np.radians(newAngle + 180)))), int(round(center[1] + (MALength / 2) * np.sin(np.radians(newAngle + 180)))))
                            #補的橢圓的長軸。
                            majorAxe = {'startPoint': startPoint, 'endPoint': endPoint}
                            #補的橢圓本身。
                            ellipse = (center, (maLength, MALength), 90 - angle)
                            newEllipses.append({'ellipse': ellipse, 'majorAxe': majorAxe})
                            newLines.append((startPoint, endPoint, True))
                            newSlopes.append(slope)
                            newAngles.append(angle)
                    #補到該組倒數第二個果園則不用繼續補，直接新增該組最後一個橢圓。
                    if index2 == len(group[first:-1]) - 1:
                        newEllipses.append(group[first + index2 + 1])
                        newLines.append(self._lines[index1][first + index2 + 1])
                        newSlopes.append(self._slopes[index1][first + index2 + 1])
                        newAngles.append(self._angles[index1][first + index2 + 1])
            #沒有需要補的斜紋（界線之前沒有辨識到任何斜紋）。
            except StopIteration:
                newGroupedEllipses.append(group)
                newGroupedLines.append(self._lines[index1])
                newGroupedSlopes.append(self._slopes[index1])
                newGroupedAngles.append(self._angles[index1])
                continue
            newGroupedEllipses.append(newEllipses)
            newGroupedLines.append(newLines)
            newGroupedSlopes.append(newSlopes)
            newGroupedAngles.append(newAngles)
        self._lines = newGroupedLines
        self._slopes = newGroupedSlopes
        self._angles = newGroupedAngles
        return newGroupedEllipses
    
    #合併兩個橢圓[e1]與[e2]。
    def _createCombinedEllipse(self, e1, e2):
        center = None
        maLength = (min(e1['ellipse'][1]) + min(e2['ellipse'][1])) / 2
        MALength = None
        slope = None
        angle = None
        majorAxe = None
        MALength1 = np.sqrt((e1['majorAxe']['startPoint'][0] - e2['majorAxe']['endPoint'][0]) ** 2 + (e1['majorAxe']['startPoint'][1] - e2['majorAxe']['endPoint'][1]) ** 2)
        MALength2 = np.sqrt((e1['majorAxe']['endPoint'][0] - e2['majorAxe']['startPoint'][0]) ** 2 + (e1['majorAxe']['endPoint'][1] - e2['majorAxe']['startPoint'][1]) ** 2)
        if MALength1 > MALength2:
            center = (int(round((e1['majorAxe']['startPoint'][0] + e2['majorAxe']['endPoint'][0]) / 2)), int(round((e1['majorAxe']['startPoint'][1] + e2['majorAxe']['endPoint'][1]) / 2)))
            MALength = MALength1
            slope = (e1['majorAxe']['startPoint'][1] - e2['majorAxe']['endPoint'][1]) / (e2['majorAxe']['endPoint'][0] - e1['majorAxe']['startPoint'][0])
            angle = np.arctan(slope) * 180 / np.pi
            majorAxe = {'startPoint': e1['majorAxe']['startPoint'], 'endPoint': e2['majorAxe']['endPoint']}
        else:
            center = (int(round((e1['majorAxe']['endPoint'][0] + e2['majorAxe']['startPoint'][0]) / 2)), int(round((e1['majorAxe']['endPoint'][1] + e2['majorAxe']['startPoint'][1]) / 2)))
            MALength = MALength2
            slope = (e2['majorAxe']['startPoint'][1] - e1['majorAxe']['endPoint'][1]) / (e1['majorAxe']['endPoint'][0] - e2['majorAxe']['startPoint'][0])
            angle = np.arctan(slope) * 180 / np.pi
            majorAxe = {'startPoint': e2['majorAxe']['startPoint'], 'endPoint': e1['majorAxe']['endPoint']}
        return ({'ellipse': (center, (maLength, MALength), 90 - angle), 'majorAxe': majorAxe}, (majorAxe['startPoint'], majorAxe['endPoint']), slope, angle)
    
    #計算將圖片上的一點[point]以圖片中心為錨點旋轉[angle]角度後的新座標。
    def _rotatePointAroundImageCenter(self, point, angle):
        originalX, originalY = point
        imageCenterX = int(round(self._img.shape[1] / 2))
        imageCenterY = int(round(self._img.shape[0] / 2))
        originalX -= imageCenterX
        originalY -= imageCenterY
        angle = angle * np.pi / 180
        rotatedX = int(round(originalX * np.cos(angle) - originalY * np.sin(angle))) + imageCenterX
        rotatedY = int(round(originalX * np.sin(angle) + originalY * np.cos(angle))) + imageCenterY
        return (rotatedX, rotatedY)
        
        
        
        
        
        
        
