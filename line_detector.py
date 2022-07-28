import cv2
import numpy as np

class LineDetector:
    def __init__(self, borderY, upward = True):
        self._borderY = borderY
        self._upward = upward
        self._groupMeanCenterXs = []
    
    def detect(self, img):
        self._img = img
        #將原始圖片進行預處理。
        preprocessImg = self._preprocess()
        #使用Canny邊緣檢測找出斜紋的邊緣。
        canny = cv2.Canny(preprocessImg, 100, 100, apertureSize = 3)
        #找出圖片中的輪廓並對每一個輪廓都適配（Fit）一個橢圓。 
        fittedEllipses = self._findContourAndFitEllipse(canny)
        #移除屬於離群值的橢圓（面積異常小或大、角度異常小或大）。
        fittedEllipses = self._removeOutliersEllipses(fittedEllipses, lowerFactor = 2.0, upperFactor = 3.0)
        #使用橢圓的長軸來表示鋼纜上的斜紋。
        fittedEllipses = self._findEllipseMajorAxes(fittedEllipses)
        #將橢圓照著長軸起始點的X座標由小到大（圖片中由左到右）排序。
        fittedEllipses = sorted(fittedEllipses, key = lambda fe: fe['majorAxe']['startPoint'][0])
        #將同一條鋼纜上的橢圓分組。
        groupedFittedEllipses = self._groupEllipses(fittedEllipses)
        if len(groupedFittedEllipses) != len(self._groupMeanCenterXs) and len(self._groupMeanCenterXs) > 0:
            return None
        #獲得代表斜紋的直線（橢圓長軸），與該線的斜率與角度。
        self._lines, self._slopes, self._angles = self._computeSlopeAndAngle(groupedFittedEllipses)
        if self._upward:
            #將每組橢圓照著長軸起始點旋轉後的Y座標由小到大（圖片中由上到下）排序。
            for i in range(0, len(groupedFittedEllipses)):
                sortResult = sorted(zip(groupedFittedEllipses[i], self._lines[i], self._slopes[i], self._angles[i]), key = lambda x: self._rotatePointAroundImageCenter(x[0]['majorAxe']['startPoint'], np.mean(self._angles[i]))[1])
                groupedFittedEllipses[i] = [x[0] for x in sortResult]
                self._lines[i] = [x[1] for x in sortResult]
                self._slopes[i] = [x[2] for x in sortResult]
                self._angles[i] = [x[3] for x in sortResult]
        else:
            #將每組橢圓照著長軸起始點旋轉後的Y座標由大到小（圖片中由下到上）排序。
            for i in range(0, len(groupedFittedEllipses)):
                sortResult = sorted(zip(groupedFittedEllipses[i], self._lines[i], self._slopes[i], self._angles[i]), key = lambda x: self._rotatePointAroundImageCenter(x[0]['majorAxe']['startPoint'], np.mean(self._angles[i]))[1], reverse = True)
                groupedFittedEllipses[i] = [x[0] for x in sortResult]
                self._lines[i] = [x[1] for x in sortResult]
                self._slopes[i] = [x[2] for x in sortResult]
                self._angles[i] = [x[3] for x in sortResult]
        #將同一條斜紋上的破碎橢圓接在一起。
        groupedFittedEllipses = self._combineSmallEllipses(groupedFittedEllipses)
        #平移所有橢圓，使它們的中心點有相同的X座標。
        groupedFittedEllipses = self._translateEllipses(groupedFittedEllipses)
        #補齊沒有被辨識到的橢圓。
        groupedFittedEllipses = self._compensate(groupedFittedEllipses)
        return (groupedFittedEllipses, self._lines, self._slopes, self._angles)
    
    def _preprocess(self):
        #將原始圖片轉為灰階，以進行後續處理。
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        #使用高斯模糊消除圖片中的雜訊，避免其干擾後續辨識。
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #將圖片進行二值化，突顯出要辨識的斜紋。
        _, threshold = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
        #使用 Morphological Transformations 清楚分開每條斜紋。 
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        return morph
    
    def _findContourAndFitEllipse(self, img):
        #找出圖片中的輪廓。 
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        fittedEllipse = []
        for c in contours:
            if len(c) > 5:
                ellipse = cv2.fitEllipse(c)
                a, b = ellipse[1]
                minorLength = min(a,b)
                majorLength = max(a,b)
                area = (minorLength / 2) * (majorLength / 2) * np.pi
                fittedEllipse.append({'ellipse':ellipse, 'area':area})
        return fittedEllipse
    
    def _removeOutliersEllipses(slef, ellipses, minArea = 80, lowerFactor = 1.0, upperFactor = 1.0):
        #使用 Median Absolute Deviation(MAD) 檢測面積與角度的離群值，移除面積異常小或大、角度異常小或大的橢圓。
        #先將面積小於 minArea 的橢圓直接移除，避免其影響中值（Median）的大小，使得正常面積的橢圓反而被當成離群值。
        ellipses = list(filter(lambda e: e['area'] > minArea, ellipses))
        areas = [e['area'] for e in ellipses]
        angles = [e['ellipse'][2] for e in ellipses]
        medianAreas = np.median(areas)
        medianAngles = np.median(angles)
        diffAreas = np.abs(areas - medianAreas)
        diffAngles = np.abs(angles - medianAngles)
        scalingFactorAreas = np.median(diffAreas)
        scalingFactorAngles = np.median(diffAngles)
        lowerAreas = medianAreas - lowerFactor*scalingFactorAreas
        upperAreas = medianAreas + upperFactor*scalingFactorAreas
        lowerAngles = medianAngles - lowerFactor*scalingFactorAngles
        upperAngles = medianAngles + upperFactor*scalingFactorAngles
        ellipses = list(filter(lambda e: (e['area'] < upperAreas and e['area'] > lowerAreas and e['ellipse'][2] < upperAngles and e['ellipse'][2] > lowerAngles), ellipses))
        #Area之後用不到，先移除。
        ellipses = [{k: v for k, v in e.items() if k != 'area'} for e in ellipses]
        return ellipses
    
    def _findEllipseMajorAxes(self, ellipses):
        for e in ellipses:
            (x0, y0), (a, b), angle = e['ellipse']
            majorLength = max(a,b) / 2
            if angle > 90:
                angle = angle - 90
            else:
                angle = angle + 90
            startPoint = (int(round(x0 + majorLength * np.cos(np.radians(angle)))), int(round(y0 + majorLength * np.sin(np.radians(angle)))))
            endPoint = (int(round(x0 + majorLength * np.cos(np.radians(angle + 180)))), int(round(y0 + majorLength * np.sin(np.radians(angle + 180)))))
            e['majorAxe'] = {'startPoint': startPoint, 'endPoint': endPoint}
        return ellipses
    
    def _groupEllipses(self, ellipses):
        groupedEllipses = [[ellipses[0]]]
        for index, ma in enumerate([e['majorAxe'] for e in ellipses[1:]]):
            currentStartPointX = ma['startPoint'][0]
            currentEndPointX = ma['endPoint'][0]
            meanGroupEndPointX = np.mean([e['majorAxe']['endPoint'][0] for e in groupedEllipses[-1]])
            inNextGroup = True
            for previousMA in [e['majorAxe'] for e in groupedEllipses[-1]]:
                previousStartPointX = previousMA['startPoint'][0]
                previousEndPointX = previousMA['endPoint'][0]
                #如果該橢圓與該組任一個現有橢圓有交錯，且其StartPoint的X座標小於該組平均EndPointX座標，則將它分到該組。
                if ((currentStartPointX < previousStartPointX and currentEndPointX > previousStartPointX) or (currentEndPointX > previousStartPointX and currentStartPointX < previousEndPointX) or (currentStartPointX > previousStartPointX and currentEndPointX < previousEndPointX) or (currentStartPointX < previousStartPointX and currentEndPointX > previousEndPointX)) and (currentStartPointX < meanGroupEndPointX):
                    groupedEllipses[-1].append(ellipses[index+1])
                    inNextGroup = False
                    break
            if inNextGroup:
                groupedEllipses.append([ellipses[index+1]])
        #將橢圓數量極少的組移除（少數破碎的橢圓會被演算法單獨分成一組）。
        groupedEllipses = list(filter(lambda g: len(g) > 5 , groupedEllipses))
        return groupedEllipses
    
    def _computeSlopeAndAngle(self, groupedEllipses):
        #分組儲存代表每條斜紋的橢圓長軸。
        lines = [[0] * len(group) for group in groupedEllipses]
        #分組儲存每條斜紋（橢圓長軸）的斜率。
        slopes = [[0] * len(group) for group in groupedEllipses]
        #分組儲存每條斜紋（橢圓長軸）的角度。
        angles = [[0] * len(group) for group in groupedEllipses]
        for index1, group in enumerate(groupedEllipses):
            for index2, e in enumerate(group):
                startPoint = e['majorAxe']['startPoint']
                endPoint = e['majorAxe']['endPoint']
                slope = (startPoint[1] - endPoint[1]) / (endPoint[0] - startPoint[0])
                angle = np.arctan(slope) * 180 / np.pi
                lines[index1][index2] = (startPoint, endPoint)
                slopes[index1][index2] = slope
                angles[index1][index2] = angle
        return (lines, slopes, angles)
    
    def _combineSmallEllipses(self, groupedEllipses):
        newGroupedEllipses = []
        newGroupedLines = []
        newGroupedSlopes = []
        newGroupedAngles = []
        for index1, group in enumerate(groupedEllipses):
            newEllipses = []
            newLines = []
            newSlopes = []
            newAngles = []
            previousCombined = False
            for index2, e in enumerate(group[:-1]):
                if previousCombined:
                    previousCombined = False
                    continue
                (x1, y1), (a1, b1), _ = e['ellipse']
                (x2, y2), (a2, b2), _ = group[index2 + 1]['ellipse']
                if x2 - x1 == 0:
                    newEllipses.append(e)
                    newLines.append(self._lines[index1][index2])
                    newSlopes.append(self._slopes[index1][index2])
                    newAngles.append(self._angles[index1][index2])
                    previousCombined = False
                    continue
                combinedSlope = (y1 - y2) / (x2 - x1)
                groupSlopes = self._slopes[index1]
                groupSlopes.append(combinedSlope)
                medianSlopes = np.median(groupSlopes)
                diffSlopes = np.abs(groupSlopes - medianSlopes)
                scalingFactorSlopes = np.median(diffSlopes)
                lowerSlopes = medianSlopes - 2.0 * scalingFactorSlopes
                upperSlopes = medianSlopes + 2.0 * scalingFactorSlopes
                if not (combinedSlope < lowerSlopes or combinedSlope > upperSlopes):
                    #合併兩個橢圓。
                    newEllipse, newLine, newSlope, newAngle = self._createCombinedEllipse(e, group[index2 + 1])
                    newEllipses.append(newEllipse)
                    newLines.append(newLine)
                    newSlopes.append(newSlope)
                    newAngles.append(newAngle)
                    previousCombined = True
                else:
                    #不合併。
                    newEllipses.append(e)
                    newLines.append(self._lines[index1][index2])
                    newSlopes.append(self._slopes[index1][index2])
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
    
    def _translateEllipses(self, groupedEllipses):
        if len(self._groupMeanCenterXs) == 0:
            self._groupMeanCenterXs = [np.mean([e['ellipse'][0][0] for e in group]) for group in groupedEllipses]
        for index1, group in enumerate(groupedEllipses):
            groupMeanCenterX = self._groupMeanCenterXs[index1]
            for index2, e in enumerate(group):
                centerX, centerY = e['ellipse'][0]
                slope = self._slopes[index1][index2]
                c = centerY + slope * centerX
                newCenterY = -slope * groupMeanCenterX + c
                e['ellipse'] = ((groupMeanCenterX, newCenterY), e['ellipse'][1], e['ellipse'][2])
                newStartPointX = int(round(e['majorAxe']['startPoint'][0] + (groupMeanCenterX - centerX)))
                newStartPointY = int(round(-slope * newStartPointX + c))
                newEndPointX = int(round(e['majorAxe']['endPoint'][0] + (groupMeanCenterX - centerX)))
                newEndPointY = int(round(-slope * newEndPointX + c))
                self._lines[index1][index2] = ((newStartPointX, newStartPointY), (newEndPointX, newEndPointY))
        return groupedEllipses
    
    def _compensate(self, groupedEllipses):
        newGroupedEllipses = []
        newGroupedLines = []
        newGroupedSlopes = []
        newGroupedAngles = []
        for index1, group in enumerate(groupedEllipses):
            groupGaps = [np.abs(group[i + 1]['ellipse'][0][1] - e['ellipse'][0][1]) for i, e in enumerate(group[:-1])]
            medianGroupGaps = np.median(groupGaps)
            diffGroupGaps = np.abs(groupGaps - medianGroupGaps)
            scalingFactorGroupGaps = np.median(diffGroupGaps)
            upperGroupGaps = medianGroupGaps + 30.0 * scalingFactorGroupGaps
            try:
                #找出開始補斜紋的位置（界線之前才需要補）。
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
                    gap = np.abs(group[first + index2 + 1]['ellipse'][0][1] - e['ellipse'][0][1])
                    if gap > upperGroupGaps:
                        nCompensate = int(round(gap / medianGroupGaps)) - 1
                        #補線
                        newGap = gap / (nCompensate + 1)
                        for i in range(1, nCompensate + 1):
                            center = (self._groupMeanCenterXs[index1], e['ellipse'][0][1] + newGap * i) if self._upward else (self._groupMeanCenterXs[index1], e['ellipse'][0][1] - newGap * i)
                            maLength = np.mean([min(e['ellipse'][1]) for e in group])
                            MALength = np.mean([max(e['ellipse'][1]) for e in group])
                            slope = np.mean(self._slopes[index1])
                            angle = np.mean(self._angles[index1])
                            newAngle = 90 - angle
                            if newAngle > 90:
                                newAngle = newAngle - 90
                            else:
                                newAngle = newAngle + 90
                            startPoint = (int(round(center[0] + (MALength / 2) * np.cos(np.radians(newAngle)))), int(round(center[1] + (MALength / 2) * np.sin(np.radians(newAngle)))))
                            endPoint = (int(round(center[0] + (MALength / 2) * np.cos(np.radians(newAngle + 180)))), int(round(center[1] + (MALength / 2) * np.sin(np.radians(newAngle + 180)))))
                            majorAxe = {'startPoint': startPoint, 'endPoint': endPoint}
                            ellipse = (center, (maLength, MALength), 90 - angle)
                            newEllipses.append({'ellipse': ellipse, 'majorAxe': majorAxe})
                            newLines.append((startPoint, endPoint, True))
                            newSlopes.append(slope)
                            newAngles.append(angle)
                    if index2 == len(group[first:-1]) - 1:
                        newEllipses.append(group[first + index2 + 1])
                        newLines.append(self._lines[index1][first + index2 + 1])
                        newSlopes.append(self._slopes[index1][first + index2 + 1])
                        newAngles.append(self._angles[index1][first + index2 + 1])
            except StopIteration:
                #沒有需要補的斜紋（界線之前沒有辨識到任何斜紋）。
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
        
        
        
        
        
        
        
