from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np
from utils import Utils

class LineTracker:
    def __init__(self, nGroup, borderY, upward = True, maxDisappear = 5):
        self._nGroup = nGroup
        self._borderY = borderY
        self._upward = upward
        self._maxDisappear = maxDisappear
        self._nextLineIds = [0] * self._nGroup
        self._groupedLines = [OrderedDict() for i in range(0, self._nGroup)]        
        self._groupedDisappears = [OrderedDict() for i in range(0, self._nGroup)] 
    
    def _register(self, index, line):
        self._groupedLines[index][self._nextLineIds[index]] = line
        self._groupedDisappears[index][self._nextLineIds[index]] = 0
        self._nextLineIds[index] += 1
    
    def _deregister(self, index, lineId):
        del self._groupedLines[index][lineId]
        del self._groupedDisappears[index][lineId]
        
    def track(self, groupedLines):
        #沒有辨識到任何斜紋的情況，將所有現有斜紋的disappear加一，加完後若超過maxDisappear則解除註冊該斜紋。
        if sum([len(group) for group in groupedLines]) == 0:
            for index, lineIds in enumerate([list(d.keys()) for d in self._groupedDisappears]):
                for lineId in lineIds:
                    self._groupedDisappears[index][lineId] += 1
                    if self._groupedDisappears[index][lineId] > self._maxDisappear:
                        self._deregister(index ,lineId)
            return self._groupedLines
        #初始情況，替Frame中所有斜紋註冊。
        if sum([len(lines) for lines in self._groupedLines]) == 0:
            for index, lines in enumerate(groupedLines):
                for i in range(0, len(lines)):
                    if (self._upward and Utils.getLineCentroid(lines[i])[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(lines[i])[1] < self._borderY):
                        self._register(index, lines[i])
        for index, lines in enumerate(self._groupedLines):
            if len(lines) == 0:
                for l in groupedLines[index]:
                    if (self._upward and Utils.getLineCentroid(l)[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(l)[1] < self._borderY):
                        self._register(index, l)
        for index, lines in enumerate(self._groupedLines):
            lineIds = list(lines.keys())
            centroids = [Utils.getLineCentroid(l) for l in list(lines.values())]
            if len(centroids) == 0:
                continue
            D = dist.cdist(np.array(centroids), [Utils.getLineCentroid(l) for l in groupedLines[index]])
            rows = D.min(axis = 1).argsort()
            cols = D.argmin(axis = 1)[rows]
            usedRows = set()
            usedCols = set()
            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                lineId = lineIds[row]
                self._groupedLines[index][lineId] = groupedLines[index][col]
                self._groupedDisappears[index][lineId] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for row in unusedRows:
                lineId = lineIds[row]
                self._groupedDisappears[index][lineId] += 1
                if self._groupedDisappears[index][lineId] > self._maxDisappear:
                    self._deregister(index, lineId)
            for col in unusedCols:
                if (self._upward and Utils.getLineCentroid(groupedLines[index][col])[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(groupedLines[index][col])[1] < self._borderY): 
                    self._register(index, groupedLines[index][col])
        return self._groupedLines