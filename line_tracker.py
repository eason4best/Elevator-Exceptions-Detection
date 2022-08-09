from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np
from utils import Utils

class LineTracker:
    def __init__(self, nGroup, borderY = None, upward = True, maxDisappear = 5):
        #鋼纜數量。
        self._nGroup = nGroup
        #界線的位置。
        self._borderY = borderY
        #鋼纜移動方向。
        self._upward = upward
        #斜紋最多能消失幾幀還能繼續被追蹤。
        self._maxDisappear = maxDisappear
        #每組提供給新被追蹤的斜紋的Id。
        self._nextLineIds = [0] * self._nGroup
        #每組被追蹤中的斜紋。
        self._groupedLines = [OrderedDict() for i in range(0, self._nGroup)]   
        #每組被追蹤中的斜紋已消失了幾幀。
        self._groupedDisappears = [OrderedDict() for i in range(0, self._nGroup)] 
    
    #將斜紋[line]註冊到第[index]組，開始追蹤它。
    def _register(self, index, line):
        self._groupedLines[index][self._nextLineIds[index]] = line
        self._groupedDisappears[index][self._nextLineIds[index]] = 0
        self._nextLineIds[index] += 1
    
    #將第[index]組中Id為[lineId]的斜紋取消註冊，不再追蹤它。
    def _deregister(self, index, lineId):
        del self._groupedLines[index][lineId]
        del self._groupedDisappears[index][lineId]
        
    #透過斜紋的中心點在幀與幀間的變化來追蹤斜紋。
    def track(self, groupedLines):
        #沒有辨識到任何斜紋的情況，將所有現有斜紋的disappear加一，加完後若超過maxDisappear則取消註冊該斜紋。
        if sum([len(group) for group in groupedLines]) == 0:
            for index, lineIds in enumerate([list(d.keys()) for d in self._groupedDisappears]):
                for lineId in lineIds:
                    self._groupedDisappears[index][lineId] += 1
                    if self._groupedDisappears[index][lineId] > self._maxDisappear:
                        self._deregister(index ,lineId)
            return self._groupedLines
        #初始情況，替第一幀中所有斜紋註冊。
        if sum([len(group) for group in self._groupedLines]) == 0:
            for index, lines in enumerate(groupedLines):
                if self._borderY is None:
                    for i in range(0, len(lines)):
                        self._register(index, lines[i])
                else:
                    for i in range(0, len(lines)):
                        if (self._upward and Utils.getLineCentroid(lines[i])[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(lines[i])[1] < self._borderY):
                            self._register(index, lines[i])
        #若有組在第一幀中沒有任何斜紋被註冊，則在此先註冊。
        for index, lines in enumerate(self._groupedLines):
            if len(lines) == 0:
                if self._borderY is None:
                    for l in groupedLines[index]:
                        self._register(index, l)
                else:
                    for l in groupedLines[index]:
                        if (self._upward and Utils.getLineCentroid(l)[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(l)[1] < self._borderY):
                            self._register(index, l)
        #透過目前已被追蹤的舊斜紋與這一幀辨識到的新斜紋，其中心點之間的距離來判斷新斜紋是來自哪一條舊斜紋。
        for index, lines in enumerate(self._groupedLines):
            #新斜紋的Id。
            lineIds = list(lines.keys())
            #新斜紋的中心點。
            centroids = [Utils.getLineCentroid(l) for l in list(lines.values())]
            #如沒有辨識到任何屬於這組的新斜紋，則不處理這組。
            if len(centroids) == 0:
                continue
            #計算舊斜紋與新斜紋中心點之間的距離矩陣。
            D = dist.cdist(np.array(centroids), [Utils.getLineCentroid(l) for l in groupedLines[index]])
            #將距離矩陣從左至右，由距離小到大排序。
            rows = D.min(axis = 1).argsort()
            cols = D.argmin(axis = 1)[rows]
            usedRows = set()
            usedCols = set()
            #對於每個舊斜紋，與其距離最近的新斜紋即為同一個斜紋。
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
            #尚未處理到的舊斜紋（沒有新斜紋與其對應）。
            for row in unusedRows:
                lineId = lineIds[row]
                #將該斜紋已消失的幀數加一。
                self._groupedDisappears[index][lineId] += 1
                #如果該斜紋已消失的幀數超過maxDisappear，則取消追蹤它。
                if self._groupedDisappears[index][lineId] > self._maxDisappear:
                    self._deregister(index, lineId)
            #尚未處理到的新斜紋（沒有舊斜紋與其對應）。
            if self._borderY is None:
                for col in unusedCols:
                    self._register(index, groupedLines[index][col])
            else:
                for col in unusedCols:
                    if (self._upward and Utils.getLineCentroid(groupedLines[index][col])[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(groupedLines[index][col])[1] < self._borderY): 
                        self._register(index, groupedLines[index][col])
        return self._groupedLines