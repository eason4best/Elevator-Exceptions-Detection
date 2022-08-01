from utils import Utils

class LineCounter:
    def __init__(self, nGroup, borderY, upward = True):
        #鋼纜數量。
        self._nGroup = nGroup
        #界線的位置。
        self._borderY = borderY
        #鋼纜移動方向。
        self._upward = upward
        #每組的斜紋數量。
        self._counts = [0] * self._nGroup
        #每組的累積斜紋數量。
        self._cumCounts = []
        #每組的斜紋補償數量。
        self._compensateCounts = [0] * self._nGroup
        #每組的累積斜紋補償數量。
        self._cumCompensateCounts = []
        #存放已辨識過的斜紋的Id。
        self._examinedLines = []
    
    def count(self, groupedLines):
        for index, group in enumerate(groupedLines):
            for lineId, line in group.items():
                #用來在所有斜紋間區別的Id。
                newId = '{}-{}'.format(index+1, lineId)
                #如果該斜紋的中心超過界線且該斜紋不在examinedLines中，則需更新計數。
                if ((self._upward and Utils.getLineCentroid(line)[1] < self._borderY) or (not self._upward and Utils.getLineCentroid(line)[1] > self._borderY)) and newId not in self._examinedLines:
                    #該組斜紋數量加一。
                    self._counts[index] += 1
                    #新增目前的累積斜紋數量。
                    self._cumCounts.append(self._counts.copy())
                    #該斜紋是補的。
                    if len(line) == 3:
                        #該組斜紋補償數量加一。
                        self._compensateCounts[index] += 1
                    #新增目前的累積補償斜紋數量。
                    self._cumCompensateCounts.append(self._compensateCounts.copy())
                    #將該斜紋的Id加入examinedLines，避免重複辨識。
                    self._examinedLines.append(newId)
        return (self._counts, self._cumCounts, self._compensateCounts, self._cumCompensateCounts)
