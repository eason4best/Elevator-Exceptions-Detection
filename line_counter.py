from utils import Utils

class LineCounter:
    def __init__(self, nGroup, borderY, upward = True):
        self._nGroup = nGroup
        self._borderY = borderY
        self._upward = upward
        self._counts = [0] * self._nGroup
        self._cumCounts = []
        self._compensateCounts = [0] * self._nGroup
        self._cumCompensateCounts = []
        self._examinedLines = []
    
    def count(self, groupedLines):
        for index, group in enumerate(groupedLines):
            for lineId, line in group.items():
                newId = '{}-{}'.format(index+1, lineId)
                if ((self._upward and Utils.getLineCentroid(line)[1] < self._borderY) or (not self._upward and Utils.getLineCentroid(line)[1] > self._borderY)) and newId not in self._examinedLines:
                    self._counts[index] += 1
                    self._cumCounts.append(self._counts.copy())
                    #該線條是補的。
                    if len(line) == 3:
                        self._compensateCounts[index] += 1
                    self._cumCompensateCounts.append(self._compensateCounts.copy())    
                    self._examinedLines.append(newId)
        return (self._counts, self._cumCounts, self._compensateCounts, self._cumCompensateCounts)
