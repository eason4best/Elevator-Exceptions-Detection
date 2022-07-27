from utils import Utils

class LineCounter:
    def __init__(self, nGroup, borderY, upward = True):
        self._nGroup = nGroup
        self._borderY = borderY
        self._upward = upward
        self._counts = [0] * self._nGroup
        self._examinedLines = []
    
    def count(self, groupedLines):
        for index, group in enumerate(groupedLines):
            for lineId, line in group.items():
                newId = '{}-{}'.format(index+1, lineId)
                if ((self._upward and Utils.getLineCentroid(line)[1] < self._borderY) or (not self._upward and Utils.getLineCentroid(line)[1] > self._borderY)) and newId not in self._examinedLines:
                    self._counts[index] += 1
                    self._examinedLines.append(newId)
        return self._counts
