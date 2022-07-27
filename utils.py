class Utils:
    @staticmethod
    def groupColors():
        return [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 140, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255), (102, 102, 255)]
    
    @staticmethod
    def milliseconds2HMS(milliseconds):
        milliseconds = int(milliseconds)
        seconds = (milliseconds / 1000) % 60
        seconds = int(seconds)
        seconds = str(seconds) if seconds > 9 else '0{}'.format(seconds)
        minutes = (milliseconds / (1000 * 60)) % 60
        minutes = int(minutes)
        minutes = str(minutes) if minutes > 9 else '0{}'.format(minutes)
        hours = (milliseconds / (1000 * 60 * 60)) % 24
        hours = int(hours)
        hours = str(hours) if hours > 9 else '0{}'.format(hours)
        
        return (hours, minutes, seconds)
    
    @staticmethod
    def getLineCentroid(line):
        return (int(round((line[0][0] + line[1][0]) / 2)), int(round((line[0][1] + line[1][1]) / 2)))