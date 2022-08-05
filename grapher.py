import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines
from utils import Utils

class Grapher:
    def __init__(self):
        #設定字體，讓圖表能正常顯示中文。
        plt.rcParams['font.sans-serif'] = ['Noto Sans TC']
    
    #畫出斜紋累積總數[cumCounts]、斜紋累積補償數[cumCompensateCounts]的折線圖與斜紋總數[counts]、斜紋補償數[compensateCounts]的表格。
    def plotCountsGraphAndTable(self, counts, compensateCounts, cumCounts, cumCompensateCounts):
        _, axs = plt.subplots(2, sharex = True, figsize=(40, 30))
        #幀數，為X軸的單位。
        frames = list(range(1, len(cumCounts) + 1))
        #斜紋累積總數的折線圖。
        cumCounts = [list(c) for c in zip(*cumCounts)]
        #依序用不同顏色畫出每條鋼纜累積總數的折線。
        for i, c in enumerate(cumCounts):
            color = tuple(reversed(Utils.groupColors()[i]))
            color = tuple(c / 255 for c in color)
            axs[0].plot(frames, c, label = i + 1, color = color)
        #在折線上標示其代表第幾條鋼纜。
        labelLines(axs[0].get_lines(), align = False, fontSize = 36)
        #標示折線圖標題。
        axs[0].set_title('斜紋總數', fontsize = 40)
        #標示折線圖Y軸名稱。
        axs[0].set_ylabel('個數', fontsize = 36, rotation = 0, labelpad = 40)
        #設定折線圖Y軸僅顯示整數標線。
        axs[0].yaxis.set_major_locator(plt.MaxNLocator(integer = True))
        #斜紋累積補償數的折線圖。
        cumCompensateCounts = [list(c) for c in zip(*cumCompensateCounts)]
        #依序用不同顏色畫出每條鋼纜累積補償數的折線。
        for i, c in enumerate(cumCompensateCounts):
            color = tuple(reversed(Utils.groupColors()[i]))
            color = tuple(c / 255 for c in color)
            axs[1].plot(frames, c, label = i + 1, color = color)
        #在折線上標示其代表第幾條鋼纜。
        labelLines(axs[1].get_lines(), align = False, fontSize = 36)
        #標示折線圖標題。
        axs[1].set_title('斜紋補償數', fontsize = 40)
        #標示折線圖X軸名稱。
        axs[1].set_xlabel('幀數', fontsize = 36)
        #標示折線圖Y軸名稱。
        axs[1].set_ylabel('個數', fontsize = 36, rotation = 0, labelpad = 40)
        axs[1].set_xticklabels([])
        #設定折線圖Y軸僅顯示整數標線。
        axs[1].yaxis.set_major_locator(plt.MaxNLocator(integer = True))
        #斜紋總數與補償數的表格。
        table = axs[1].table([counts, compensateCounts], 
                  rowLabels = ['總數', '補償數'], 
                  colLabels = ['鋼纜{}'.format(i + 1) for i, _ in enumerate(counts)], 
                  bbox = [0.0, -0.8, 1.0, 0.5],
                  )
        #設定折線圖字體大小。
        axs[0].tick_params(axis = 'x', labelsize = 24)
        axs[0].tick_params(axis = 'y', labelsize = 24)
        axs[1].tick_params(axis = 'x', labelsize = 24)
        axs[1].tick_params(axis = 'y', labelsize = 24)
        #設定表格字體大小。
        table.set_fontsize(36)
        #設定圖表間的垂直間距。
        plt.subplots_adjust(hspace = 0.2)
        
    #畫出斜紋斜率隨時間變化的圖。
    def plotSlopesGraph(self, cumLines, lineRanges):
        _, axs = plt.subplots(2, sharex = True, figsize=(40, 30))
        #幀數，為X軸的單位。
        frames = list(range(1, len(cumLines) + 1))
        cumLines = [list(l) for l in zip(*cumLines)]
        #範圍一（滑輪上）平均斜率隨時間變化的趨勢線圖。
        cumLines1 = cumLines.copy()
        cumLines1 = [[[line for line in frameLines if (Utils.getLineCentroid(line)[1] > lineRanges[0][0] and Utils.getLineCentroid(line)[1] < lineRanges[0][1])] for frameLines in group] for group in cumLines1]
        #每組每一幀的平均斜紋斜率。
        cumSlopes1 = []
        for group in cumLines1:
            cumGroupSlope = []
            for frameLines in group:
                averageSlope = np.mean([(line[0][1] - line[1][1]) / (line[1][0] - line[0][0]) for line in frameLines if (line[1][0] - line[0][0]) != 0])
                cumGroupSlope.append(averageSlope)
            cumSlopes1.append(cumGroupSlope)
        #依序用不同顏色畫出每條鋼纜平均斜紋斜率隨時間變化的趨勢線。
        for i, s in enumerate(cumSlopes1):
            color = tuple(reversed(Utils.groupColors()[i]))
            color = tuple(c / 255 for c in color)
            index = np.isfinite(frames) & np.isfinite(s)
            z = np.polyfit(np.array(frames)[index], np.array(s)[index], 20)
            p = np.poly1d(z)
            axs[0].plot(frames, p(frames), label = i + 1, color = color)
        #在趨勢線上標示其代表第幾條鋼纜。
        labelLines(axs[0].get_lines(), align = False, fontSize = 36)
        #標示趨勢線圖標題。
        axs[0].set_title('滑輪上斜紋平均斜率', fontsize = 40)
        #標示趨勢線圖Y軸名稱。
        axs[0].set_ylabel('斜率', fontsize = 36, rotation = 0, labelpad = 40)
        #範圍二（滑輪下）平均斜率隨時間變化的趨勢線圖。
        cumLines2 = cumLines.copy()
        cumLines2 = [[[line for line in frameLines if (Utils.getLineCentroid(line)[1] > lineRanges[1][0] and Utils.getLineCentroid(line)[1] < lineRanges[1][1])] for frameLines in group] for group in cumLines2]
        #每組每一幀的平均斜紋斜率。
        cumSlopes2 = []
        for group in cumLines2:
            cumGroupSlope = []
            for frameLines in group:
                averageSlope = np.mean([(line[0][1] - line[1][1]) / (line[1][0] - line[0][0]) for line in frameLines if (line[1][0] - line[0][0]) != 0])
                cumGroupSlope.append(averageSlope)
            cumSlopes2.append(cumGroupSlope)
        #依序用不同顏色畫出每條鋼纜平均斜紋斜率隨時間變化的趨勢線。
        for i, s in enumerate(cumSlopes2):
            color = tuple(reversed(Utils.groupColors()[i]))
            color = tuple(c / 255 for c in color)
            index = np.isfinite(frames) & np.isfinite(s)
            z = np.polyfit(np.array(frames)[index], np.array(s)[index], 20)
            p = np.poly1d(z)
            axs[1].plot(frames, p(frames), label = i + 1, color = color)
        #在趨勢線上標示其代表第幾條鋼纜。
        labelLines(axs[1].get_lines(), align = False, fontSize = 36)
        #標示趨勢線圖標題。
        axs[1].set_title('滑輪下斜紋平均斜率', fontsize = 40)
        #標示趨勢線圖X軸名稱。
        axs[1].set_xlabel('幀數', fontsize = 36)
        #標示趨勢線圖Y軸名稱。
        axs[1].set_ylabel('斜率', fontsize = 36, rotation = 0, labelpad = 40)
        axs[1].set_xticklabels([])
        #設定趨勢線圖字體大小。
        axs[0].tick_params(axis = 'x', labelsize = 24)
        axs[0].tick_params(axis = 'y', labelsize = 24)
        axs[1].tick_params(axis = 'x', labelsize = 24)
        axs[1].tick_params(axis = 'y', labelsize = 24)
        #設定圖表間的垂直間距。
        plt.subplots_adjust(hspace = 0.2)
        
        
        
        
        
