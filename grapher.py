import matplotlib.pyplot as plt
from utils import Utils

class Grapher:
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['Noto Sans TC']
    
    def countsTable(self, counts, compensateCounts, cumCounts, cumCompensateCounts):
        _, axs = plt.subplots(2, sharex = True, figsize=(40, 30))
        frames = list(range(1, len(cumCounts) + 1))
        #斜紋累積總數的折線圖。
        cumCounts = [list(c) for c in zip(*cumCounts)]
        for i, c in enumerate(cumCounts):
            color = tuple(reversed(Utils.groupColors()[i]))
            color = tuple(c / 255 for c in color)
            axs[0].plot(frames, c, label = '鋼纜{}'.format(i + 1), color = color)
        axs[0].legend(loc = 4, prop={'size': 20})
        axs[0].set_title('斜紋總數', fontsize = 40)
        axs[0].set_ylabel('個數', fontsize = 36, rotation = 0, labelpad = 40)
        axs[0].yaxis.set_major_locator(plt.MaxNLocator(integer = True))
        #斜紋累積補償數的折線圖。
        cumCompensateCounts = [list(c) for c in zip(*cumCompensateCounts)]
        for i, c in enumerate(cumCompensateCounts):
            color = tuple(reversed(Utils.groupColors()[i]))
            color = tuple(c / 255 for c in color)
            axs[1].plot(frames, c, label = '鋼纜{}'.format(i + 1), color = color)
        axs[1].legend(loc = 4, prop={'size': 20})
        axs[1].set_title('斜紋補償數', fontsize = 40)
        axs[1].set_xlabel('時間', fontsize = 36)
        axs[1].set_ylabel('個數', fontsize = 36, rotation = 0, labelpad = 40)
        axs[1].set_xticklabels([])
        axs[1].yaxis.set_major_locator(plt.MaxNLocator(integer = True))
        #斜紋總數與補償數的表格。
        table = axs[1].table([counts, compensateCounts], 
                  rowLabels = ['總數', '補償數'], 
                  colLabels = ['鋼纜{}'.format(i + 1) for i, _ in enumerate(counts)], 
                  bbox = [0.0, -0.8, 1.0, 0.5],
                  )
        axs[0].tick_params(axis = 'x', labelsize = 24)
        axs[0].tick_params(axis = 'y', labelsize = 24)
        axs[1].tick_params(axis = 'x', labelsize = 24)
        axs[1].tick_params(axis = 'y', labelsize = 24)
        table.set_fontsize(36)
        plt.subplots_adjust(hspace = 0.2)
        
    def cumCountsGraph(self):
        print('fuck')
        
    def cumCompensateCountsGraph(self):
        print('fuck')

