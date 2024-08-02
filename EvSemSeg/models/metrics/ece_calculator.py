import numpy as np
import matplotlib.pyplot as plt

def plot_reliability_diagram_bar(Acc, bins, base_name = 'test'):
   plt.clf()

   # Calculate bin centers for x-axis ticks
   bin_edges = np.linspace(0.0, 1.0, bins + 1)
   bin_centers = []
   for i in range(bins):
      bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2.0)

   # Plotting the bar graph
   plt.bar(bin_centers, Acc, width=1.0*(bin_edges[1] - bin_edges[0]), color='blue', alpha=0.7)

   # y = x graph
   x = np.linspace(0, 1, 100)
   y = x
   plt.plot(x, y, label='y = x', color='gray')

   # Adding labels and title
   plt.xlabel('Confidence')
   plt.ylabel('Accuracy')
   plt.title('Reliability Diagram')

   # Display the plot
   plt.show()
   plt.savefig(f'{base_name}_bin{bins}.png')

def optimized_ece_with_bin(isCorrect, certainties, bin, withMCE = False, base_name = 'test'):
    # Optional
    # isCorrect = isCorrect.view(-1)
    # certainties = certainties.view(-1) 

    borders = np.linspace(0.0, 1.0, bin + 1)
    Acc, Conf, Bm = np.zeros(bin), np.zeros(bin), np.zeros(bin)
    
    for m in range(bin):
        if m == 0:
            Bm[m]   = ((certainties >= borders[m]) & (certainties < borders[m+1])).sum()
            if Bm[m] != 0:
                Acc[m]  = np.average(isCorrect[(certainties >= borders[m]) & (certainties < borders[m+1])])
                Conf[m] = np.average(certainties[(certainties >= borders[m]) & (certainties < borders[m+1])])
        else:
            Bm[m]   = ((certainties > borders[m]) & (certainties < borders[m+1])).sum()
            if Bm[m] != 0:
                Acc[m]  = np.average(isCorrect[(certainties >= borders[m]) & (certainties < borders[m+1])])
                Conf[m] = np.average(certainties[(certainties >= borders[m]) & (certainties < borders[m+1])])

    ece = 0.0
    for m in range(bin):
        ece += Bm[m] * np.abs((Acc[m] - Conf[m]))
    ECE = ece / np.sum(Bm)
    plot_reliability_diagram_bar(Acc, bin, base_name)
    
    if withMCE:
        MCE = np.max(np.abs((Acc - Conf)))
        return ECE, MCE
    return ECE