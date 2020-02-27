import matplotlib.pyplot as plt
import numpy as np

f = open('all_distrib_40f.txt', "r")
values = f.read().split('\n')
n = len(values)

means = [float(x.split('\t')[0]) for x in values]
stds = [float(x.split('\t')[1]) for x in values]

plt.hist(means, color = 'blue', edgecolor = 'black', bins = 50)
plt.title('Luminanica de todos os vídeos')
plt.xlabel('Intensidade média dos pixels')
plt.ylabel('Número de vídeos')
plt.xticks(np.arange(0, 255, 25))

plt.show()