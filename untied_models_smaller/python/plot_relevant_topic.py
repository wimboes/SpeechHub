import numpy as np
import matplotlib.pyplot as plt

topic_id = 41

word_axis = np.load('word_axis.npz')
word_axis = word_axis['word_axis']

interpolation = np.load('interpolation.npz')
interpolation = interpolation['interpolation']

print(word_axis)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(word_axis +1, interpolation[:, topic_id],'y', label = 'topic ' + str(topic_id) + ': weer') # np.arange(np.shape(interpolation)[0])
plt.plot(word_axis +1, interpolation[:, 86],'g', label = 'topic ' + str(86)+ ': wielrennen')
plt.plot(word_axis +1, interpolation[:, 363],'b', label = 'topic ' + str(363) + ': voetbal')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., fontsize = 13)

    
plt.xlabel(r'Number of sentence in test data', fontsize = 14)
plt.ylabel(r'Probability assigned to topic', fontsize = 14)


plt.savefig('topic_demo.eps')
#plt.show()
