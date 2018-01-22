import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#print .p file

f = open('lost_history2_100.p', 'rb')
a = pickle.load(f,encoding = 'latin1')

iter_log = np.empty((0,len(a)), int)

train_loss = np.empty((0,len(a)), int)
val_loss = np.empty((0, len(a)), int)

for i in range(0, len(a)):
      iter_log = np.append(iter_log, i)

for i in range(0, len(a)):
      train_loss = np.append(train_loss, a[i][0])
      val_loss = np.append(val_loss, a[i][1])

ax = plt.subplot(111)
plt.plot(iter_log, train_loss, label='train_loss_100', color= "#1f77b4", linewidth=3)
plt.plot(iter_log, val_loss, label='val_loss_100', color = '#e377c2', linewidth=3)
ax.legend(loc='up_center', bbox_to_anchor=(0.5,0.2), shadow=True, ncol=2)
plt.show()



