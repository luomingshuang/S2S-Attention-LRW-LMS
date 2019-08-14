import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
 
T = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
T_old_rl = np.array([23,24,25,26,27,28,29,30,31,32,33])
power = np.array([0.75, 0.66, 0.352, 0.378, 0.27, 0.269, 0.299, 0.218, 0.235, 0.226, 
                  0.232, 0.224, 0.269, 0.259, 0.247, 0.216, 0.207, 0.197, 0.195, 0.194, 0.188, 0.185, 0.183, 0.183])
power_old_rl = np.array([0.183, 0.186, 0.175, 0.166, 0.166, 0.164, 0.162, 0.161,0.161,0.160,0.160])
power_old_combine = np.array([0.183, 0.180, 0.175, 0.168, 0.162,0.158,0.156, 0.152,0.151,0.150,0.150])

xnew = np.linspace(T.min(),T.max(),1000) #300 represents number of points to make between T.min and T.max

xnew1 = np.linspace(T_old_rl.min(), T_old_rl.max(),1000)

power_old_rl_smooth = spline(T_old_rl, power_old_rl, xnew1)
power_old_combine_smooth = spline(T_old_rl, power_old_combine, xnew1)
print(len(T), len(power)) 

power_smooth = spline(T,power,xnew)
plt.plot(xnew, power_smooth,'-r',label='baseline',linewidth=1, color='dodgerblue')
plt.axhline(y=0.183, xmin=0, xmax=23,linewidth=1,color='dodgerblue',linestyle="-.")
plt.annotate('%.3f' %(0.183), xy=(33,0.183), xytext=(35,0.183))
plt.plot(xnew1, power_old_rl_smooth,label='MLE+CER',linewidth=1,color='orange')
plt.axhline(y=0.160, xmin=0, xmax=33,linewidth=1,color='orange',linestyle="-.")
plt.annotate('%.3f' %(0.161), xy=(33,0.155), xytext=(35,0.150))
plt.plot(xnew1, power_old_combine_smooth,label='MLE+CER_combine', linewidth=1, color='green')
plt.axhline(y=0.150, xmin=0, xmax=33,linewidth=1,color='green',linestyle="-.")
plt.annotate('%.3f' %(0.150), xy=(33,0.135), xytext=(35,0.120))
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('wer')
plt.title('WER-EPOCH')
plt.legend()
plt.show()
