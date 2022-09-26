import numpy as np
import matplotlib.pyplot as plt

b = np.arange(0, 70*np.pi/180, 0.001)
bg = np.arange(0, 70*np.pi/180, 0.001)
bv, bgv = np.meshgrid(b[722:808], bg[722:808])

power = 1367.6/144/1000*0.11*np.cos(bv-bgv)*(2*120*12*(2*175-(450-600*np.cos(2*bgv-bv))))

# plt.contour(np.array([b, bg]), power)
plt.imshow(power, cmap='hot')
plt.figure()
final_bg = np.zeros((len(b),))
# for i in range(808-722):
#     final_bg[i+722] = bg[np.argmax(power[i])]
final_bg[:722] = b[:722]
final_bg[808:] = (b[808:]+46.2378*np.pi/180)/2
slope = (final_bg[808] - final_bg[721])/(b[808] - b[721])
final_bg[722:808] = slope*(b[722:808]-b[721]) + final_bg[721]

plt.plot(b*180/np.pi, final_bg*180/np.pi)
plt.title('BGA vs Beta')
plt.xlabel('Beta')
plt.ylabel('BGA')
plt.show()
