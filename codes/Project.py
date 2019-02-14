import numpy as np
from matplotlib import pyplot as plt
import matplotlib
V = np.matrix('0 0;0 1')
Ut = np.matrix([[8,0]])
U = np.matrix.transpose(Ut)
P = np.matrix([-16.0,16.0]).T
B = np.matrix([[-256]])
F = np.matrix([[0]])
Qa = np.vstack((V,Ut))
Qa = np.hstack((Qa,np.vstack((U,F))))
W = np.matmul((np.hstack((P.T,np.matrix([1])))),Qa)
n_tangent = np.matrix([W[0,0],W[0,1]]).T
omat = np.array([[0,1],[-1,0]])
dir_tangent = np.matmul(np.linalg.inv(omat),n_tangent)

# Finding A and B
def find_xintercept(P,n):
	temp = -P[1,0]/n[1,0]
	temp = P[0,0]+temp*n[0,0]
	return np.matrix([temp,0]).T
A = find_xintercept(P,dir_tangent)
B = find_xintercept(P,n_tangent)

# Center of Circle is the mid point of AB as PB and PA are perpendicular
# So AB is the diameter of Circle
C = np.matrix([(A[0,0]+B[0,0])/2,0]).T
radius = A[0,0]-B[0,0]/2

# AS PB and PC are radii of circle PB = PC
# Therefore angle_CBP = angle_CPB
# angle_CBP = slope of normal at P
angle_CPB = n_tangent[1,0]/n_tangent[0,0]
print'Angle_CPB =',angle_CPB
# Ploting
len = 10
lam_1 = np.linspace(-2,0,len)
x_PA = np.zeros((2,len))
x_PB = np.zeros((2,len))
for i in range(len):
	temp1 =P + lam_1[i]*dir_tangent
	x_PA[:,i]= temp1.T
	# Ploting
lam_1 = np.linspace(-1,0,len)
x_PB = np.zeros((2,len))
x_PC = np.zeros((2,len))
for i in range(len):
	temp1 = P + lam_1[i]*n_tangent
	x_PB[:,i]=temp1.T
	temp1 = P + lam_1[i]*(P-C)
	x_PC[:,i]=temp1.T
fig, ax = plt.subplots()
plt.plot(x_PA[0,:],x_PA[1,:],label = '$PA-Tangent$')
plt.plot(x_PB[0,:],x_PB[1,:],label = '$PB-Normal$')
plt.plot(x_PC[0,:],x_PC[1,:],label = '$CP = CB$',color = 'gray')
C1 = plt.Circle((C[0,0],C[1,0]),radius = 20,color = 'b',fill=False)
ax.add_artist(C1)
y = np.linspace(-20,+20,1000)
x = -y**2/16
plt.plot(x,y)
plt.axhline(y=0,color='gray')
plt.axvline(x=0,color='gray')
plt.plot(P[0,0],P[1,0],'ok')
plt.text(P[0,0]+0.3,P[1,0]+0.8,'P')
plt.plot(A[0,0],A[1,0],'ok')
plt.text(A[0,0]+0.5,A[1,0]+0.5,'A')
plt.plot(B[0,0],B[1,0],'ok')
plt.text(B[0,0]+0.5,B[1,0]+0.5,'B')
plt.plot(C[0,0],C[1,0],'ok')
plt.text(C[0,0]+0.5,C[1,0]+0.5,'C')
plt.legend(loc='best')
plt.show()