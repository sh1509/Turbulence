import numpy as np
import math

n= int(input('Write the value of n: '))

# theta_xaqib = np.linspace(0,math.pi,n,endpoint='True')
theta_xaqib = np.linspace(0,math.pi,n,endpoint='False')
theta_xaqib[0] += 0.005
theta_xaqib[-1] += -0.005
print(n,theta_xaqib)

# for i in range(len(theta_xaqib)):
    # print(int(theta_xaqib[i]))  

right_t = np.zeros((4,1))
print(right_t)

# list_const = [int(input(' Enter P, R , v, k, G: ')) for i in range(5)]

list_const = [int(x) for x in input('Enter P, R , v, k, G D: ').split()]
# list_const = input('Enter P, R , v, k, G: ')
print(list_const)

P = list_const[0]
R = list_const[1]
v = list_const[2]
k = list_const[3]
G = list_const[4]
D = list_const[5]

print(f"P: {P} R: {R} v: {v} k: {k} G: {G} D: {D}")

# RIght side of the matrix is defined here:
# This is used with f during calculations
# right_t[0] = (P*R*(1-v)/G)*((1+k + (2*(1-k)*f))/2)
# define f for each 4*4 matrix
# f = math.cos()

var_k = np.zeros((4,4))
k_rr_n1 = []
for i in range(n):
    print(f"i:{i}")
    if i == 0:
        k_rr_n1.append(2*G/R)
        # continue
    elif i == 1:
        k_rr_n1.append(2*G/R)
        # continue
    else:
        k_rr_n1.append(2*G*((2*i + 1) - (2*v*(i+1)))/(R*(3-4*v)))
print(f"krr_n1: {k_rr_n1}")
krr_n2 = [((D / (R**2)) + (k / (R**4))*(i**2 -1)*(i**2 -1)) for i in range(n)]
print(f"krr_n2: {krr_n2}")
# krr_n2 = [] 
# for i in range(5):
    # krr_n2.append((D / (R**2)) + (k / (R**4))*(i**2 -1)(i**2 -1)) 
# print(f"krr_n2: {krr_n2}")

krt_n1 = []
for i in range(n):
    if i==0:
        krt_n1.append(0)
    elif i==1:
        krt_n1.append(2*G/R)
    else:
        krt_n1.append(2*G*((i + 2) - (2*v*(i+1)))/(R*(3-4*v)))

print(f"krt_n1: {krt_n1}")
krt_n2 = [i*D/(R**2) for i in range(n)]
print(f"krt_n2: {krt_n2}")

ktt_n1 = []
for i in range(n):
    if i==0:
        ktt_n1.append(0)
    elif i==1:
        ktt_n1.append(2*G/R)
    else:
        ktt_n1.append(2*G*((2*i + 1) - (2*v*(i+1)))/(R*(3-4*v)))

print(f"ktt_n1: {ktt_n1}")
ktt_n2 = [((i**2)*D/(R**2)) for i in range(n)]
print(f"ktt_n2: {ktt_n2}")

# defining the matrix value for final calculation:
# var_k[0][0] = math.cos()


def fill(big_matrix, small_matrix, i, theta):
    pass

big_matrix = np.zeros((4*n, 4*n))
print(f"len of big matrix array: {len(big_matrix)}")
print(big_matrix)

for theta, theta_i in theta_xaqib, range(len(theta_xaqib)):
    for i in range(1,n):
        right_t[0] = (P*R*(1-v)/G)*((1+k + (2*(1-k)*math.cos(2*theta)))/2)
        var_k[0][0] = math.cos(i*theta)
        var_k[0][1] = -math.cos(i*theta)
        var_k[1][0] = k_rr_n1[i]*math.cos(i*theta)
        var_k[1][1] = krr_n2[i]*math.cos(i*theta)
        var_k[1][2] = krt_n1[i]*math.cos(i*theta)
        var_k[1][3] = krt_n2[i]*math.cos(i*theta)
        var_k[2][0] = krt_n1[i]*math.sin(i*theta)
        var_k[2][2] = ktt_n1[i]*math.sin(i*theta)
        var_k[3][1] = krt_n2[i]*math.sin(i*theta)
        var_k[3][3] = ktt_n2[i]*math.sin(i*theta)
        # print(f"var_k_{i} theta_{theta*180/math.pi}: {var_k} ")
        # alpha = np.linalg.inv(var_k)
        # print(f"alpha_{i} theta_{theta*180/math.pi}: {alpha}")

    
    


