import numpy as np
import math
from scipy import linalg

np.set_printoptions(formatter={'float_kind':'{:f}'.format})
n= int(input('Write the value of n: '))

# theta_xaqib = np.linspace(0,math.pi,n,endpoint='True')
theta_xaqib = np.linspace(0,math.pi,n,endpoint=True)
# n+=1
# theta_xaqib[0] += 0.005
# theta_xaqib[-1] += -0.005
print(n,theta_xaqib)

# for i in range(len(theta_xaqib)):
    # print(int(theta_xaqib[i]))  

right_t = np.zeros((4,1))
# print(right_t)

# list_const = [int(input(' Enter P, R , v, k, G: ')) for i in range(5)]

list_const = [float(x) for x in input('Enter P, R , v, k, G D: ').split()]
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
    if i==0:
        k_rr_n1.append(2*G/R)
    elif i==1:
        k_rr_n1.append(2*G/R)
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


# for f, z in zip(range(n_x*theta, n_x*(theta+1)), range(n_y*i, n_y*(i+1))):

def fill(big_matrix, small_matrix, i, theta):
    n_x, n_y = small_matrix.shape
    k = 0
    l=0
    for f in range(n_x*theta, n_x*(theta+1)):
        l = 0
        for z in range(n_y*i, n_y*(i+1)):
            big_matrix[f][z] = small_matrix[k][l]
            l+=1
        k +=1
        # l+=1
    return big_matrix


big_matrix = np.zeros((4*n, 4*n))
print(f"len of big matrix array: {big_matrix.shape}")
# print(big_matrix)
right_matrix = np.zeros((4*n,1))

x_right = [3*i + i for i in range(n)]
flt = 0
for theta in theta_xaqib:
    right_matrix[x_right[flt]] = (P*R*(1-v)/G)*((1+k + (2*(1-k)*math.cos(2*theta)))/2)
    flt +=1


theta_range = theta_xaqib.shape[0]
for theta, theta_i in zip(theta_xaqib, range(0,theta_range)):
    for i in range(0,n):
        right_t[0] = (P*R*(1-v)/G)*((1+k + (2*(1-k)*math.cos(2*theta)))/2)
        # if theta==0:
            # theta += 0.01
            #  var_k[2][2] = 1
            # var_k[3][3] =  1
        # if theta == math.pi/2:
            # var_k[0][2] = 0.00000001
            # var_k[1][2] = 0.00000001
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
        big_matrix = fill(big_matrix, var_k, i, theta_i)
        # right_matrix = fill(right_matrix, right_t, )
        # x_right = 3*i + i




# print(f"len of big matrix array: {big_matrix}")
# big_matrix_inverse = np.linalg.inv(big_matrix)

for i in range(big_matrix.shape[0]):
    for j in range(big_matrix.shape[1]):
        if i==j:
            big_matrix[i][j] += 0.000000000000000000000001

print(f"len of big matrix array: {big_matrix}")
big_matrix_inverse = np.linalg.inv(big_matrix)

# print(f"Inverse of a matrix: {big_matrix_inverse}")

print(f"right_matrix is : {right_matrix}")

# print(f"identity if exists: {np.dot(big_matrix,big_matrix_inverse).astype(int)}")

# file = open('x.txt', 'w+')
# file.write(str(big_matrix))
# file.close()
# np.set_printoptions(suppress=True)
final_n_values = big_matrix_inverse.dot(right_matrix)
# final_n_values = np.dot(big_matrix_inverse,right_matrix)
print(f"final needed matrix is: {final_n_values}")

    


