from scipy import sparse
import scipy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import sqrt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


np.set_printoptions(threshold=100)
def vec_to_grid(vec, N):
    grid = []
    for i in range(0,N):
        grid.append(list(vec[i*N:(i+1)*N]))
    return grid


def matrix_clamped_BC(N):
    molecule = [[0 ,0  ,1  ,0  ,0],
                [0 ,2  ,-8 ,2  ,0],
                [1 ,-8 ,20 ,-8 ,1],
                [0 ,2  ,-8 ,2  ,0],
                [0 ,0  ,1  ,0  ,0]]


    nab = sparse.lil_matrix((N**2,N**2))

    for i in range(0,N):
        for j in range(0,N):
            for Di in [-2,-1,0,1,2]:
                for Dj in [-2,-1,0,1,2]:
                    if i +Di in range(0,N) and j +Dj in range(0,N):
                        nab[N*i + j, N*(i+Di) + (j+Dj)] = molecule[Di+2][Dj+2]*(N-1)**4

    for j in range(0,N):
        nab[j,j] +=(N-1)**4
    for j in range(0,N):
        nab[N*(N-1) + j, N*(N-1) + j] +=(N-1)**4
    for i in range(0,N):
        nab[N*i, N*i] +=(N-1)**4
    for i in range(0,N):
        nab[N*i + N -1 , N*i + N-1] +=(N-1)**4

    return nab

def matrix_free_BC(N,v, free_edges):
    h=1/N
    n = N+1

    a= -2*(v**2 + 2*v -3)
    b = 1 - v**2
    c = -2*(v-1)
    d = 15 - 8*v -5*v**2
    e = - 4 *(v**2+ v - 2)
    f = 2-v
    g = -2*(v-3)
    k = -2*(3*v**2 + 4*v - 8)

    In=np.identity(n)
    
    I0n=np.identity(n)
    I0n[0,0]=0
    I0n[1,1]=0
    I0n[n-2,n-2]=0
    I0n[n-1,n-1]=0
    
    Jn=np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1)
    jn=np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1)
    jn[0,1]=sqrt(2)
    jn[1,0]=sqrt(2)
    jn[n-2,n-1]=sqrt(2)
    jn[n-1,n-2]=sqrt(2)
    
    J0n=np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1)
    J0n[0,1]=0
    J0n[1,0]=0
    J0n[n-2,n-1]=0
    J0n[n-1,n-2]=0
    
    J2n=np.diag(np.ones(n-2),2)+np.diag(np.ones(n-2),-2)
    j2n=np.diag(np.ones(n-2),2)+np.diag(np.ones(n-2),-2)
    j2n[0,2]=sqrt(2)
    j2n[2,0]=sqrt(2)
    j2n[n-3,n-1]=sqrt(2)
    j2n[n-1,n-3]=sqrt(2)
    
    Fn=np.zeros((n,n))
    Fn[1,1]=1
    Fn[n-2,n-2]=1
    
    Gn=np.zeros((n,n))
    Gn[0,0]=1
    Gn[n-1,n-1]=1
    
    Hn=np.zeros((n,n))
    Hn[1,0]=1
    Hn[n-2,n-1]=1
    
    An=In
    An[0,0]=b
    An[n-1,n-1]=b

    
    Bn=-8*In+2*Jn
    Bn[0,0]=-e
    Bn[n-1,n-1]=-e
    Bn[0,1]=sqrt(2)*f
    Bn[1,0]=sqrt(2)*f
    Bn[n-2,n-1]=sqrt(2)*f
    Bn[n-1,n-2]=sqrt(2)*f
    
    Cn=-g*In+f*Jn
    Cn[0,0]=-a
    Cn[n-1,n-1]=-a
    Cn[0,1]=sqrt(2)*f
    Cn[1,0]=sqrt(2)*c
    Cn[n-2,n-1]=sqrt(2)*c
    Cn[n-1,n-2]=sqrt(2)*f
    
    Dn=20*In-8*Jn+j2n
    Dn[0,0]=k
    Dn[n-1,n-1]=k
    Dn[1,1]=19
    Dn[n-2,n-2]=19
    Dn[0,1]=-sqrt(2)*g
    Dn[1,0]=-sqrt(2)*g
    Dn[n-2,n-1]=-sqrt(2)*g
    Dn[n-1,n-2]=-sqrt(2)*g
    
    DDn=19*In-8*Jn+j2n
    DDn[0,0]=d
    DDn[n-1,n-1]=d
    DDn[1,1]=18
    DDn[n-2,n-2]=18
    DDn[0,1]=-sqrt(2)*g
    DDn[1,0]=-sqrt(2)*g
    DDn[n-2,n-1]=-sqrt(2)*g
    DDn[n-1,n-2]=-sqrt(2)*g
    
    En=k*In-e*Jn+b*j2n
    En[0,0]=2*a
    En[n-1,n-1]=2*a
    En[1,1]=d
    En[n-2,n-2]=d
    En[0,1]=-sqrt(2)*a
    En[1,0]=-sqrt(2)*a
    En[n-2,n-1]=-sqrt(2)*a
    En[n-1,n-2]=-sqrt(2)*a


    P2=(np.kron(j2n,An)+np.kron(J0n,Bn)+np.kron(I0n,Dn)+np.kron(Fn,DDn)+np.kron(Gn,En)+np.kron(Hn,sqrt(2)*Cn)+np.kron(Hn,sqrt(2)*Cn).transpose())/(h*h*h*h)

    return P2




N= 32  ###side length of plate
v=0.3  ###passions ratio for plate

bi_lep= matrix_free_BC(N,v, [])
##bi_lep= matrix_clamped_BC(N)


plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,labeltop=False) # labels along the bottom edge are off

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,labeltop=False) # labels along the bottom edge are off

print("matrix done")



vals, vecs = np.linalg.eig(bi_lep)
idx = vals.argsort()  
vals = vals[idx]
vecs = vecs[:,idx]

with plt.style.context("seaborn-white"):
    fig,axs=plt.subplots(2,5)

    
    A = np.real(np.array(vec_to_grid(vecs[:, 6],N+1)))
    B = np.real(np.array(vec_to_grid(vecs[:, 7],N+1)))
    
    for i in range(0,5):
        axs[0][i].imshow(A*(i/4)+ B*(1-i/4), vmin = -0.03, vmax =0.03, rasterized=True, cmap = 'bwr')
        axs[1][i].contour(range(0,N+1), range(0,N+1), A*(i/4)+ B*(1-i/4), levels=[0],colors='red')
        axs[0][i].set_title("c = "+ str(i/4))

        axs[0][i].axis('scaled')
        axs[1][i].axis('scaled')

        plt.setp(axs[0][i].get_xticklabels(), visible=False)
        plt.setp(axs[0][i].get_yticklabels(), visible=False)
        plt.setp(axs[1][i].get_xticklabels(), visible=False)
        plt.setp(axs[1][i].get_yticklabels(), visible=False)

    plt.show()
    
    
    for k in range(0,(N+1)*(N+1)):
        grid = np.array(vec_to_grid(vecs[:,k],N+1))
        grid = np.real(grid)
        plt.imshow(grid, vmin = -0.03, vmax =0.03, rasterized=True, cmap = 'bwr')
        plt.show()

    #plt.show()
    #plt.savefig('node' +str(k) + '.png')

