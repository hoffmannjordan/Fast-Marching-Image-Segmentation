import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from mpi4py import MPI
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import scipy.ndimage
from scipy import ndimage
from skimage import measure
import scipy as sp
import time
import glob

def rgb2gray(rgb):
  r, g, b  = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

def import_data(name, cutoff):
  #im = Image.open(name)
  im = np.genfromtxt(name,delimiter=',')
  imarray = np.array(im)
  print np.shape(imarray)
  #imarray = rgb2gray(imarray)
  imarray = scipy.ndimage.filters.gaussian_filter(imarray,0.5)
  print np.shape(imarray)
  dim1,dim2 = np.shape(imarray)
  data = np.zeros((dim1,dim2))
  for i in xrange(dim1):
    for j in xrange(dim2):
      data[i][j] = 1-imarray[i][j]
  data += np.abs(np.amin(data))
  data = data/np.amax(data)
  tmp = np.zeros(np.shape(data))
  for i in xrange(dim1):
    for j in xrange(dim2):
      if data[i][j] > cutoff:
        tmp[i][j] = 1.0
  return tmp


def connected_comp(spec , matrix):
  L = sp.ndimage.measurements.label(matrix)[0]
  maxv = np.amax(L)
  perm = np.random.permutation(range(1,maxv+3))
  dup = np.zeros(np.shape(L))
  dim1,dim2 = np.shape(L)
  for i in xrange(dim1):
    for j in xrange(dim2):
      if L[i][j] != 0:
        dup[i][j] = perm[L[i][j]]
  plt.imshow(L,cmap='nipy_spectral',interpolation='none')
  plt.show()
  plt.imshow(dup,cmap='nipy_spectral',interpolation='none')
  plt.title(spec)
  plt.savefig('./22'+spec+'.png', dpi=300,bbox_inches='tight')
  plt.clf()
  return dup

def by_size(mat,thresh):
  maxv = np.amax(mat)
  locs = [[] for i in xrange(int(maxv))]
  dim1,dim2 = np.shape(mat)
  for i in xrange(dim1):
    for j in xrange(dim2):
      if int(mat[i][j]) != 0:
        locs[int(mat[i][j]-1)].append([i,j])
  centroids = []
  for i in xrange(len(locs)):
    if len(locs[i])>thresh:
      xs,ys = np.transpose(np.array(locs[i]))
      dx = np.amax(xs) - np.amin(xs)
      dy = np.amax(ys) - np.amin(ys)
      if dx > 90 and dy > 90:
        a = range(len(xs))
        a = np.random.permutation(a)
        for j in xrange(100):
          centroids.append([ys[a[j]],xs[a[j]]])
      else:
        cx = np.mean(xs)
        cy = np.mean(ys)
        centroids.append([cy,cx]) 
  xt = np.transpose(centroids)[0]
  yt = np.transpose(centroids)[1]
  return np.array(centroids)    

if __name__=='__main__':
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()  
  file_list = ['./probability_'+str(i)+'.csv' for i in range(0,499)]
  #print file_list
  #exit()
  #print file_list
  ll = len(file_list)
  print ll
  delta = ll/size
  start =  rank*delta
  stop = start +delta #+ 1
  if stop > ll:
    stop=ll
  local_list = file_list[start:stop]
  if rank == 0:
    print 'rank 0'
    print local_list
  #
  #exit()
  for filetodo in local_list:
  #for i in range(90):
    print filetodo
    #filetodo='./probability_'+str(i)+'.csv'
    spec = filetodo.split('/')[1].split('.')[0]
    data = import_data(filetodo , 0.825)
    objects = connected_comp(spec,data)
    np.savetxt('./'+spec+'_vel2.csv',1-data,delimiter=',')
    cs = by_size(objects,2)
    np.savetxt(spec+'_cents2.txt', cs, delimiter=',')