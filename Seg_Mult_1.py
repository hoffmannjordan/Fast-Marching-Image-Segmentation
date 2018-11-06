import numpy as np
import sklearn.cluster

def get_seg(name):
	a = np.genfromtxt(name,delimiter=',')
	print np.shape(a)
	return a

def get_new(data):
	upper = int(np.amax(data))
	print upper
	xs = []
	ys = []
	for j in range(1,upper+1):
		[X,Y] = np.where(data==j)
		XY= np.array([1.*X,1.*Y]).T
		kmeans = sklearn.cluster.KMeans(n_clusters=4, random_state=0).fit(XY) 
		centers = kmeans.cluster_centers_
		centers = np.rint(centers).astype(int)
		#print centers
		for i in range(len(centers)):
			xs.append(centers[i][0])
			ys.append(centers[i][1])
	return xs,ys

if __name__=='__main__':
	data = get_seg('./probability_0_FMM_seg2.csv')
	xs,ys = get_new(data)
	b = np.array([ys,xs]).T
	np.savetxt('./Cents_1.csv',b.astype(int),fmt='%i',delimiter=',')