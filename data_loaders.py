import pickle
import numpy as np
import h5py

classes = ['bpsk', 'qpsk', '8psk','16qam']
print('Loading dataset....')

# if the data is in a mat file use this code
with h5py.File("NomaDL_2022_v2.mat", 'r') as Xd:
	#Xd = f3
	X = np.array(Xd['data_Y'])
	lbl = np.array(Xd['true_Mods'])
	snrs = np.array(Xd['snrs'])
cplx = X['real'] + X['imag']*1j
c_t = cplx.transpose()
Xs = np.zeros((c_t.shape[0],2,c_t.shape[1]))
Xs[:,0,:] = c_t.real
Xs[:,1,:] = c_t.imag
#=================

# if the data is in a pickle file use this code
with open("NomaDL_2022_v2.pkl", 'rb') as f:
	Xd = pickle.load(f, endoding = 'latin1')
	
X = Xd['signals']
lbl = Xd['mod_type']
snrs = Xd['snr']
Xs = np.zeros((X.shape[0],2,200))
Xs[:,0,:] = X.real
Xs[:,1,:] = X.imag
#=====================