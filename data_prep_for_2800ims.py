import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import pickle

def prepare_data():
  dat_path=('/home/maria/Downloads/natimg2800_M170717_MP033_2017-08-20.mat')
  mat = scipy.io.loadmat(dat_path)
  im_path='/home/maria/Downloads/images_natimg2800_all.mat'
  im2800=scipy.io.loadmat(im_path)['imgs']

  im=mat['stim']['istim'][0][0]
  resp=mat['stim']['resp'][0][0]


  im2800_=im2800.transpose((2,0,1))
  im2800_=im2800.reshape((2800,68,270,1))

  X=[]
  y=[]
  for i in range(0,resp.shape[0]):
      if im[i]!=2801:
          X.append(im2800_[im[i][0]-1])
          y.append(resp[i])

  y=np.array(y)
  y[y<0]=0
  X=np.array(X)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

  stat = mat['stat']['med']
  ypos = np.array([stat[n][0][0][0] for n in range(len(stat))])
  # (notice the python list comprehension [X(n) for n in range(N)])
  xpos = np.array([stat[n][0][0][1] for n in range(len(stat))])

  data_dict={'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test,'xpos':xpos,'ypos':ypos}

  with open('data.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

prepare_data()
