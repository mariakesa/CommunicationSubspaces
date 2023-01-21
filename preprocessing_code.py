from one.api import ONE
import numpy as np
import brainbox
import pandas as pd
from brainbox.task import trials
from sklearn.linear_model import LinearRegression
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import warnings
from sklearn.model_selection import train_test_split

class EIDData():
    def __init__(self,eid,one):
        self.load_data(eid,one)

    def load_data(self,eid,choice=-1,contrastLeft=1.0):
        self.annot = one.load_dataset(eid, 'alf/_ibl_trials.table.pqt')
        self.df=pd.DataFrame(self.annot)
        print(self.df)
        self.trials=self.df.loc[(self.df['choice'] ==-1)].index
        #Try to download probe01 spike trains, some experiments have 2 probes
        self.spike_times=one.load_dataset(eid, 'alf/probe00/pykilosort/spikes.times.npy')
        self.spike_amps=one.load_dataset(eid, 'alf/probe00/pykilosort/spikes.amps.npy')
        self.spike_depths=one.load_dataset(eid, 'alf/probe00/pykilosort/spikes.depths.npy')
        self.spike_clusters=one.load_dataset(eid, 'alf/probe00/pykilosort/spikes.clusters.npy')
        self.channels=one.load_dataset(eid, 'alf/probe00/pykilosort/clusters.channels.npy')
        self.brain_areas=one.load_dataset(eid, 'channels.brainLocationIds_ccf_2017.npy')
        # Use numpy.take() to select the corresponding brain area for each channel in the clusters.channels file
        self.cluster_brain_areas= np.take(self.brain_areas, self.channels)
        #Can change other events like reward delivery
        self.events=self.annot["stimOn_times"].values

    def make_rasters(self):
        self.raster_dict={}
        print(self.df.shape)
        print(self.events.shape)
        print(self.trials.shape)
        for cl in list(np.unique(self.spike_clusters)):
            #neuron=np.where(self.spike_clusters==n)
            #print(neuron)
            #Can modify parameters for getting rasters
            #print(self.spike_times[self.spike_clusters==cl])
            raster=trials.get_event_aligned_raster(self.spike_times[self.spike_clusters==cl], self.events, tbin=0.1, values=None, epoch=[-0.4, 1], bin=True)[0]
            raster[np.isnan(raster)] = 0.00000001
            #print(raster.shape)
            self.raster_dict[cl]=raster
        #print(self.raster_dict)
        #print(np.where(self.raster_dict[n]==np.nan))


    def make_psth(self):
        self.psth_dict={}
        for n in self.raster_dict.keys():
            psth=trials.get_psth(self.raster_dict[n])
            self.psth_dict[n]=psth

    def trial_to_trial_fluctuations(self):
        self.fluctuations_dict={}
        for n in self.psth_dict.keys():
            sub=self.raster_dict[n]-self.psth_dict[n][0]
            fluctuations=np.nanstd(sub,axis=0)
            self.fluctuations_dict[n]=fluctuations

    def brain_area_dict(self):
        #print(self.cluster_brain_areas)
        #print(self.raster_dict)
        b_lst=[]
        self.brain_area_dict={}
        for b in np.unique(self.cluster_brain_areas):
            inds=np.where(self.cluster_brain_areas==b)[0]
            i_lst=[]
            for i in inds:
                try:
                    i_lst.append(self.fluctuations_dict[i])
                except:
                    pass
            self.brain_area_dict[b]=np.array(i_lst)

            print(self.brain_area_dict[b].shape)

        '''
        #These are single neuron predictions! Not brain areas
        self.brain_area_dict={}
        for i in range(len(self.cluster_brain_areas)):
            self.brain_area_dict[self.channels[i]=self.cluster_brain_areas[i]
        b_lst=[]
        for n1 in len(list(np.unique(self.cluster_brain_areas))):
            b_lst.append([])
        for n2 in fluctuations_dict.keys():
            self.brain_area_dict[n1]
        '''

eid = '58b1e920-cfc8-467e-b28b-7654a55d0977'
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)
dat=EIDData(eid,one)
dat.make_rasters()
dat.make_psth()
#print(dat.psth_dict)
dat.trial_to_trial_fluctuations()
dat.brain_area_dict()
'''
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    with mlflow.start_run():

        one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)


        eid = '58b1e920-cfc8-467e-b28b-7654a55d0977'

        dat=EIDData(eid)
        dat.make_rasters()
        dat.make_psth()
        dat.trial_to_trial_fluctuations()
        fluctuations_dict=dat.fluctuations_dict
        for n1 in fluctuations_dict.keys():
            for n2 in fluctuations_dict.keys():
                if n1!=n2:
                    #print(len(fluctuations_dict.keys()))
                    X=np.array(fluctuations_dict[n1])
                    y=np.array(fluctuations_dict[n2])
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4284)
                    reg = LinearRegression()
                    reg.fit(X_train,y_train)
                    y_pred=reg.predict(X_test)
                    #r2 = cross_val_score(reg, X, y, cv=10,scoring='r2')
                    r2=r2_score(y_pred,y_test)
                    print(r2)
                    #mlflow.sklearn.log_model(reg,"ols_regression")
                    #mlflow.log_metric('R2_score',r2)
'''
