from one.api import ONE
import numpy as np
import brainbox
import pandas as pd
from brainbox.task import trials

one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)


eid = '58b1e920-cfc8-467e-b28b-7654a55d0977'

class EIDData():
    def __init__(self,eid):
        self.load_data(eid)

    def load_data(eid,choice=1,contrastLeft=1.0):
        self.annot = one.load_dataset(eid, 'alf/_ibl_trials.table.pqt')
        df=pd.DataFrame(annot)
        self.trials=df.loc[(df['choice'] == choice) & (df['contrastLeft'] == contrastLeft)].index
        self.spike_times=one.load_dataset(eid, 'alf/probe00/pykilosort/spikes.times.npy')
        self.spike_amps=one.load_dataset(eid, 'alf/probe00/pykilosort/spikes.amps.npy')
        self.spike_depths=one.load_dataset(eid, 'alf/probe00/pykilosort/spikes.depths.npy')
        self.spike_clusters=one.load_dataset(eid, 'alf/probe00/pykilosort/spikes.clusters.npy')
        self.channels=one.load_dataset(eid, 'alf/probe00/pykilosort/clusters.channels.npy')
        self.brain_areas=one.load_dataset(eid, 'channels.brainLocationIds_ccf_2017.npy')
        # Use numpy.take() to select the corresponding brain area for each channel in the clusters.channels file
        self.cluster_brain_areas= np.take(self.brain_areas, self.channels)
        #Can change other events like reward delivery
        self.events=["stimOn_times"]

    def make_rasters(self):
        self.raster_dict={}
        for n in list(np.unique(self.spike_clusters)):
            neuron=np.where(self.spike_clusters==n)
            if len(list(neuron[0]))!=0:
                #Can modify parameters for getting rasters
                raster=trials.get_event_aligned_raster(spike_times[neuron], events, tbin=0.1, values=None, epoch=[-0.4, 1], bin=True)
                if np.isnan(raster[0]).any()==False:
                    self.raster_dict[n]=raster[0]

    def make_psth(self):
        self.psth_dict={}
        for n in self.raster_dict.keys():
            psth=get_psth(self.raster[n],trial_ids=self.trials)
            self.psth_dict[n]=psth

    def sub_psth_comp_std(self):
        self.fluctuations_dict={}
        for n in self.raster_dict.keys():
            sub=self.raster_dict[n]-self.psth_dict[n]
            fluctuations=np.std(sub)
            self.fluctuations_dict[n]=fluctuations
