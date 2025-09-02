import numpy as np 
import healpy as hp 
class no_data_change: 

    def __call__(self, data):
        return data 
    
    def no_spass_data(self, data_keys):
        return False 

class just_cbass:

    def __call__(self, data): 

        mask = (data['cbass005']['I_smth']==hp.UNSEEN)
        for k,v in data['spass002'].items():
            if 'smth' in k:
                data['spass002'][k][~mask] = hp.UNSEEN

        return data 

    def no_spass_data(self, data_keys):
        if not any(['spass' in k for k in data_keys]):
            return True 
        else:
            return False  
