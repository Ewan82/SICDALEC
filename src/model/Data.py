import numpy as np
import os
import re

class dalecData( ): 

    def __init__(self, lenrun, startrun=0, k=1):
        
        #Extract the data
        self.homepath = os.path.expanduser("~")
        self.f = open(self.homepath+"/projects/SICDALEC/data/dalec_drivers.txt","r")
        self.allLines = self.f.readlines()
        self.lenrun = lenrun
        self.startrun = startrun
        self.data = np.array([[-9999.]*9 for i in range(self.lenrun)])
        n = -1
        for x in xrange(self.startrun, self.lenrun+self.startrun):
            n = n + 1
            allVars = self.allLines[x].split()
            for i in xrange(0, 9):
                self.data[n,i] = float(allVars[i])
        
        #'I.C. for carbon pools gCm-2'
        self.Cf = 58.0
        self.Cr = 102.0
        self.Cw = 770.0
        self.Cl = 40.0
        self.Cs = 9897.0
        self.Clist = np.array([[self.Cf,self.Cr,self.Cw,self.Cl,self.Cs]])
        
        #'Background variances for carbon pools & B matrix'
        self.sigB_cf = (self.Cf*0.2)**2 #20%
        self.sigB_cw = (self.Cw*0.2)**2 #20%
        self.sigB_cr = (self.Cr*0.2)**2 #20%
        self.sigB_cl = (self.Cl*0.2)**2 #20%
        self.sigB_cs = (self.Cs*0.2)**2 #20% 
        self.B = np.matrix([[self.sigB_cf,0,0,0,0],[0,self.sigB_cr,0,0,0],
                          [0,0,self.sigB_cw,0,0],[0,0,0,self.sigB_cl,0],
                          [0,0,0,0,self.sigB_cs]])
        self.B2 = np.matrix([[9.44612809e+02,   0.00000000e+00,   0.00000000e+00,
                        0.00000000e+00,   0.00000000e+00],
                        [0.00000000e+00,   2.85089785e+04,   0.00000000e+00,
                        0.00000000e+00,   0.00000000e+00],
                        [0.00000000e+00,   0.00000000e+00,   2.84417115e+07,
                        0.00000000e+00,   0.00000000e+00],
                        [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                        6.98290722e+04,   0.00000000e+00],
                        [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                        0.00000000e+00,   1.17380833e+06]])

        
        #'Observartion variances for carbon pools and NEE' 
        self.sigO_cf = (self.Cf*0.1)**2  # 10%
        self.sigO_cw = (self.Cw*0.1)**2  # 10%
        self.sigO_cr = (self.Cr*0.3)**2  # 30%
        self.sigO_cl = (self.Cl*0.3)**2  # 30%
        self.sigO_cs = (self.Cs*0.3)**2  # 30%
        self.sigO_nee = 0.5  # (gCm-2day-1)**2
        self.sigO_lf = 0.2**2
        self.sigO_lw = 0.2**2
        self.sigO_lai = 0.5**2
        self.sigO_g_resp = 0.5**2
        
        #'Daily temperatures degC'
        self.T_mean = self.data[:,1].tolist()*k
        self.T_max = self.data[:,2].tolist()*k
        self.T_min = self.data[:,3].tolist()*k
        self.T_range = np.array(self.T_max) - np.array(self.T_min)
        self.T = 0.5*(np.exp(0.0693*np.array(self.T_mean))) #Temp term
        
        #'Driving Data'
        self.I = self.data[:,4].tolist()*k #incident radiation
        self.phi_d = self.data[:,5].tolist()*k #max. soil leaf water potential difference
        self.R_tot = self.data[:,7].tolist()*k #total plant-soil hydrolic resistance
        self.C_a = 355.0 #atmospheric carbon
        self.N = 2.7 #average foliar N    
        self.D = self.data[:,0].tolist()*k #day of year
        self.bigdelta = 0.908 #latitutde of forest site in radians
        
        #'Parameters'
        self.a_1 = 2.155
        self.a_2 = 0.0142
        self.a_3 = 217.9
        self.a_4 = 0.980
        self.a_5 = 0.155
        self.a_6 = 2.653
        self.a_7 = 4.309
        self.a_8 = 0.060
        self.a_9 = 1.062
        self.a_10 = 0.0006
        self.p_1 = 0.00000441
        self.p_2 = 0.47
        self.p_3 = 0.31
        self.p_4 = 0.43
        self.p_5 = 0.0027
        self.p_6 = 0.00000206
        self.p_7 = 0.00248
        self.p_8 = 0.0228
        self.p_9 = 0.00000265

    def assimilation_obs(self, obs_str):
        possibleobs = ['gpp', 'lf', 'lw', 'rt', 'nee', 'cf', 'cl', \
                       'cr', 'cw', 'cs']
        Obslist = re.findall(r'[^,;\s]+', obs_str)
    
        for ob in Obslist:
            if ob not in possibleobs:
                raise Exception('Invalid observations entered, please check \
                                 function input')

        Obs_dict = {ob:np.ones(self.lenrun)*-9999. for ob in Obslist}
        Obs_err_dict = {ob+'_err':np.ones(self.lenrun)*-9999. \
                        for ob in Obslist}
  
        n = -1
        for x in xrange(self.startrun, self.lenrun+self.startrun):
            n = n + 1
            allVars = self.allLines[x].split()
            for i in xrange(9, len(allVars)):
                for ob in Obslist:
                    if allVars[i] == ob:
                        Obs_dict[ob][n] = float(allVars[i+1])
                        Obs_err_dict[ob+'_err'][n] = float(allVars[i+2])
                        
        return Obs_dict, Obs_err_dict
        
