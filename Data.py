import numpy as np
import ad


class dalecData( ): 

  def __init__(self):
    
    self.f=open("dalec_drivers.txt","r")
    self.allLines=self.f.readlines()
    self.lenrun=1095
    
    #'Observation data for DA'
    self.data=np.matrix([[None]*9 for i in range(self.lenrun)])
    self.gpp=[None]*self.lenrun
    self.gpp_err=[None]*self.lenrun
    self.lf=[None]*self.lenrun
    self.lf_err=[None]*self.lenrun
    self.lw=[None]*self.lenrun
    self.lw_err=[None]*self.lenrun
    self.rt=[None]*self.lenrun
    self.rt_err=[None]*self.lenrun
    self.nee=[None]*self.lenrun
    self.nee_err=[None]*self.lenrun
    self.cf=[None]*self.lenrun
    self.cf_err=[None]*self.lenrun

    n=-1
    for x in xrange(0,self.lenrun):
        n=n+1
        allVars=self.allLines[x].split()
        for i in xrange(0,9):
            self.data[n,i]=float(allVars[i])
        for i in xrange(9,len(allVars)):
            if allVars[i]=='gpp':
                self.gpp[n]=float(allVars[i+1])
                self.gpp_err[n]=float(allVars[i+2])
            elif allVars[i]=='lf':
                self.lf[n]=float(allVars[i+1])
                self.lf_err[n]=float(allVars[i+2])
            elif allVars[i]=='lw':
                self.lw[n]=float(allVars[i+1])
                self.lw_err[n]=float(allVars[i+2])
            elif allVars[i]=='rt':
                self.rt[n]=float(allVars[i+1])
                self.rt_err[n]=float(allVars[i+2])
            elif allVars[i]=='nee':
                self.nee[n]=float(allVars[i+1])
                self.nee_err[n]=float(allVars[i+2])
            elif allVars[i]=='cf':
                self.cf[n]=float(allVars[i+1])
                self.cf_err[n]=float(allVars[i+2])

    #'I.C. for carbon pools gCm-2'
    self.Cf=58.0
    self.Cr=102.0
    self.Cw=770.0
    self.Cl=40.0
    self.Cs=9897.0
    self.Clist=[[self.Cf,self.Cr,self.Cw,self.Cl,self.Cs]]

    #'Background variances for carbon pools & B matrix'
    self.sigB_cf=(self.Cf*0.2)**2 #20%
    self.sigB_cw=(self.Cw*0.2)**2 #20%
    self.sigB_cr=(self.Cr*0.2)**2 #20%
    self.sigB_cl=(self.Cl*0.2)**2 #20%
    self.sigB_cs=(self.Cs*0.2)**2 #20% 
    self.B=np.matrix([[self.sigB_cf,0,0,0,0],[0,self.sigB_cw,0,0,0],[0,0,self.sigB_cr,0,0],[0,0,0,self.sigB_cl,0],[0,0,0,0,self.sigB_cs]])

    #'Observartion variances for carbon pools and NEE' 
    self.sigO_cf=(self.Cf*0.1)**2 #10%
    self.sigO_cw=(self.Cw*0.1)**2 #10%
    self.sigO_cr=(self.Cr*0.3)**2 #30%
    self.sigO_cl=(self.Cl*0.3)**2 #30%
    self.sigO_cs=(self.Cs*0.3)**2 #30% 
    self.sigO_nee=0.5**2 #(gCm-2day-1)**2

    #'Daily temperatures degC'
    self.T_mean=self.data[:,1]
    self.T_max=np.array(self.data[:,2])
    self.T_min=np.array(self.data[:,3])

    self.T_range=self.T_max-self.T_min
    self.T=[None]*len(self.T_mean)
    for i in range(len(self.T_mean)):
        self.T[i]=0.5*(np.exp(0.0693*float(self.T_mean[i]))) #Temp term

    #'Driving Data'
    self.I=self.data[:,4] #incident radiation
    self.phi_d=self.data[:,5] #max. soil leaf water potential difference
    self.R_tot=self.data[:,7] #total plant-soil hydrolic resistance
    self.C_a=355.0 #atmospheric carbon
    self.N=2.7 #average foliar N    
    self.D=self.data[:,0]#day of year
    self.bigdelta=0.908 #latitutde of forest site in radians
    
    #'Parameters'
    self.a_1=2.155
    self.a_2=0.0142
    self.a_3=217.9
    self.a_4=0.980
    self.a_5=0.155
    self.a_6=2.653
    self.a_7=4.309
    self.a_8=0.060
    self.a_9=1.062
    self.a_10=0.0006
    self.p_1=0.00000441
    self.p_2=0.47
    self.p_3=0.31
    self.p_4=0.43
    self.p_5=0.0027
    self.p_6=0.00000206
    self.p_7=0.00248
    self.p_8=0.0228
    self.p_9=0.00000265



