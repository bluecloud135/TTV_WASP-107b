import numpy as np
import pandas as pd
from glob import glob

class GetData():
    def __init__(self, target, indir, period, IDs, saveData = False):
        self.target= target
        self.indir = indir
        self.period = period
        self.IDs = IDs
        self.saveData = saveData

    def getPaths(self):
        csvs = glob(self.indir+'*/'+'*fitparams*.csv')
        dats = glob(self.indir+'*/'+'posteriors.dat')
        paths = sorted(csvs + dats, key=lambda x: x[:12])
        return paths
    
    def getTime(self):
        paths = self.getPaths()
        median = []
        up = []
        down = []
        ini_path = []
        for path in paths:
            if '.csv' in path:
                now = pd.read_csv(path)
                now.set_index('Parameter',inplace=True)
                t0_name = 't0'
            else:
                now = pd.read_csv(path,sep=r'\t\s+', names=['Parameter','50th','+1sigma','-1sigma'], engine='python',skiprows=1)
                now.set_index('Parameter',inplace=True)
                now.index = [x.strip(' ') for x in now.index]
                t0_name = 't0_p1'

            median.append(float(now.loc[t0_name,'50th']))
            up.append(float(now.loc[t0_name,'+1sigma']))
            down.append(abs(float(now.loc[t0_name,'-1sigma'])))
            ini_path.append(path)
        time = pd.DataFrame({'Median':median, '+1 sigma':up, '-1 sigma':down, 'Path':ini_path})
        time['Median'] += 2400000.5
        time.sort_values(['Median'])
        time.index = range(len(time))
        epoch = round((time['Median']-time.loc[0,'Median'])/self.period)
        time['epoch'] = [int(x) for x in epoch]
        time['ID'] = self.IDs
        if self.saveData:
            time.to_csv(self.indir+'time.csv', index=False)
            time.to_latex(self.indir+'time.tex', columns=['Median','+1 sigma','-1 sigma','ID'], index=False, float_format="%.7f")
        return time
    
    def getHST(self,path):
        hst = pd.read_excel(path)
        try:
            htime = hst[hst['name']==self.target]['T_mid_i']
            herror = [max(x,y) for x,y in zip(hst[hst['name']==self.target]['m_error_i'],hst[hst['name']==self.target]['p_error_i'])]
            hst_time = pd.DataFrame({'T_mid':htime, 'Uncertainty':herror, 'Label':['HST-Ma']*len(htime), 'Reference':['Ma et al. 2024']*len(htime)})
            HST = True
        except:
            HST = False
            hst_time = None
        return HST, hst_time
    
    def getTESS(self,path1, path2, target):
        tess_all = pd.read_csv(path1)
        tess_all = tess_all[tess_all['System']==target]
        others = tess_all[tess_all['Reference']!='This work']
        others_time = pd.DataFrame({'T_mid':others['T_mid'],'Uncertainty':others['Uncertainty (days)'],'Label':['Literature']*len(others),'Reference':others['Reference']})
        try:
            tess = pd.read_csv(path2,names=['Orbit number','T_mid','Uncertainty (days)'])
            tess_time = pd.DataFrame({'T_mid':tess['T_mid'],'Uncertainty':tess['Uncertainty (days)'],'Label':['TESS-Wang']*len(tess),'Reference':['Wang et al. 2024']*len(tess)})

        except:
            tess = tess_all[tess_all['Reference']=='This work']
            tess_time = pd.DataFrame({'T_mid':tess['T_mid'],'Uncertainty':tess['Uncertainty (days)'],'Label':['TESS-I&W']*len(tess),'Reference':['Ivshina & Winn 2022']*len(tess)})
        return tess_time,others_time

    def getT(self,hst_path, tess_path1, tess_path2, tess_target):
        HST, hst_time = self.getHST(hst_path)
        tess_time, others_time = self.getTESS(tess_path1, tess_path2, tess_target)
        time = self.getTime()
        time['Uncertainty'] = np.max([time['+1 sigma'],time['-1 sigma']],axis=0)
        jwst_time = pd.DataFrame({'T_mid':time['Median'], 'Uncertainty':time['Uncertainty'],'Label':['JWST']*len(time),'Reference':['This Work']*len(time)})
        if HST:
            t = pd.concat([tess_time, hst_time, jwst_time,others_time])
        else:
            t = pd.concat([tess_time, jwst_time,others_time])
        t = t.sort_values(['T_mid'])
        t.index = range(len(t))
        epoch = round((t['T_mid']-t.loc[0,'T_mid'])/self.period)
        t['Epoch'] = [int(x) for x in epoch]
        if self.saveData:
            t.to_csv(self.indir+'t.csv', index=False)
            t.to_latex(self.indir+'t.tex', index=False, float_format="%.7f")
        return t
    
    def checkDuplicates(self, t):
        T_mid_re = []
        epoch_re = []
        uncertainty_re = []
        label_re = []
        reference_re = []
        rej_temp = 5
        duplicate = len(set(t['Epoch'])) != len(t['Epoch'])
        if duplicate:
            print('Duplicate.')
            for epo in set(t['Epoch']):
                temp = t[t['Epoch']==epo].copy()
                if len(temp)>1:
                    std_temp = np.std(temp['T_mid'],ddof=1)
                    if std_temp == 0:
                        temp_re = temp.copy()
                        T_mid_re.append(temp_re['T_mid'].mean())
                        epoch_re.append(epo)                
                        uncertainty_re.append(temp_re['Uncertainty'].max())
                        label_re.append('&'.join(set(temp_re['Label'])))
                        reference_re.append('&'.join(set(temp_re['Reference'])))
                    else:
                        temp_re = temp[abs(temp['T_mid']-temp['T_mid'].mean())<rej_temp*std_temp]
                        scale = temp_re['Uncertainty'].min()/temp_re['Uncertainty']
                        scale /= scale.sum()
                        T_mid_re.append(sum([x*y for x,y in zip(temp_re['T_mid'],scale)]))
                        epoch_re.append(epo)                
                        uncertainty_re.append(sum([x*y for x,y in zip(temp_re['Uncertainty'],scale)]))
                        label_re.append('&'.join(set(temp_re['Label'])))
                        reference_re.append('&'.join(set(temp_re['Reference'])))
                else:
                    T_mid_re.append(temp['T_mid'].values[0])
                    epoch_re.append(epo)                
                    uncertainty_re.append(temp['Uncertainty'].values[0])
                    label_re.append(temp['Label'].values[0])
                    reference_re.append(temp['Reference'].values[0])
            t_re = pd.DataFrame({'Epoch':epoch_re,'T_mid':T_mid_re,'Uncertainty':uncertainty_re,'Label':label_re,'Reference':reference_re})
            t_re = t_re.sort_values(['T_mid'])
            t_re.index = range(len(t_re))
            if self.saveData:
                t_re.to_csv(self.indir+'t_re.csv', index=False)
                t_re.to_latex(self.indir+'t_re.tex', index=False, float_format="%.7f")
            return t_re
        else:
            print('No duplicate.')