import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

class LightCurve:
    def __init__(self, target, indir, period, nbin = None, savePlots = False):
        self.target= target
        self.indir = indir
        self.period = period
        self.nbin = nbin
        self.savePlots = savePlots
    
    def plotTransits(self,time):
        fig, ax = plt.subplots(figsize=(16,9))
        xdata = time['epoch']
        ydata = time['Median']
        error = [time['-1 sigma'],time['+1 sigma']]

        ax.plot(xdata, ydata, '-', color='dimgray')
        ax.errorbar(xdata, ydata, yerr=error,
            ms=15 , fmt='.', zorder=10, color='white',ecolor='dimgray',mec='deepskyblue',elinewidth=1,capsize=3, label='time')
        ax.tick_params(direction='in', which='both', labelsize=18)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.set_xticks(xdata)
        ax.set_xlabel(r'$epoch$',fontsize=23)
        ax.set_ylabel(r'$t_0$[BJD-TDB]',fontsize=23)
        plt.tight_layout()
        if self.savePlots:
            plt.savefig(self.indir+'T-epoch.png', facecolor='white', dpi=300)
        plt.show()

    @staticmethod
    def binData(data, nbin=100, err=False):
        newdata = np.array(data[:nbin*int(len(data)/nbin)])
        # Bin data
        binned = np.nanmean(newdata.reshape(nbin, -1), axis=1)
        if err:
            binned /= np.sqrt(int(len(data)/nbin)-1)
        return binned

    def plotLightCurve(self,indir,t0,id=''):
        files = glob(indir+'*')
        n = 0
        for file in files:
            if '.npy' in file:
                items = np.load(file)
                n += 1
            if 'lc.dat' in file:
                lc = pd.read_csv(file,names=['time','relative flux','error','instrument'],sep=' ')
                n += 1
            if '.txt' in file:
                white = pd.read_csv(file,comment='#',sep=' ')
                n += 1
        if (n==2):
            data = pd.DataFrame({'time':lc['time'],'relative flux':lc['relative flux'],'error':np.concatenate(items[4]),'model':np.concatenate(items[0])})
        elif (n==1):
            data = pd.DataFrame({'time':white['time'],'relative flux':white['lcdata'],'error':white['lcerr'],'model':white['model']})
        else:
            print('Error')
            return
        center = data['model'].max()-1
        data['relative flux'] -= center
        data['model'] -= center
        data['residuals'] = (data['relative flux'] - data['model'])
        data['t'] = (data['time']-t0+2400000.5)*24
        fig,ax = plt.subplots(2,1,figsize = (20,12),sharex=True, height_ratios=[3,1])

        if self.nbin is not None:
            ax[0].set_title(r'White Light Curve of %s (ID = %s, $t_0 = $ %.6f BJD-TDB, nbins = %s)'%(self.target,id,t0,self.nbin), fontsize=28)
            binned_flux = self.binData(data['relative flux'], self.nbin)
            binned_t = self.binData(data['t'], self.nbin)
            binned_residuals = self.binData(data['residuals'], self.nbin)
            binned_error = self.binData(data['error'], self.nbin, err=True)
            ax[0].errorbar(binned_t, binned_flux, yerr=binned_error, ms=15 , fmt='.',
                            color='white',ecolor='deepskyblue',mec='deepskyblue')
            ax[1].errorbar(binned_t, binned_residuals*1e6, yerr=binned_error*1e6,
                ms=15 , fmt='.', zorder=10,color='white',ecolor='dimgray',mec='deepskyblue',elinewidth=1,capsize=3)
            ax[1].axhline(0, ls='--', c='dimgray')
            xpos = np.percentile(binned_t, 1)
            sigma = binned_residuals.std(ddof=1)*1e6
            ax[1].text(xpos, (np.max(binned_residuals*1e6) + np.max(binned_error*1e6)*0.70),
                        'std = {:.2f}ppm'.format(sigma),fontsize=26)
            # ax[1].fill_between(binned_t, -binned_error*1e6, binned_error*1e6, color='dimgray', alpha=0.1)
            suffix = 'bin%s'%self.nbin
        else:
            ax[0].set_title(r'White Light Curve of %s (ID = %s, $t_0 =$ %.6f BJD-TDB)'%(self.target,id,t0), fontsize=16)
            ax[0].errorbar(data['t'], data['relative flux'], yerr=data['error'], ms=15 , fmt='.',
                            color='white',ecolor='deepskyblue',mec='deepskyblue')
            ax[1].errorbar(data['t'], data['residuals']*1e6, yerr=data['error']*1e6,
                ms=15 , fmt='.', zorder=10,color='white',ecolor='dimgray',mec='deepskyblue',elinewidth=1,capsize=3)
            ax[1].axhline(0, ls='--', c='dimgray')
            xpos = np.percentile(data['t'], 1)
            sigma = data['residuals'].std(ddof=1)*1e6
            ax[1].text(xpos, (np.max(data['residuals']*1e6) + np.max(data['error']*1e6)*0.70),
                        'std = {:.2f}ppm'.format(sigma),fontsize=26)
            # ax[1].fill_between(data['t'], -data['error']*1e6, data['error']*1e6, color='dimgray', alpha=0.1)
            suffix = 'full'
        ax[0].plot(data['t'], data['model'], color='dimgray', zorder=10)
        ax[0].set_ylabel('Relative Flux', fontsize=23)
        ax[0].set_xlim(np.min(data['t']), np.max(data['t']))
        # ax[0].xaxis.set_major_formatter(plt.NullFormatter())
        ax[0].tick_params(direction='in', which='both', labelsize=18)
        ax[0].xaxis.set_ticks_position('both')
        ax[0].yaxis.set_ticks_position('both')

        ax[1].set_xlim(np.min(data['t']), np.max(data['t']))
        ax[1].set_ylabel('Residuals\n(ppm)', fontsize=23)
        ax[1].set_xlabel('Time from Transit Midpoint [hrs]', fontsize=23)
        ax[1].tick_params(direction='in', which='both', labelsize=18)
        ax[1].xaxis.set_ticks_position('both')
        ax[1].yaxis.set_ticks_position('both')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        if self.savePlots:
            plt.savefig(indir+'white_lightcurve_%s_%s.png'%(id,suffix), facecolor='white',dpi=300)
            plt.savefig(indir+'white_lightcurve_%s_%s.pdf'%(id,suffix), facecolor='white',dpi=600)
        plt.show()
        return data
    
    def plotAllClightCurves(self,time):
        datas  =[]
        t0s = []
        ids = []
        for dir, t0, id in zip(['/'.join(s.split('/')[:-1])+'/' for s in time['Path']], time['Median'], time['ID']):
            datas.append(self.plotLightCurve(dir, t0, id))
            t0s.append(t0)
            ids.append(id)
        return t0s, datas, ids
    
    def plotLightCurveTogether(self, t0s, datas, ids):
        fig,ax = plt.subplots(figsize=(20, 16), facecolor='white', rasterized=False)
        offset = 0
        xpos = max([max(data['t']) for data in datas])*0.38
        for t0, data, id in zip(t0s, datas, ids):
            offset += 0.02
            center = data['model'].max()-1
            data['relative flux'] -= center
            data['model'] -= center
            if self.nbin is not None:
                ax.set_title('White Light Curves of %s'%self.target, fontsize=32)
                binned_flux = self.binData(data['relative flux']+offset, self.nbin)
                binned_t = self.binData(data['t'], self.nbin)
                binned_error = self.binData(data['error'], self.nbin, err=True)
                ax.errorbar(binned_t, binned_flux, yerr=binned_error, ms=15 , fmt='.',
                                color='white',ecolor='deepskyblue',mec='deepskyblue')
                suffix = 'bin%s'%self.nbin
            else:
                ax.errorbar(data['t'], data['relative flux']+offset, yerr=data['error'], ms=15 , fmt='.',
                                color='white',ecolor='deepskyblue',mec='deepskyblue')
                suffix = 'full'

            ax.plot(data['t'], data['model']+offset, color='dimgray', zorder=10)
            ax.text(xpos,data['relative flux'].max()+offset, r'ID = %s, $t_0 = $%.6f'%(id,t0), fontsize=23)
        ax.set_ylabel('Relative Flux', fontsize=23)
        ax.set_xlabel('Time from Transit Midpoint [hrs]', fontsize=23)
        ax.set_xlim(np.min(data['t']), np.max(data['t']))
        ax.tick_params(direction='in', which='both', labelsize=18)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.tight_layout()
        if self.savePlots:
            plt.savefig(self.indir+'white_lightcurves_%s.png'%suffix, facecolor='white', dpi=300)
            plt.savefig(self.indir+'white_lightcurves_%s.pdf'%suffix, facecolor='white', dpi=600)
        plt.show()