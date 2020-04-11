#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


#plot_his = True


def plot_his(ifile, pname, nbins):
    #ifile = 'output12/par_pertu_1.txt'
    #    ifile = 'out_rmse/out_rmse_out_Q_pert.txt'

    data = np.loadtxt(ifile, delimiter=',')
    npar = data.shape[1]
    for par in range(npar):

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6.5)
        x = data[:, par]
        mu = np.mean(x)
        #print(f'Mu = {str(mu)}')
        sigma = np.std(x)
        #mu, sigma = 100, 15
        #x = mu + sigma*np.random.randn(10000)

        # the histogram of the data
        n, bins, patches = plt.hist(
            x, nbins, normed=1, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        y = mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=1)

        plt.xlabel(f'RMSE (m)')
        plt.ylabel('Probability')

        #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=mu,\ \sigma=sigma$')
        plt.title(f'Parameter: {pname[par]}')
        #plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        ofile = 'fig_sens_' + pname[par] + '_.png'
        fig.savefig(ofile, dpi=150, transparent=False, bbox_inches='tight')
        print(f'Figure {ofile} was saved.')

        # plt.show()
    return fig
