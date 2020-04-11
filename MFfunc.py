import pandas as pd
import math
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statistics as stat
from mpl_toolkits.basemap import Basemap

# All functions for MODFLOW related data processing

def read_os(ifile='', ichar_hed = 'n'):
    with open(ifile) as f:
        #    c0 = 0
        hobs = []  # An empty list object
        hcal = []
        dobs = []  # An empty list object
        dcal = []
        gobs = []  # An empty list object
        gcal = []
        for line in f:
            data = line.split()
            char = data[2][0]
#            print(char)
            ncol = len(data)
            if (ncol <= 3 and char == ichar_hed):    # Read head data from the second line
                hobs.append(float(data[1]))
                hcal.append(float(data[0]))
            elif (ncol <= 3 and char == 'd'):    # Read drn data
                dobs.append(float(data[1]))
                dcal.append(float(data[0]))
            elif (ncol <= 3 and char == 'g'):    # Read GHB data
                gobs.append(float(data[1]))
                gcal.append(float(data[0]))
        df_hed = pd.DataFrame({'hobs': hobs, 'hcal': hcal})
        df_drn = pd.DataFrame({'dobs': dobs, 'dcal': dcal})
        df_ghb = pd.DataFrame({'gobs': gobs, 'gcal': gcal})

#        print(df.head)
        return df_hed, df_drn, df_ghb


def MapPlotPV(domain_coor='', plot_background=False, df='', ofile='', vmin=0, vmax=10, legend_name=''):
    xmin, xmax, ymin, ymax = domain_coor  # NV
    s_size = 12
    s_edgewidth = 0.1
    cmap = 'rainbow'

    # Create the figure and basemap object
    fig, ax = plt.subplots()
    #fig = plt.gcf()
    fig.set_size_inches(8, 6.5)
    #    m = Basemap(projection='robin', lon_0=0, resolution='c')
    m = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax,
                urcrnrlat=ymax, projection='npstere',
                resolution='l', lon_0=-115, lat_0=36, epsg=4269)
    #    m = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax, projection='robin',
    #                resolution='l', lon_0=-115, lat_0=36)

    #

    # m.bluemarble(scale=0.2)   # full scale will be overkill
    # Plot BACKGROUND =========================================================

    if plot_background == True:
        #        m.arcgisimage(service='ESRI_Imagery_World_2D',
        #                      xpixels=1500, ypixel=None, dpi=300, verbose=True, alpha=1)  # Background
        m.arcgisimage(service='World_Shaded_Relief',
                      xpixels=1500, dpi=300, verbose=True, alpha=0.25)  # Background

    # Watershed
    m.readshapefile('/scratch/hpham/shp/WBDHU12', name='NAME',
                    color='#bfbfbf', linewidth=0.25, drawbounds=True)

    # road layer
    m.readshapefile('/scratch/hpham/shp/tl_2016_us_primaryroads', name='NAME',
                    color='#a52a2a', linewidth=0.5, drawbounds=True)

    # Plot variance at well locations ==========================================
    #x, y = [df["Lon"].values, df["Lat"].values]
    z_value = df.std_hed
    #vmin = min(z_value)
    #vmax = max(z_value)

#    x, y = map(df["Lat"], df["Long"])
#    print(f'x={x}, y={y}')

#    plt1 = plt.scatter(x, y, edgecolors=s_color, color='k',
#                s=s_size, linewidth=s_width, alpha=1)

    x, y = m(df["Lon"].values, df["Lat"].values)  # transform coordinates

    plt1 = plt.scatter(x, y, c=z_value, vmin=vmin, vmax=vmax,
                       edgecolors='k', s=z_value*5,
                       linewidth=s_edgewidth, cmap=cmap, alpha=1)

    # Legend
    #fig.colorbar(plt1)
    #cbar_ax = fig.add_axes([0.1, 0.1, 0.25, 0.01])
    cbar_ax = fig.add_axes([0.2, 0.18, 0.25, 0.01])
    #cbar_ax = fig.add_axes([0.2, 0.08, 0.2, 0.01])
    cbar = fig.colorbar(plt1, orientation='horizontal',
                        extend='both', extendfrac='auto', cax=cbar_ax)
    cbar.set_label(legend_name, labelpad=0, y=0.08, rotation=0)


#    ax.set_title('One standard deviation of simulated heads')

    # m.drawcountries(linewidth=1.0)
    # m.drawcoastlines(linewidth=.5)

    #ofile = 'LS_' + 'TS' + str(ts).rjust(2, '0') + '.png'
    #ofile = 'map_stdv_of_cal_hed.png'
    fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
    # plt.show()

def plot_spring_flow(df2='',fig='', ax1='', ax2='', sim_color_line = '#bfbfbf'):
    #color = ['#bfbfbf']

    
    # Plot and save the figure
    #fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, sharey=False)
    #fig.set_size_inches(8, 3.6) 
    
    #kwargs = dict(ecolor='k', color='c', capsize=2,
    #        elinewidth=1, linewidth=0.6, ms=5, alpha=0.95)
    plt.rc('grid', linestyle="dotted", color='gray', alpha = 0.15)

    # Bennetts
    i1, i2 = [0, 13]
    #ax1.plot(df2.Date, -df2.Qcal_Bennetts,'b', label= 'Simulated')
    #ax1.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2], 'or', label= 'Observed')
    ax1.plot(df2.Date, -df2.Qcal_Bennetts,color=sim_color_line, linewidth=0.1)
    #ax1.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2], marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)
        
    
    # Manse
    #kwargs = dict(ecolor='k', color='c', capsize=2,
    #    elinewidth=1, linewidth=0.6, ms=5, alpha=0.95)
    i1, i2 = [13, 55]
    #ax2.plot(df2.Date, -df2.Qcal_Manse,'b', label= 'Simulated')
    #ax2.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2], 'or', label= 'Observed')
    ax2.plot(df2.Date, -df2.Qcal_Manse,color=sim_color_line, linewidth=0.1)
    #ax2.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2], marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)

    #fig.savefig('Spring_flow.png', dpi=200, transparent=False, bbox_inches='tight')
    #plt.show()



'''
    nobs=dfQ.shape[0]
    rmse_Q = math.sqrt(sum((dfQ.Qobs-dfQ.Qcal)*(dfQ.Qobs-dfQ.Qcal))/nobs)
    f_rmse_Q = 100*rmse_Q/(max(dfQ.Qobs)-min(dfQ.Qobs))

    # Interpolate to get flow at observation times only
    val = f_rmse_hed + f_rmse_Q
    #out = [rmse_hed rmse_Q]
    f1 = open('fitness.txt', 'w')
    f1.write("val rmse_hed rmse_Q \n")
    f1.write("%9.6f %9.6f %9.6f \n" % (val, rmse_hed, rmse_Q))
    f1.close()
'''


def read_multi_os_files(path_to_os_files='', nsamples='', npars='',  of1='', get_id_err = ''):
    print(f'\nReading os files using function read_multi_os_files')
    # Calculate RMSE
    run_id = []
    rmse = []
    run_id_err = []

    for i in range(nsamples):
        # Second run, lo/up range = +-80% opt par val
        ifile = path_to_os_files + 'out_' + str(i+1) + '._os'

        if i % 1 == 0:
            print(f'Reading os file {ifile}')
        if os.path.isfile(ifile):  # File ._os is available
            index_char_for_hed = 'n' # 'h' to get hed, 'n' to get no_hed
            if i == 0:
                print(f'Index used to get head in ._os is {index_char_for_hed}')
            df, df_drn, df_ghb = read_os(ifile=ifile, ichar_hed=index_char_for_hed)
            if i == 0:   # Only read one file to get nobs
                nobs = df.shape[0]
            rmse.append(
                math.sqrt(sum((df.hobs-df.hcal)*(df.hobs-df.hcal))/nobs))
            run_id.append(i+1)
        else:
            print('WARNING: File %s does not not exist' % (ifile))
            run_id_err.append(i+1)
        # new dataframe to save all heds
        if i == 0:
            # nobs = df.shape[0]
            new_name = 'r' + str(i+1)
            df2=df.rename({'hobs':'hobs', 'hcal':new_name}, axis=1)
        else:
            new_name = 'r' + str(i+1)
            df_new=df.rename({'hobs':'hobs', 'hcal':new_name}, axis=1)
            df_new=df_new[new_name] # get hcal            
            df2 = pd.concat([df2, df_new], axis=1)

    #    
    df = pd.DataFrame({'run_id': run_id, 'rmse': rmse})
    ifile = 'rmse_' + str(npars) + 'pars_' + str(nsamples) + 'samples.csv'
    print(f'{df.shape[0]} os files were read.')
    print(f'The output file is {ifile}, saved at the current directory.')
    df.to_csv(ifile,  index=None)
    df2.to_csv(of1,  index=None)
    print(f'The output file is {of1}, saved at the current directory.')


    

    
    if get_id_err:
        df_err = pd.DataFrame({'run_id': run_id_err})
        df_err.to_csv('err.csv')
        # print(df.head)
        # Get id of err runs
        df = pd.read_csv('BK_all_par_pertu.dat', header=None)
        df2 = df.loc[df_err.run_id-1]
        df2.to_csv('re_run.dat', header=None,  index=None)
    return df, df2

def process_mult_gwlevels(nts, nwells,nsamples,wel,std_hed,df_date,df_obs_wel,df_obs, df, \
                          o_std_file,odir,unit_,ifile_welid='', save_fig=''):

    # ts to export standard deviation
    #ts_for_stdv = [1247, 1847]
    #col_name = ['Name', 'Lat', 'Lon', 'y2016', 'y2066']
    #dfstd = pd.DataFrame(names=col_name)
    #std_hed_at_given_ts = []
    ts_to_plot_err = range(0,nts,60) 
    #kwargs = dict(fmt='o', color='r', markersize=3, capsize=2, linewidth=1, linestyle=None, zorder = 1000)
    lw = 0.25 # LineWidth
    stdv_arr = np.zeros(shape=(nts,nwells))
    #Lon = []
    #Lat = []
    for i in range(nwells):
        print(f'Well: {wel.NAME[i]}')
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4.5) 

        # Get id of well
        id_start = wel.Idstart[i]-1 
        id_stop = wel.Idstop[i]
        
        # Calculate standard deviation
        
        std_hed_curr_well = std_hed[id_start:id_stop] # 1848 ts x 1
        print(f'id_start={id_start}, stdv={std_hed_curr_well.iloc[0]}')
        #std_hed_curr_well.reset_index()
        stdv_arr[:,i] = std_hed_curr_well
        
        if save_fig:
            # Get date
            date_ = df_date # 1913-2066
            x_ = pd.concat([date_]*(nsamples+1), axis=1)
            # Get hcal of current well, all hed realizations
            hcal = df.loc[id_start:id_stop-1] # nobs x nrealizations
            # 
            
        
            # Plot h with optimal parameters
            hopt = df.r1[id_start:id_stop]
            

            # Plot hobs ===========================================================

            # Reading ID of each well    
            ifile_id_of_each_well_12327 = ifile_welid
            df_wel_id = pd.read_csv(ifile_id_of_each_well_12327)
            id_start_12327 = df_wel_id.Idstart[i]-1 
            id_stop_12327 = df_wel_id.Idstop[i]
            xobs = df_obs['Date'][id_start_12327:id_stop_12327]
            xobs = pd.to_datetime(xobs)
            yobs = df_obs['hobs'][id_start_12327:id_stop_12327]
            
            #ax2 = ax.twinx()
            #ax2.plot(xobs, yobs*3.28084, marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)
            #ax2.set_ylabel('Groundwater level (m) above NGVD 1929')
            if unit_=='US':
                plt.plot(x_, hcal*3.28084, color='#bfbfbf', linewidth=lw)
                plt.plot(date_, hopt*3.28084, 'r-')
                plt.plot(xobs, yobs*3.28084, marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)
                temp = hopt.iloc[ts_to_plot_err]*3.28084
                #ax.errorbar(date_.iloc[ts_to_plot_err], temp.values, std_hed_curr_well.iloc[ts_to_plot_err]*3.28084, **kwargs)
                
                plt.ylabel('Groundwater level (ft) above NGVD 1929')
                ax.set_ylim([2300, 2660])
            else:
                plt.plot(x_, hcal, color='#bfbfbf', linewidth=lw)
                plt.plot(date_, hopt, 'r-')
                plt.plot(xobs, yobs, marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)
                plt.ylabel('Groundwater level (m) above NGVD 1929')


            
            #ymin, ymax = ax.get_ylim()
            #ax.set_yticks(np.round(np.linspace(ymin, ymax, 5), 2))
            ax.set_title('Well: ' + wel.NAME[i])


            #plt.xlabel('Time')
            
            # plt.grid(True)
            ofile = str(i+1) + '_hed_' + wel.NAME[i] + '.png'
            fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')




    # Create a dataframe for stdv
    col_names = []
    for i in range(nts):
        col_names.append('ts' + str(i+1))
    df_stdv = pd.DataFrame(data=stdv_arr.T, columns=col_names)

    # Create the final dataframe
    df_std_at_obs_well = pd.concat([df_obs_wel, df_stdv], axis=1)
    df_std_at_obs_well.to_csv(o_std_file, index=None)
    print(f'Stdv of simulated heads was saved, filename: {df_std_at_obs_well}')

    # move all png files to odir
    cmd = 'mv ' + '*.png ' + odir
    subprocess.call(cmd, shell=True)        