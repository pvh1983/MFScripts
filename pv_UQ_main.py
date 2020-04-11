import pandas as pd
import math
import os
import datetime as dt
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from MFfunc import *
import statistics as stat
#from func_pv import *
import flopy.utils.binaryfile as bf
from pathlib import Path
import matplotlib.pylab as pl


# Update 08072019
# Last visit: 08312019

# Use this function to:
# [1] Read multiple os files from MCMC simulations
# [2] 

#
x_min, x_max, y_min, y_max = [-116.215, -115.567, 35.861, 36.47]  # NV
domain_coor = [x_min, x_max, y_min, y_max]

#
gwmodel = 'tr'   # Choose: ss or tr
nsamples = 500  # ss:1000, tr: 500
nsamples_to_plot = 500
nts = 1848 # 154x12 # prediction model
# nos_files = 12500


# [-] ==========================================================================
#[00]
unit_ = 'US' # 'US' ft CFA or leave empty for metric


# [11] Read os files, cal rmse, and save all ==================================
read_multi_os_files_and_cal_rmse = False # In use
# path to multiple os file named as out_1._os, out_2._os ...
path_to_os_files = 'all_os_files/' 
#iofile_all_hed_mcmc = 'all_head_old_os.csv'  # use this to get 12327 obs [run once]
iofile_all_hed_mcmc = 'all_head.csv'  # 500+1 cols, 12327 rows

#
# [12] Load several inputs files ==============================================
#      Run read_multi_os_files_and_cal_rmse FIRST 
# This create some folder and define some input files
load_well_data = True # In use
### [input file 1] start/stop IDs of each well. 'n' as keyword
ifile_id_of_each_well = 'obswell_locations_no_hed.csv'
ifile_id_of_each_well_12327 = 'obswell_locations_12327.csv' # id of obs 1913-2016
### [input file 2] date for head measurements
ifile_obs_date = 'date_for_hob_12327.csv'
### [input file 3] HObs include date (12327 obs)
ifile_Hobs = 'hobs_Pahrump.csv'
### [input file 4] meta 134 obs well [name lat lon idstar idstop ...]
ifile_obs_well_meta = 'meta_134_obs_well.csv'
# NExt step is 21


# [21] Use key is 'no_hed' for ._os file, for predictive model only
#      currently using this for UQ of the predictive model
#     hob file = 134 x 2 wells (134 for pred, 134 for mcali)
#     must use ifile_id_of_each_well=obswell_locations_no_hed.csv for 
#     This also calculate stdv of simulated heads at all wells
#     Make sure load_well_data = True and read_all_hed_data=True
plot_all_GWlevel_plots_no_hed = True  # In use - for 1913-2066 pred model
plot_and_save_fig = True   # False if only need stdv (to save time)
read_all_hed_data = True   # False if run other to save time

# [3] Calculate variances of simulated heads
#     is done in plot_all_GWlevel_plots_no_hed
o_std_file = 'std_hed_at_all_obs_wells.csv'

# [4] Plot variance
plot_variance = False # In use

# [5] Read os and ccf files to get hed and Q 
Bennetts_and_Manse = False # In use
# 1913-2066 model
mname_prefix = 'TR_BLM_hpham5_pred'
path_to_file = '/scratch/hpham/cmaes/pv_UQ_hpham22_HQ2/model_final'
path_to_ts   = '/scratch/hpham/cmaes/pv_UQ_hpham22_HQ2/list_of_ts.csv'
#n_stress_period = 104  # mcali, each is one years, divided into 12 months
n_stress_period = 154  # mcali + pred 
n_ts_in_one_stress_period = 12
return_coeficient = 0.3

# [6] Plot sping flow and uncertainty
plot_spring_flow_UQ = False       # 
path_to_spring_flow_files='/scratch/hpham/cmaes/pv_UQ_hpham22_HQ2/all_spring_flow_files/'

# [7] Plot southwest flow and uncertainty
plot_southwest_outflow_UQ = False   # 
path_to_budget_files='/scratch/hpham/cmaes/pv_UQ_hpham22_HQ2/all_wbud_files/'



# [.] Plot all hydrographs (hobs, hopt, h_mcmc) - OLD NOT IN USING
#     for predictive 1913-2066 model ONLY.
#     key char when reading ._os is 'h' for no_head
#     must use file obswell_locations_hed.csv?
#     and read_all_hed_data=True
#     hob file = 134 x 2 wells (134 for pred, 134 for mcali)
plot_all_GWlevel_plots = False # Not in use - Use for pred 1913-2066 model
if plot_all_GWlevel_plots:
    ifile_id_of_each_well = 'obswell_locations_hed.csv'


# 
# [13] Combine simulated data and predicted data OLD option, not use?
#      and generate plots (years from 1913-2016?)
#      must use file obswell_locations.csv 
#      12327 obs (org hob file)
combine_hed_data = False  # Not in use - Use for mcali 1913-2016 model?








# [6] Read *.hed files using Flopy
read_hed_using_flopy = False


plot_ccf_output = False

# [Step 31] NO longer used, use read_multi_os_files_and_cal_rmse instead.
tr_model_susie = False




plot_fitness_base_par = False  # Plot for baseline scenarios
plot_fitness_per_pars = False  # Plot for pertubative scenarios

generate_oc_file = False
read_out_file = False

#
read_os_and_save_hed = False    # read raw *._os of model and save hcal as df
                                # Specify nosb_start and nobs_stop
read_one_os_file = False # Not available here, see func_mf.py


# [Step 3x] Generate rmse_tr_npars.csv. Next, step 12.
combine_csv_df_files = False

read_out_of_tr_runs = False

plot_errorbar = False   # Plot error bar. Run R code to get delsa_q.csv



# [Step 00] choose 'out_HK' or 'out_other_pars' or 'out_all_pars'
out_par = 'output_UQ'

#
if os.path.isdir(out_par):   # Folder exists
    pass
else:
    cmd = 'mkdir ' + out_par
    subprocess.call(cmd, shell=True)

# ???
#pname = ["HK_2", "HK_4", "HK_3", "HK_6", "HK_5", "HK_7", "HK_8", "HK_1", "HK_9", "HK_10", "HK_60", "GHB_14", "GHB_13",
#            "GHB_15", "HFB_16", "HFB_22", "HFB_18", "HFB_24", "HFB_21", "HFB_17", "HFB_19", "HFB_20", "HFB_23", "VANI_200"]

# HP's hpham22 model, with HnQ2, par_range.csv, last updated 09042019
pname = ["EVT_701","EVT_702","EVT_703","GHB_501","GHB_502","GHB_503",
            "HK_101","HK_102","HK_103","HK_104","HK_105","HK_106","HK_107",
            "HK_108","HK_109","HK_110","HK_111","HK_112","HK_113","HK_114",
            "HK_115","HK_116","SY_301","SY_302","SY_303","SY_304","SY_305",
            "SY_306","SY_307","SY_308","SY_309","SY_310","SY_311","SS_201"]

npars = len(pname)

# Load data ===================================================================
cwd = os.getcwd()


if load_well_data:
    odir = cwd + '/outpng_all_hydrographs/'
    if not os.path.exists(odir):  # Make a new directory if not exist
        os.makedirs(odir)
        print(f'Folder {odir} was created.')

    # Date for head observations
    
    hed_obs_date = pd.read_csv(ifile_obs_date)
    print(f'Reading obs date from file {ifile_id_of_each_well}')
    hed_obs_date['Date'] = pd.to_datetime(hed_obs_date['Date'])

    # load id of each well    
    wel = pd.read_csv(ifile_id_of_each_well) 
    print(f'Reading indices of head measurements for each well from file {ifile_id_of_each_well}')
    nwells = wel.shape[0]

    # load name lat lon id ... of 134 obs wells
    df_obs_wel = pd.read_csv(ifile_obs_well_meta) 
    print(f'Reading obswell meta file {ifile_obs_well_meta}')

    # Reading Hobs values with Date include         
    df_obs = pd.read_csv(ifile_Hobs)
    print(f'Reading Hobs from file {ifile_Hobs}')

    # read all head data
    
    if read_all_hed_data:
        print(f'Reading all Hcal (500 rlz) from file {iofile_all_hed_mcmc}')
        df = pd.read_csv(iofile_all_hed_mcmc)
        hobs = df.hobs
        df = df.drop(['hobs'], axis=1)
        df_hobs = pd.concat([hed_obs_date, hobs], axis = 1)
        #df_hobs.to_csv('hobs_Pahrump.csv', index = None)

        std_hed = df.std(axis=1) # at each measurements
        flag_data_load = 1
    
  



#
# Create a dataframe for monthly data
start_date = dt.date(1913, 1, 1)
end_date = dt.date(2066, 12, 1)
df_dt = pd.DataFrame({'Date': [start_date, end_date], 'Val': [999, 999]})
df_dt['Date'] = pd.to_datetime(df_dt['Date'])
df_date = df_dt.resample('M', on='Date').mean()   # resample by month
df_date_yr = df_dt.resample('A', on='Date').mean()   # resample by year
df_date = df_date.reset_index()
df_date = df_date.drop('Val', 1)
df_date_yr = df_date_yr.reset_index()
df_date_yr = df_date_yr.drop('Val', 1)


#
# [Step 01] Read os and cal rmse of steady state model
if read_multi_os_files_and_cal_rmse:
    print(f'\n Read multi os files, calculate rmse and save to csv file.')
    df_, df2_ = read_multi_os_files(path_to_os_files=path_to_os_files, \
    nsamples=nsamples, npars=npars, of1=iofile_all_hed_mcmc, get_id_err = False)


if plot_all_GWlevel_plots:

    #   
    for i in range(nwells):
        print(f'Well: {wel.NAME[i]}')
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4.5) 

        # Plot all hcal
        id_start = wel.Idstart[i]-1 
        id_stop = wel.Idstop[i]
        date_ = hed_obs_date.Date[id_start:id_stop]
        x_ = pd.concat([date_]*(nsamples+1), axis=1)
        hcal = df.loc[id_start:id_stop-1]
        # 
        plt.plot(x_, hcal, color='#bfbfbf')
    
        # Plot h with optimal parameters
        hopt = df.r1[id_start:id_stop]
        plt.plot(date_, hopt, 'r-')

        # Plot hobs         
        plt.plot(date_, hobs[id_start:id_stop], marker='o', linestyle='dotted',
                markerfacecolor='none', markersize=6)

    #    ymin, ymax = ax.get_ylim()
    #    ax.set_yticks(np.round(np.linspace(ymin, ymax, 5), 2))
        ax.set_title('Well: ' + wel.NAME[i])


    #    plt.xlabel('Time')
        plt.ylabel('Groundwater level (m) above NGVD 1929')
        # plt.grid(True)
        
        ofile = str(i+1) + '_hed_' + wel.NAME[i] + '.png'
        fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')

        # plt.show()
    cmd = 'mv ' + '*.png ' + odir
    subprocess.call(cmd, shell=True)




if plot_variance:   
    # Read std values from file
    print(f'Reading file {o_std_file} to get stdv of simulated heads')
    df = pd.read_csv(o_std_file)
    ts_for_plotting_stdv = [1247, 1847] # start id from 0
    vmin = 0
    vmax = 10
    lename = '1-stdv. of sim. heads (m)'
    for i in ts_for_plotting_stdv:
        cname = 'ts' + str(i)
        ofile_curr_ts = 'map_stdv_of_cal_hed_' + cname + '_meter.png'
        if unit_=='US':
            df[cname] = df[cname]*3.28084
            vmax = 30
            lename = '1-stdv. of sim. heads (ft)'
            ofile_curr_ts = 'map_stdv_of_cal_hed_' + cname + '_ft.png'

        col_keep = ['Name', 'Lon', 'Lat', cname]
        df_for_a_ts=df[col_keep]
        df_for_a_ts_renamed = df_for_a_ts.rename({cname:'std_hed'}, axis=1)
        # call MapPlotPV func to plot
        
        MapPlotPV(domain_coor=domain_coor, plot_background=True, df=df_for_a_ts_renamed, 
        ofile=ofile_curr_ts, vmin=vmin, vmax=vmax, legend_name = lename)



if Bennetts_and_Manse:

    #mname_prefix = 'TR_BLM_update_FINAL_FINAL_2016'
    #mname_prefix = 'TR_BLM_hpham5'
    #mname_prefix = 'TR_BLM_update_FINAL_FINAL_2016_HKparams'
    
    # 1913-2016 model
    #mname_prefix = 'TR_BLM_update_FINAL_FINAL_2016_HKparams'
    #path_to_file = '/scratch/hpham/cmaes/pv_cma_tr_susie_HQ2/best_susie'
    #path_to_ts = '/scratch/hpham/cmaes/pv_cma_tr_susie_HQ2/list_of_ts.csv'
    
    # 1913-2066 model susie's model
    #mname_prefix = 'TR_BLM_update_FINAL_FINAL_2066_HKparams'
    #path_to_file = '/scratch/hpham/cmaes/pv_UQ_tr_HnQ_NEW2/model_final'
    #path_to_ts = '/scratch/hpham/cmaes/pv_UQ_tr_HnQ_NEW2/list_of_ts.csv'
    
    print(f'Path to ts file  {path_to_ts}')
    

    # Read ._os file and calculate rmse between sim. vs. obs. heads
    id_err = list(range(1521-1,1604)) + list(range(2818-1,2884)) + \
             list(range(4124-1,4244)) + list(range(7108-1,7127)) + \
             list(range(2735-1,2817)) + list(range(3055-1,3136)) + \
             list(range(3137-1,3212)) + list(range(5141-1,5463)) + \
             list(range(7108-1,7127))     # python array from 0, not 1. 

    # Several pumping test points
    id_err.append(831-1)     # FID_10, Great Basin
    id_err.append(2885-1)     # add 2884 to the err list; FID_39: North Leslie
    id_err.append(4043-1)     # FID_53: Stump Spring Well
    id_err.append(7719-1)     # FID_110: 162  S20 E53 06CB  1
    id_err.append(8762-1)     # err = -47 m; FID_118: 162  S19 E53 28AA  1
    id_err.append(8956-1)     # err = ~7 m; FID_121: 162  S19 E53 10CB  1


    ifile = path_to_file + '/' + mname_prefix + '._os'
    print(f'Reading file {ifile}')
    with open(ifile) as f:
        #    c0 = 0
        hobs = []  # An empty list object
        hcal = []
        for line in f:
            data = line.split()
            char = data[2][0]
    #            print(char)
            ncol = len(data)
            if (ncol <= 3 and char == 'h'):    # Read head data from the second line
                hobs.append(float(data[1]))
                hcal.append(float(data[0]))
        for i in sorted(id_err, reverse=True):
            del hobs[i]
            del hcal[i]
        nobs = len(hobs)
        #print(nobs)
        df_hed = pd.DataFrame({'hobs': hobs, 'hcal': hcal})
        rmse_hed = math.sqrt(sum((df_hed.hobs-df_hed.hcal)*(df_hed.hobs-df_hed.hcal))/nobs)
        f_rmse_hed = 100*rmse_hed/(max(hobs)-min(hobs))
        #print(f_rmse_hed)

    #
    #
    # Read ccf file and calculate spring flow
    # Read ccf file of a TRANSIENT model to get budget at given cells
    # Use python2, read transient ccf files




    ifile = path_to_file + '/' + mname_prefix + '.ccf'
    print(f'Reading file {ifile}')
    cbb = bf.CellBudgetFile(ifile)
    #list_unique_records = cbb.list_unique_records()  # Print all RECORDS
    #ncols = 3  # make sure you change this
    #data_out = np.empty([n_stress_period, ncols])

    # Load data
    list_of_spring_cells = 'list_of_drt_cells.csv'
    drt_cell = np.loadtxt(list_of_spring_cells,
                            delimiter=',', dtype=int)  # use "" for python2
    n_drt_cells = len(drt_cell)
    drt_cell = drt_cell - 1  # IMPORTANT -  Python index is from 0
    Q_Bennetts = []
    Q_Manse =[]
    fid=open('flow_at_cells.csv', 'w') # To write output
    for ts in range(n_stress_period):
        for month in range(n_ts_in_one_stress_period):
            DRT = cbb.get_data(kstpkper=(month, ts), text='DRAINS (DRT)', full3D=True)[0]
            DRT = np.where(DRT.mask, 0, DRT)

            #
            Q_springs = []
            for k in range(n_drt_cells):  # Cell k
                # print(k)
                # print(DRT.sum())
                    Q_springs.append(
                        DRT[drt_cell[k, 0], drt_cell[k, 1], drt_cell[k, 2]])
            fid.write(f'{ts+1}, {month+1}, {Q_springs} \n')
            Q_Bennetts.append(sum(Q_springs[0:2])/(1-return_coeficient)) # Two first cells are for Bennetts
            Q_Manse.append(sum(Q_springs[2:5])/(1-return_coeficient))
    fid.close()
    # Load time step
    df1 = pd.read_csv(path_to_ts)
    df1.Date = pd.to_datetime(df1.Date)
    # Simulated flow df2 are:
    df2 = pd.DataFrame({'Date':df1.Date, 'Qcal_Bennetts':Q_Bennetts, 'Qcal_Manse':Q_Manse})
    df2.to_csv('simulated_flow.csv')
    #print(df2)

    # Load observed spring flow
    df = pd.read_csv('spring_flow_rate.csv')
    df.Date = pd.to_datetime(df.Date)
    #print(df)

    df_Bennetts = df.iloc[:13] # Observed flow at Bennetts
    df_Manse = df.iloc[13:]    # Observerd flows at Manse
    #print(df_Manse)
    df_Bennetts_New = pd.merge(df2, df_Bennetts, how='inner', on=['Date'])
    df_Bennetts_New =  df_Bennetts_New.drop(columns='Qcal_Manse')
    df_Bennetts_New = df_Bennetts_New.rename(index=str, columns={"Qcal_Bennetts":"Qcal"})


    df_Manse_New = pd.merge(df2, df_Manse, how='inner', on=['Date'])
    df_Manse_New = df_Manse_New.head=df_Manse_New.drop(columns='Qcal_Bennetts')
    df_Manse_New = df_Manse_New.rename(index=str, columns={"Qcal_Manse":"Qcal"})


    dfQ = pd.concat([df_Bennetts_New, df_Manse_New], sort=False)
    dfQ.to_csv('spring_flow.csv', index=None)

    #
    # Plot and save the figure
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, sharey=False)
    fig.set_size_inches(8, 3.6) 
    kwargs = dict(ecolor='k', color='c', capsize=2,
            elinewidth=1, linewidth=0.6, ms=5, alpha=0.95)
    plt.rc('grid', linestyle="dotted", color='gray', alpha = 0.15)

    # Bennetts
    i1, i2 = [0, 13]
    ax1.plot(df2.Date, -df2.Qcal_Bennetts,'b', label= 'Simulated')
    ax1.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2], 'or', label= 'Observed')
    ax1.set_ylabel('Discharge (m$^3$/day)')
    #ax1.legend('Cal', 'Obs')
    ax1.legend(loc='upper right', borderaxespad=0.5)
    ax1.set_ylim([0, 25000])
    ax1.set_title('(a) Bennetts Spring')
    ax1.grid(True)
    
    # Manse
    kwargs = dict(ecolor='k', color='c', capsize=2,
        elinewidth=1, linewidth=0.6, ms=5, alpha=0.95)
    i1, i2 = [13, 55]
    ax2.plot(df2.Date, -df2.Qcal_Manse,'b', label= 'Simulated')
    ax2.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2], 'or', label= 'Observed')
    #ax2.set_ylabel('Discharge (m$^3$/day)')
    ax2.legend(loc='upper right', borderaxespad=0.5)
    ax2.set_ylim([0, 25000])
    ax2.set_title('(b) Manse Spring')
    ax2.grid(True)

    fig.savefig('Spring_flow.png', dpi=200, transparent=False, bbox_inches='tight')
    #plt.show()




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

if read_hed_using_flopy:

    # Setup contour parameters   
    Lx = 68800.
    Ly = 77200.
    nlay = 5
    nrow = 193
    ncol = 172
    levels = np.linspace(0, 10, 11)
    delr = Lx / ncol    
    delc = Ly / nrow
    extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)
    print('Levels: ', levels)
    print('Extent: ', extent)
    
    odir = 'out_head_csv_files'
    if not os.path.exists(odir):  # Make a new directory if not exist
        os.makedirs(odir)
        print(f'Directory {odir} was created.')


    print(f'read_hed_using_flopy')   
    # Load time step for the prediction model
    # Date for head observations
    ifile_pred_ts = 'list_of_ts.csv'
    hed_pred_date = pd.read_csv(ifile_pred_ts)
    print(f'Reading time steps of the prediction model, file {ifile_pred_ts}')
    hed_pred_date['Date'] = pd.to_datetime(hed_pred_date['Date'])

    # Get a list of obs well locations
    df=pd.read_csv('MFid_of_obs_wells.csv')
    df2 = df[['Lay', 'Row', 'Col' ]]
    idx = [tuple(x) for x in df2.values]

    #idx = [(0, int(nrow/2) - 1, int(ncol/2) - 1), (1, int(nrow/2) - 1, int(ncol/2) - 1)]
    for r in range(nsamples):
        ifile_hed = 'all_hed_files/out_' + str(r+1) + '.hed'
        print(f'Reading file {ifile_hed}.')       
        hdobj = bf.HeadFile(ifile_hed, precision='single')
    #    hdobj.list_records()
    #    rec = hdobj.get_data(kstpkper=(1, 50))
    #    times = hdobj.get_times()    
    #    hdobj.plot(totim=times[-1])

        #Export model output data to a shapefile at a specific location
    #    hdobj.to_shapefile('test_heads_sp6.shp', totim=times[-1])
        ofile = odir + '/hed_pred_at_wells_rlz_' + str(r+1) + '.csv'
        #df = pd.DataFrame({'Time':hed_pred_date, ''})
        ts = hdobj.get_ts(idx)
        np.savetxt(ofile, ts, delimiter=",")


    # Plot the head versus time
    plot_hed = False
    if plot_hed:       
        
        
        plt.subplot(1, 1, 1)
        ttl = 'Head at cell ({0},{1},{2})'.format(idx[0] + 1, idx[1] + 1, idx[2] + 1)
        plt.title(ttl)
        plt.xlabel('Time')
        plt.ylabel('Head (m)')
        #plt.plot(ts[:, 0], ts[:, 1], 'bo-')
        plt.plot(hed_pred_date['Date'], ts[:, 1], color='#bfbfbf')
        #plt.savefig('tutorial2-ts.png')
        #plt.show()
    

if combine_hed_data:
    # data is from df (by loading all_head.csv) nobs x nrealizations
    
    nwells = 134 # 
    for i in range(nwells):
        # Get data for each well
        print(f'Well: {wel.NAME[i]}')
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4.5) 


        # Get predictive data
        id_start_pred = wel.Idstart[i]-1 
        id_stop_pred = wel.Idstop[i]
        hcal_pred = df.loc[id_start_pred:id_stop_pred-1]
        
        # Get simulated data 1/1/1913 - 12/1/2016
        j = i + nwells
        id_start_mcali = wel.Idstart[j]-1 
        id_stop_mcali = wel.Idstop[j]
        hcal_mcali = df.loc[id_start_mcali:id_stop_mcali-1]
        
        # Merge two dataframes
        hcal = pd.concat([hcal_mcali, hcal_pred], axis=0)
        hcal.reset_index()
        
        
        #hcal_rsm = hcal.resample('M', on='Date').mean()   # resample by month
        #df3 = df2.resample('M', on='lev_hed_pred_date').mean()   # resample by month

        date_pred = hed_pred_date.Date[id_start_pred:id_stop_pred]
        date_mcali = hed_pred_date.Date[id_start_mcali:id_stop_mcali]
        date_ = pd.concat([date_mcali, date_pred], axis=0)
        date_.reset_index()      
        x_ = pd.concat([date_]*(nsamples+1), axis=1)

        #hcal_with_date = pd.concat([x_, ])
        
        # Plot all hcal 
        plt.plot(x_, hcal, color='#bfbfbf')
        
        ax.set_title('Well: ' + wel.NAME[i])


        # Plot simulated h with optimal parameters 1913-2016
        hopt_mcali = df.r1[id_start_mcali:id_stop_mcali]
        hopt_pred = df.r1[id_start_pred:id_stop_pred]
        hopt = pd.concat([hopt_mcali, hopt_pred], axis=0)
        hopt.reset_index()
        plt.plot(date_, hopt, 'r-')

        # Plot hobs 1913-2016
        plt.plot(date_mcali, hobs[id_start_mcali:id_stop_mcali], marker='o', linestyle='dotted',
                markerfacecolor='none', markersize=6)

    #    plt.xlabel('Time')
        plt.ylabel('Groundwater level (m) above NGVD 1929')
        # plt.grid(True)
        
        ofile = str(i+1) + '_hed_' + wel.NAME[i] + '.png'
        fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
    cmd = 'mv ' + '*.png ' + odir
    subprocess.call(cmd, shell=True)

if plot_all_GWlevel_plots_no_hed:
    print(f'\nRunning plot_all_GWlevel_plots_no_hed')
    process_mult_gwlevels(nts, nwells,nsamples,wel,std_hed,df_date,df_obs_wel,df_obs, df, \
    o_std_file,odir,unit_, ifile_welid=ifile_id_of_each_well_12327, save_fig=plot_and_save_fig)


if plot_spring_flow_UQ:

    fig_, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, sharey=False)
    fig_.set_size_inches(9, 9) 
    plt.rc('grid', linestyle="dotted", color='gray', alpha = 0.15)  
    lwidth = 0.35
    count = 0
    cname_B=[]
    cname_M=[]
    df500=pd.DataFrame()
    for i in range(nsamples):
        ifile = path_to_spring_flow_files +  'out_sprflow_' + str(i+1) + '.csv'
        myfile = Path(ifile)
        if myfile.is_file():
            df2 = pd.read_csv(ifile)
            df2['Date'] = pd.to_datetime(df2['Date'])
#            ax1.plot(df2.Date, -df2.Qcal_Bennetts,color='#bfbfbf', linewidth=lwidth)
#            ax2.plot(df2.Date, -df2.Qcal_Manse,color='#bfbfbf', linewidth=lwidth)
            #
            count+=1
            cB = 'B' + str(i+1)
            cM = 'M' + str(i+1)
            cname_B.append(cB)
            cname_M.append(cM)
            #cname_all.append(cname)
            data = pd.read_csv(ifile)
            df_curr_run = data[['Qcal_Bennetts', 'Qcal_Manse']]
            df_curr_run_renamed = pd.DataFrame({cB:df_curr_run['Qcal_Bennetts'], cM:df_curr_run['Qcal_Manse']})
            df500 = pd.concat([df500, df_curr_run_renamed], axis=1)
    
    df500B = df500[cname_B]
    df500M = df500[cname_M]
    stdv_QB = df500B.std(axis=1)  # stdv Q Bennetts
    stdv_QM = df500M.std(axis=1)  # stdv Q Manse
    # Plot Q MCMC 500 lines
    x_ = pd.concat([df_date]*(count), axis=1)    



    # Plot Qopt 
    ifile = path_to_spring_flow_files +  'out_sprflow_1.csv'
    df2 = pd.read_csv(ifile)
    df2['Date'] = pd.to_datetime(df2['Date'])

  
   
    
    # Save data
    df_stdv_final = pd.concat([df_date, -df2.Qcal_Bennetts, stdv_QB, -df2.Qcal_Manse, stdv_QM], axis=1)
    df_stdv_final.to_csv('stdv_of_springs.csv')


    # Load observed spring flow
    df = pd.read_csv('spring_flow_rate.csv')
    df.Date = pd.to_datetime(df.Date)
    #print(df)

    df_Bennetts = df.iloc[:13] # Observed flow at Bennetts
    df_Manse = df.iloc[13:]    # Observerd flows at Manse
    #print(df_Manse)
    df_Bennetts_New = pd.merge(df2, df_Bennetts, how='inner', on=['Date'])
    df_Bennetts_New =  df_Bennetts_New.drop(columns='Qcal_Manse')
    df_Bennetts_New = df_Bennetts_New.rename(index=str, columns={"Qcal_Bennetts":"Qcal"})


    df_Manse_New = pd.merge(df2, df_Manse, how='inner', on=['Date'])
    df_Manse_New = df_Manse_New.head=df_Manse_New.drop(columns='Qcal_Bennetts')
    df_Manse_New = df_Manse_New.rename(index=str, columns={"Qcal_Manse":"Qcal"})


    dfQ = pd.concat([df_Bennetts_New, df_Manse_New], sort=False)
    dfQ.to_csv('spring_flow.csv', index=None)


    # Plot stdv
    ts_to_plot_err = range(0,nts,60)
    #kwargs = [fmt='o', markersize=8, capsize=20]
    kwargs = dict(fmt='o', color='r', markersize=3, capsize=2, linewidth=1, linestyle=None, zorder = 1000)


    conv_coef = 0.295911 # 0.295911 AFY = 1 m3/day
    if unit_=='US':
        #QMCMC
        ax1.plot(x_, -df500B*conv_coef, color='#bfbfbf', linewidth=lwidth)
        ax2.plot(x_, -df500M*conv_coef, color='#bfbfbf', linewidth=lwidth)
        # Qopt
        ax1.plot(df2.Date, -df2.Qcal_Bennetts*conv_coef,color='r', linewidth=1)
        ax2.plot(df2.Date, -df2.Qcal_Manse*conv_coef,color='r', linewidth=1)
        # Q obs
        i1, i2 = [0, 13]
        ax1.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2]*conv_coef, marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)
        i1, i2 = [13, 55]
        ax2.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2]*conv_coef, marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)
        # stdv
        ax1.errorbar(df2.Date.iloc[ts_to_plot_err], -df2.Qcal_Bennetts.iloc[ts_to_plot_err]*conv_coef, stdv_QB.iloc[ts_to_plot_err]*conv_coef, **kwargs)
        ax2.errorbar(df2.Date.iloc[ts_to_plot_err],    -df2.Qcal_Manse.iloc[ts_to_plot_err]*conv_coef, stdv_QM.iloc[ts_to_plot_err]*conv_coef, **kwargs)
        ax1.set_ylabel('Discharge (AFA)')
        ax2.set_ylabel('Discharge (AFA)')
        ymax = 10000
        ofile = 'Spring_flow_AFA.png' # Cubic ft anually
    else:
        #QMCMC
        ax1.plot(x_, -df500B, color='#bfbfbf', linewidth=lwidth)
        ax2.plot(x_, -df500M, color='#bfbfbf', linewidth=lwidth)
        # Qopt
        ax1.plot(df2.Date, -df2.Qcal_Bennetts,color='r', linewidth=1)
        ax2.plot(df2.Date, -df2.Qcal_Manse,color='r', linewidth=1)
        # Q obs
        i1, i2 = [0, 13]
        ax1.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2], marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)
        i1, i2 = [13, 55]
        ax2.plot(dfQ.Date[i1:i2], -dfQ.Qobs[i1:i2], marker='o', linestyle='dotted', markerfacecolor='none', markersize=6)
        # stdv
        ax1.errorbar(df2.Date.iloc[ts_to_plot_err], -df2.Qcal_Bennetts.iloc[ts_to_plot_err], stdv_QB.iloc[ts_to_plot_err], **kwargs)
        ax2.errorbar(df2.Date.iloc[ts_to_plot_err],    -df2.Qcal_Manse.iloc[ts_to_plot_err], stdv_QM.iloc[ts_to_plot_err], **kwargs)
        ax1.set_ylabel('Discharge (m$^3$/day)')
        ax2.set_ylabel('Discharge (m$^3$/day)')
        ymax = 35000
        ofile = 'Spring_flow_m3perday.png' # m3/day
        


    
    #ax1.legend('Cal', 'Obs')
    #ax1.legend(loc='upper right', borderaxespad=0.5)

    ax1.set_ylim([0, ymax])
    #ax1.set_title('(a) Bennetts Spring')
    ax1.set_title('(a)')
    ax1.grid(True)

    
    #ax2.legend(loc='upper right', borderaxespad=0.5)
    ax2.set_ylim([0, ymax])
    #ax2.set_title('(b) Manse Spring')
    ax2.set_title('(b)')
    ax2.grid(True)

    
    fig_.savefig(ofile, dpi=200, transparent=False, bbox_inches='tight')
    #plt.show()

if plot_southwest_outflow_UQ:
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4.5) 

    df500 = pd.DataFrame()
    cname_all = []
    count = 0
    for i in range(nsamples):
        ifile = path_to_budget_files + 'out_wbud_' + str(i+1) + '.csv'
        myfile = Path(ifile)
        #print(f'Reading file {ifile}')
        if myfile.is_file():
            count+=1
            cname = 'r' + str(i+1)
            cname_all.append(cname)
            data = pd.read_csv(ifile)
            df_curr_run = data.GHB_OUT
            df_curr_run_renamed = pd.DataFrame({cname:df_curr_run})
            df500 = pd.concat([df500, df_curr_run_renamed], axis=1)
    print(f'Nfiles read is {count}')
    
    #df = pd.concat([df_date_yr, df500], axis=1)
    x_GHB = pd.concat([df_date_yr]*(count), axis=1)
    y_GHB = df500
    std_GHB_all_ts = df500.std(axis=1)
    mean_std_GHB_over_ts = std_GHB_all_ts.mean()
    print(f'1-stdv of simulated GHB over 154 years was {mean_std_GHB_over_ts}')
    colors = pl.cm.Spectral(np.linspace(0,1,count))
    #for i in range(count):
    #    #plt.plot(df_date_yr, df500[cname_all[i]], color=colors[i], linewidth=0.35)
    #    plt.plot(df_date_yr, df500[cname_all[i]], color=colors[i], linewidth=0.35)
    
    
    # PLOT GHB by opt par
    
    
    print(f'Mean of southwest outflow over 154 years is {df500[cname_all[0]].mean()}')
    
    # Plot stdv
    ts_to_plot_err = range(0,154,5)
    #kwargs = [fmt='o', markersize=8, capsize=20]
    kwargs = dict(fmt='o', color='r', markersize=3, capsize=2, linewidth=1, linestyle=None, zorder = 1000)
    x=df_date_yr['Date'].iloc[ts_to_plot_err]
    y=df500[cname_all[0]].iloc[ts_to_plot_err]
    err=std_GHB_all_ts.iloc[ts_to_plot_err]

    conv_coef = 0.295911 # 0.295911 AFY = 1 m3/day
    if unit_=='US':
        # PLOT GHB by 500 realizations
        plt.plot(x_GHB, y_GHB*conv_coef, color='#bfbfbf', linewidth=0.35)
        # GHB opt
        plt.plot(df_date_yr, df500[cname_all[0]]*conv_coef, color='r', linewidth=1.5)
        # stdv
        ax.errorbar(x, y*conv_coef, err*conv_coef, **kwargs)
        plt.ylabel('Southwest outflow (AFA)')
        ofile = 'Southwest_flow_UQ_AFA.png'
    else:
        # PLOT GHB by 500 realizations
        plt.plot(x_GHB, y_GHB, color='#bfbfbf', linewidth=0.35)
        # GHB opt
        plt.plot(df_date_yr, df500[cname_all[0]], color='r', linewidth=1.5)
        # stdv
        ax.errorbar(x, y, err, **kwargs)
        plt.ylabel('Southwest outflow (m$^3$/d)')
        ofile = 'Southwest_flow_UQ_m3d.png'

    
    
    #plt.ylabel('Southwest outflow (m$^3$/d)')
    #ax.set_title('1-stdv: ' + str(round(mean_std_GHB_over_ts,2)))
    fig.savefig(ofile, dpi=200, transparent=False, bbox_inches='tight')

    #plt.show()
    