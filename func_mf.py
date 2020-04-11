import pandas as pd
import math
import os
import subprocess
import flopy.utils.binaryfile as bf
import numpy as np
import matplotlib.pyplot as plt
# from func_plots import line_plot
from matplotlib import colors
from matplotlib.ticker import PercentFormatter, FuncFormatter
#from func_plots import plot_his
import seaborn as sns
from MFfunc import *

# Notes: Use python3, env b3 student's cluster
#


# Last updated: 08-02-2019
# [1] Read .os files and calculate RMSE

# Last updated: 08-06-2019
# Last opened: 08-08-2019



# Checklist before run
#[1] Check to make sure pname is correct

# [Step xxx] Define global options
gwmodel = 'ss'  # Choose analyzing data of ss or tr model


# [Step 11] Read os files from the ss model. Next step 21.
read_os_and_cal_rmse = False

# [Step 21] Read ccf files of ss models and gen csv file. Next step 22.
read_ccf_ss = False

# [Step 13] Extract a number of pars (out of total npars)
# Create parameters folder for DELSA's input
read_slice_parset = False     # Need to change path_pval

# [Step 22] Generate input files for delsa
# Remember to change ifile
# Convert this to a func later
get_input_for_delsa = False

get_id_err = False

read_ccf_tr = False
read_ccf_tr_at_cells = False
plot_ccf_output = False

# [Step 31] NO longer used, use read_os_and_cal_rmse instead.
tr_model_susie = False

# Read one os file. ===========================================================
read_one_os_file = True
ifile_one_os = 'model_final/TR_BLM_hpham5_pred._os'

# [Step 23] Plot hist of RMSE for baseline scenarios
plot_fitness_base_par = False

plot_option = 1 # 0: scatter, 1: hist, 2: sns

# [step 24] Plot DELSA for southwest outflow, spring flow and evt
# Run R code to get delsa_q.csv. Copy all folders 'all_out_xxx' 
# to local pc and run delsa code
# run read_slice_parset to get parameters folder
# remember to upload this folder after running the DELSA sampling code
 
plot_errorbar = False   # Plot error bar. 


plot_fitness_per_pars = False  # Not used?


generate_oc_file = False
read_out_file = False

#
# Options for the tr_model
read_os_and_save_hed = False    # read raw *._os of model and save hcal as df
                                # Specify nosb_start and nobs_stop
combine_csv_df_files =  False   # [Step 3x] Generate rmse_tr_npars.csv. Next, step 12.

read_out_of_tr_runs = False






nsamples = 1000  # realizations ss:1000, tr: 500
nsamples_to_plot = 10
# nos_files = 12500
# [Step 00] choose 'out_HK' or 'out_other_pars' or 'out_all_pars'
out_par = 'out_all_pars'

#
if os.path.isdir(out_par):   # Folder exists
    pass
else:
    cmd = 'mkdir ' + out_par
    subprocess.call(cmd, shell=True)

if gwmodel == 'ss':
    # Old
    #pname = ["HK_1", "HK_2", "HK_3", "HK_4", "HK_5","HK_6", "HK_7", "HK_8", "HK_9", "HK_10", "HK_60", 
    #            "GHB_13", "GHB_14", "GHB_15", "HFB_16", "HFB_17", "HFB_18", "HFB_19", "HFB_20", "HFB_21", 
    #            "HFB_22", "HFB_23", "HFB_24","VANI_200"]
    # NEW 136 pars
    pname = ["GHB_13", "GHB_14", "GHB_15", "HFB_16", "HFB_17", "HFB_18", "HFB_19", "HFB_20", "HFB_21", "HFB_22", "HFB_23", "HFB_24", 
             "HK_1", "HK_2", "HK_3", "HK_4", "HK_5", "HK_6", "HK_7", "HK_10", "HK_60", "VANI_200", "HK_8", "HK_9"]     

else:
    pname = ["SY_1", "SY_2", "SY_3", "SY_4", "SY_5", "SY_6", "SY_7", "SY_8","SY_9", "SY_10", "SY_60",
                "GHB_13", "GHB_14", "GHB_15", "HFB_16", "HFB_22", 
                "HFB_16", "HFB_17", "HFB_18", "HFB_19", "HFB_20", "HFB_21", "HFB_22" ]


npars = len(pname)
#npars = 136
# npars = 5 # npars to calculater delsa
nos_files = nsamples*npars + nsamples  # npars + par_fr_baseline_sce
print(f'\nNumber of parameters is {npars}')
print(f'\nNumber of os files is {nos_files}')

nos_files_start = 0  # Python id from 0
nos_files_stop = nos_files_start + nsamples*(npars +1)

#
#
# [Step 02] Get input for delsa
def get_inp_for_delsa(df, outvar, nsamples, npars, pname, out_folder=''):
    if os.path.isdir(out_folder) == False:   # Folder does not exist
        cmd = 'mkdir ' + out_folder
        subprocess.call(cmd, shell=True)
    # df = pd.read_csv(ifile)
    # reshape and print to file

    # Print base parameter
    df_base = df[outvar].loc[0:nsamples-1]
    ofile1 = 'out_' + outvar + '_out_Q_base.txt'
    df_base.to_csv(ofile1, header=False, index=None)
    cmd = 'mv ' + ofile1 + ' ' + out_folder
    subprocess.call(cmd, shell=True)

    # Print pertubated parameters
    df2 = pd.DataFrame(columns=pname, index=range(nsamples))
    for ipar in range(1, npars+1, 1):  # 1 to 24
        i1 = (ipar)*nsamples
        i2 = (ipar)*nsamples+nsamples
        i = range(i1, i2, 1)
        # print(i)
        df_tmp = df[outvar].loc[i]
        df_tmp.index = range(nsamples)
        df2[pname[ipar-1]] = df_tmp
    ofile2 = 'out_' + outvar + '_out_Q_pert.txt'
    df2.to_csv(ofile2, index=None, header=None)
    # Move file to a separated folder
    cmd = 'mv ' + ofile2 + ' ' + out_folder
    subprocess.call(cmd, shell=True)


# [Step 02] Get inputs for delsa

if get_input_for_delsa:
    print(f'\nGenerating inputs for delsa ...')
    # ifile = 'rmse_ss_5pars_1000samples.csv'  # For RMSE, e.g., 12500 columns, 24x500 + 500
    # ifile = 'ccf_ss_5pars_1000samples.csv'  # For GHB, DRT and ET
    
    # ifile = 'rmse_' + str(npars) + 'pars_' + str(nsamples) + 'samples.csv'
#    ifile = 'ccf_ss_' + str(npars) + 'pars_' + str(nsamples) + 'samples.csv'                       
    
    # TR models try 1
    #ifile = 'rmse_tr_all_pars.csv'
#    ifile = 'ccf_tr_23pars_500samples.csv'
    # TR models try 2, Susie's model with pilot points
    #ifile = 'ccf_ss_136pars_1000samples.csv'
    #ifile = 'rmse_136pars_146000samples.csv'
    # Try 3 - Choose one ifile
    #ifile = 'ccf_ss_24pars_1000samples.csv'
    ifile = 'rmse_24pars_1000samples.csv'
    
    print(f'Reading file {ifile} \n')
    df2 = pd.read_csv(ifile)
#    df2 = df.drop('run_id', axis=1)  # del col run_id
    list_col = df2.columns  #Check back later. 
    list_var = list_col.tolist()
    list_var.remove('run_id')
    for ivar in list_var:
        ofolder = 'out_' + ivar
        get_inp_for_delsa(df2, ivar, nsamples, npars, pname, out_folder=ofolder)
        print(f'\n    Output file(s) are saved at folder: {ofolder}')


#
# read os and cal rmse of steady state model
if read_os_and_cal_rmse:
    print(f'\nReading os files generated by the steady steate model.')
    # Calculate RMSE
    run_id = []
    rmse = []
    run_id_err = []

    for i in range(nos_files_start, nos_files_stop, 1):

        # ifile = 'BK_run12/all_os_files/out_' + str(i+1) + '._os' # First run, 2 orders mag. for par range

        # Second run, lo/up range = +-80% opt par val
        ifile = 'all_os_files/out_' + str(i+1) + '._os'

        if i % 1 == 0:
            print(f'Reading os file {ifile}')
        if os.path.isfile(ifile):  # File ._os is available
            df, df_drn, df_ghb = read_os(ifile)
            if i == nos_files_start:   # Only read one file to get nobs
                nobs = df.shape[0]
            rmse.append(
                math.sqrt(sum((df.hobs-df.hcal)*(df.hobs-df.hcal))/nobs))
            run_id.append(i+1)
        else:
            print('WARNING: File %s does not not exist' % (ifile))
            run_id_err.append(i+1)
    df = pd.DataFrame({'run_id': run_id, 'rmse': rmse})
    ifile = 'rmse_' + str(npars) + 'pars_' + str(nsamples) + 'samples.csv'
    print(f'{df.shape[0]} os files were read.')
    print(f'The output file is {ifile}, saved at the current directory.')
    df.to_csv(ifile,  index=None)

    if get_id_err:
        df_err = pd.DataFrame({'run_id': run_id_err})
        df_err.to_csv('err.csv')
        # print(df.head)
        # Get id of err runs
        df = pd.read_csv('BK_all_par_pertu.dat', header=None)
        df2 = df.loc[df_err.run_id-1]
        df2.to_csv('re_run.dat', header=None,  index=None)


#
# read os and save head generated by the steady state model
if read_os_and_save_hed:
    print(f'\nReading os files and save heads to a csv file.')
    for i in range(nos_files_start, nos_files_stop, 1):
        ifile = 'all_os_files/out_' + str(i+1) + '._os'

        if i % 100 == 0:
            print(f'Reading os file {ifile}')
        if os.path.isfile(ifile):  # File ._os is available
            df, df_drn, df_ghb = read_os(ifile)
        
        # add hcal and hobs to dataframe
        if i == nos_files_start:
            nobs = df.shape[0]
            new_name = 'r' + str(i+1)
            df2=df.rename({'hobs':'hobs', 'hcal':new_name}, axis=1)
        else:            
            new_name = 'r' + str(i+1)
            df_new=df.rename({'hobs':'hobs', 'hcal':new_name}, axis=1)
            df_new=df_new[new_name] # get hcal            
            df2 = pd.concat([df2, df_new], axis=1)
#        nobs_new = df.shape[0]
#        if nobs_new ~= nobs:
#            print(f'Warning: nobs is not the same at run {i}')

    ifile_zip = 'hed_' + str(npars) + 'pars_' + str(nsamples) + 'samples'
    ifile = 'hed_' + str(npars) + 'pars_' + str(nsamples) + 'samples.csv'
    df2.to_csv(ifile_zip, index=None, compression='gzip')
    df2.to_csv(ifile, index=None)


# [Step 21] ===================================================================
# Read ccf file of a steady state model
if read_ccf_ss:  # Steady state
    print(f'\nReading ccf files generated by the steady state model')
    run_id = []
    outflow_to_west = []
    outflow_drains = []
    outflow_evt = []
#    run_id_err = []
    for i in range(nos_files):
        # ifile = 'BK_run12/all_ccf_files/out_' + str(i+1) + '.ccf' # First run, 2 orders mag. for par range 
        ifile = 'all_ccf_files/out_' + str(i+1) + '.ccf'
        if i % 200 == 0:
            print(f'Reading the ccf file {ifile}')

        cbb = bf.CellBudgetFile(ifile)
#        list_unique_records_ = cbb.list_unique_records()  # Print all RECORDS
#        print 'nrecords=', cbb.get_nrecords()
        HDB = cbb.get_data(text='HEAD DEP BOUNDS', full3D=True)[0]
        ET = cbb.get_data(text='ET',           full3D=True)[0]
        DRT = cbb.get_data(text='DRAINS (DRT)',           full3D=True)[0]

        # Convert masked element to zero
        HDB = np.where(HDB.mask, 0, HDB)

        outflow_to_west.append(HDB.sum())
        outflow_drains.append(DRT.sum())
        outflow_evt.append(ET.sum())
        run_id.append(i+1)
    df = pd.DataFrame({'run_id': run_id, 'southwest': outflow_to_west,
                       'drt': outflow_drains, 'evt': outflow_evt})
    ifile = 'ccf_ss_' + str(npars) + 'pars_' + str(nsamples) + 'samples.csv'                       
    df.to_csv(ifile,  index=None)


#
#
#
# [] DEL later======================================================================
if tr_model_susie:
    # Calculate RMSE
    nos_files = 1
    run_id = []
    rmse = []
    run_id_err = []
    col_name = []
    for i in range(nos_files):
        # ifile = 'all_os_files/out_' + str(i+1) + '.ccf'
        
        ifile = 'all_os_files/out_' + str(i+1) + '.ccf'
        
        print('Reading %s \n' % (ifile))
        if os.path.isfile(ifile):  # File ._os is available
            df = pd.read_csv(ifile)
            if i == 0:   # Only read one file to get nobs
                nobs = df.shape[0]
            rmse.append(math.sqrt(sum((df.obs-df.cal)*(df.obs-df.cal))/nobs))
            run_id.append(i+1)
        else:
            #        print(f'File {ifile} does not not exist')
            run_id_err.append(i+1)
    df = pd.DataFrame({'run_id': run_id, 'rmse': rmse})



# Read ccf file of a TRANSIENT model
# Use python2, read transient ccf files
if read_ccf_tr:
    # Calculate RMSE
    nos_files = 1
    run_id = []
    rmse = []
    run_id_err = []
    col_name = []
    # for i in range(nos_files):
    # ifile = 'all_os_files/out_' + str(i+1) + '.ccf'
    ifile = 'testing.ccf'
    print('\nReading %s' % (ifile))
    cbb = bf.CellBudgetFile(ifile)

    list_unique_records = cbb.list_unique_records()  # Print all RECORDS
    # print 'nrecords=', cbb.get_nrecords()

    # Get a list of unique stress periods and time steps in the file
    list_time_steps = cbb.get_kstpkper()
#    print 'list_time_steps = ', cbb.get_kstpkper()

    nts = len(list_time_steps)
    # print 'Number of stress periods = ', nts
    STO_all_ts = np.empty([nts])

    n_stress_period = 104  # each is one years, divided into 12 months
    ncols = 4  # make sure you change this

    # '+2': one for GLO budget and the other for id of run realizations
    data_out = np.empty([n_stress_period, ncols])

#    drt_cell = drt_cell - 1  # Python index is from 0
    for ts in range(n_stress_period):
        STO = cbb.get_data(kstpkper=(0, ts), text='STORAGE',  full3D=True)[0]
        # CHD = cbb.get_data(kstpkper=(0, ts), text='CONSTANT HEAD',  full3D=True)[0]
        FRF = cbb.get_data(
            kstpkper=(0, ts), text='FLOW RIGHT FACE', full3D=True)[0]
        FFF = cbb.get_data(
            kstpkper=(0, ts), text='FLOW FRONT FACE', full3D=True)[0]
        FLF = cbb.get_data(
            kstpkper=(0, ts), text='FLOW LOWER FACE', full3D=True)[0]
        # WEL = cbb.get_data(kstpkper=(0, ts), text='WELLS',          full3D=True)[0]
        DRN = cbb.get_data(kstpkper=(0, ts), text='DRAINS',
                           full3D=True)[0]
        DRT = cbb.get_data(kstpkper=(0, ts), text='DRAINS (DRT)',
                           full3D=True)[0]

        # RLK = cbb.get_data(kstpkper=(0, ts), text='RIVER LEAKAGE',  full3D=True)[0]
        EVT = cbb.get_data(kstpkper=(0, ts), text='ET',
                           full3D=True)[0]
        HDB = cbb.get_data(
            kstpkper=(0, ts), text='HEAD DEP BOUNDS', full3D=True)[0]
        # RCH = cbb.get_data(kstpkper=(0, ts), text='RECHARGE',       full3D=True)[0]

        # Convert masked element to zero
        # FRF = np.where(FRF.mask, 0, FRF)
        # FFF = np.where(FFF.mask, 0, FFF)
        # FLF = np.where(FLF.mask, 0, FLF)
        # STO = np.where(STO.mask, 0, STO)
        # RLK = np.where(RLK.mask, 0, RLK)
        HDB = np.where(HDB.mask, 0, HDB)
        # CHD = np.where(CHD.mask, 0, CHD)
        # RCH = np.where(RCH.mask, 0, RCH)
        # WEL = np.where(WEL.mask, 0, WEL)
        # EVT = np.where(EVT.mask, 0, EVT)
        DRN = np.where(DRN.mask, 0, DRN)
        DRT = np.where(DRT.mask, 0, DRT)

        #
        total_STO_at_this_TS = STO.sum()
        total_DRN_at_this_TS = DRN.sum()
        total_DRT_at_this_TS = DRT.sum()

        data_out[ts, 0] = ts+1
        data_out[ts, 1] = total_STO_at_this_TS
        data_out[ts, 2] = total_DRN_at_this_TS
        data_out[ts, 3] = total_DRT_at_this_TS
#    df = pd.DataFrame[{'TS': data_out[ts, 0], 'STO':data_out[ts, 1],
#                       'DRN':data_out[ts, 2], 'DRT':data_out[ts, 3]}]
#    df.to_csv('data_out.csv', index=None)
    np.savetxt('data_out.txt', data_out, delimiter=',',
               header="TS, STO, DRN, DRT")


# Read ccf file of a TRANSIENT model to get budget at given cells
# Use python2, read transient ccf files
if read_ccf_tr_at_cells:
    # Calculate RMSE
    nos_files = 1
    run_id = []
    rmse = []
    run_id_err = []
    col_name = []

    # for i in range(nos_files):
    # ifile = 'all_os_files/out_' + str(i+1) + '.ccf'
    ifile = 'TR_BLM_update_052019.ccf'
    print('\nReading %s' % (ifile))
    cbb = bf.CellBudgetFile(ifile)

    list_unique_records = cbb.list_unique_records()  # Print all RECORDS
    # print 'nrecords=', cbb.get_nrecords()

    # Get a list of unique stress periods and time steps in the file
    list_time_steps = cbb.get_kstpkper()
#    print 'list_time_steps = ', cbb.get_kstpkper()

    nts = len(list_time_steps)
    # print 'Number of stress periods = ', nts

    n_stress_period = 104  # each is one years, divided into 12 months
    ncols = 3  # make sure you change this

    # '+2': one for GLO budget and the other for id of run realizations
    data_out = np.empty([n_stress_period, ncols])

    # Load data
    drt_cell = np.loadtxt("list_of_drt_cells.txt",
                          delimiter=',', dtype=int)  # use "" for python2
    n_drt_cells = len(drt_cell)

    drt_cell = drt_cell - 1  # IMPORTANT -  Python index is from 0

    for ts in range(n_stress_period):
        DRT = cbb.get_data(kstpkper=(0, ts), text='DRAINS (DRT)',
                           full3D=True)[0]
        DRT = np.where(DRT.mask, 0, DRT)

        #
        Q_springs = []
        for k in range(n_drt_cells):  # Cell k
            # print(k)
            # print(DRT.sum())
            Q_springs.append(
                DRT[drt_cell[k, 0], drt_cell[k, 1], drt_cell[k, 2]])
        Q_Bennetts = sum(Q_springs[0:2])
        Q_Manse = sum(Q_springs[2:5])

        data_out[ts, 0] = ts+1
        data_out[ts, 1] = Q_Bennetts
        data_out[ts, 2] = Q_Manse
    np.savetxt('simulated_spring_flow.txt', data_out, delimiter=',',
               header="TS, Q_Bennetts, Q_Manse")




#
if plot_ccf_output:
    data = np.loadtxt('data_out.txt')
    x = data[:, 0]
    y = data[:, 2]
    line_plot(x, -y, 'DRN', 'Time step', 'Flow rate (m3/d)', 'o-')

#
if read_one_os_file:
    print(f'Reading file {ifile_one_os}')
    df_hed, df_drn, df_ghb = read_os(ifile_one_os)
    # print(df_hed)

    # Plot head
    y = df_hed['hcal']
    x = range(1, len(y)+1, 1)
    ti = 'hobs vs. hcal'
    x_label = 'x'
    y_label = 'y'
    stly = 'o--'
    #line_plot(x, y, ti, x_label, y_label, stly)
    #plt.plot(y.iloc[:600])
    plt.plot(y)
    plt.show()
'''    
    y21 = df_drn['dobs']
    y22 = df_drn['dcal']
    x = range(1, len(y21)+1, 1)
    df_hed.to_csv('df_hed.csv')
    # print(x)

    # DRT spring flow
    ti = 'dobs vs. dcal'
    #plt.plot(x, -y21, stly, x, -y22, stly, ti, x_label, y_label)
    df_drn.to_csv('df_drn.csv')
    plt.show()

'''

# [Step 03] ONLY get a number of pars (out of 24)
if read_slice_parset: 
    print(f'\nExtracting {npars} first parameters from the total number of parameters')
#    npar = 11  # To extract data
    out_folder = 'parameters_' + out_par
    
    # input folder (after runnng R code of Milan)
    #path_pval = '/scratch/hpham/delsa/delsa_ss/parameters_ss_1000samples_new/'  # ss model v1
    #path_pval = '/scratch/hpham/delsa/delsa_tr_susie/output_run2_500k/' # tr model
    path_pval = '/scratch/hpham/delsa/delsa_ss_new/Rcode_output_parameters/'  # ss model v2
    
    print(f'Input data for parameter realizations is reading from {path_pval}')
    if os.path.isdir(out_folder) == False:   # Folder does not exist
        cmd = 'mkdir ' + out_folder
        subprocess.call(cmd, shell=True)

    # slice par base
    ifile = path_pval + 'par_base_0.txt'
    data = np.loadtxt(ifile, delimiter=' ')
    data_new = data[:,:npars]
    np.savetxt('par_base_0.txt', data_new)

    # slice par partu    
    for par in range(npars):        
        ifile_pval = path_pval + 'par_pertu_' + str(par+1) + '.txt'
        print(f'Opening file {ifile_pval}')
        data_pval = np.loadtxt(ifile_pval)
        data_new = data_pval[:,:npars]
        ofile = 'par_pertu_' + str(par+1) + '.txt'
        np.savetxt(ofile, data_new)
    cmd = 'mv par_*.txt ' + out_folder
    subprocess.call(cmd, shell=True)
    print(f'Output files are at folder {out_folder}')

 

# FOR pertu scenarios: Read and plot output from delsa: Par values vs. fitness
if plot_fitness_per_pars:
    # ifile = 'out_rmse_out_Q_base.txt'
    # ifile = r'C:\Users\hpham\Documents\P23_Pahrump\19_results_delsa\01_read_os_and_cal_rmse\for_DELSA\out_drt\out_drt_out_Q_pert.txt'
    ifile = 'out_rmse/out_rmse_out_Q_pert.txt'
    nbins = 20
    ofolder = 'fig_sens_ss'
    if os.path.isdir(ofolder) == False:   # Folder does not exist
        print('Created a new folder: {ofolder}')
        cmd = 'mkdir ' + ofolder
        subprocess.call(cmd, shell=True)

    print(f'\nPlotting fitness hist for a parmater')
    print(f'pname is: {pname}')    

    plot_his(ifile, pname, nbins)
    
    # Move file
    cmd = 'mv fig_sens_*.png ' + ofolder
    subprocess.call(cmd, shell=True)

#
# FOR the baseline scenario: Plot relationship between par values 
# and fitness/outputs (e.g., rmse, flow to west)
if plot_fitness_base_par:
    #
    # ifile = r'C:\Users\hpham\Documents\P23_Pahrump\19_results_delsa\01_read_os_and_cal_rmse\for_DELSA\out_west\out_west_out_Q_base.txt'      
    
    # RMSE
    if gwmodel=='ss':
        #ifile = '/scratch/hpham/delsa/delsa_ss/out_rmse/out_rmse_out_Q_base.txt' # ss model v1 no pilot points only 24 pars
        #ifile = '/scratch/hpham/delsa/delsa_ss_new/out_rmse/out_rmse_out_Q_base.txt' # ss model v2 with pilot points 136 pars               
        ifile = '/scratch/hpham/delsa/delsa_ss_new2/out_rmse/out_rmse_out_Q_base.txt' # ss model v3 with pilot points 138 pars               
        #bins=range(0,100,5) # steady state model v1
        bins=range(5,18,2) # steady state model v1
        ofile = 'fig_hist_par_base_0_ss_rmse.png'
        
    else:
        ifile = '/scratch/hpham/delsa/delsa_tr_susie/out_rmse/out_rmse_out_Q_base.txt' # tr model ver 1 no sens for HK
        bins=range(0,1000,50) # tr model
        ofile = 'fig_hist_par_base_0_ts_rmse.png'
    print(f'Reading file {ifile}')
    #ifile = 'out_southwest/out_southwest_out_Q_base.txt'
    #ifile = 'out_drt/out_drt_out_Q_base.txt'
    #ifile = 'out_evt/out_evt_out_Q_base.txt'
    
    data = np.loadtxt(ifile, delimiter=',')
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 2.5) 
    
    n_bins = 20
    #
    pval_cur = range(nsamples)  # 1000 realizations
    fitness = data[:nsamples_to_plot]  # Fitness values (e.g., RMSE)

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    kwargs = dict(ecolor='k', color='c', capsize=2,
            elinewidth=1, linewidth=0.6, ms=5, alpha=0.95)
    plt.rc('grid', linestyle="dotted", color='gray', alpha = 0.15)

    
    if plot_option == 0:  # Point plots
        plt.scatter(pval_cur, fitness, s=100, cmap='rainbow',
                    marker='o', linewidths=0.25, edgecolors='k', alpha=0.75)

        plt.xlabel('Realizations')
        # plt.ylabel('Outflow southwest (m3/day)')
        plt.ylabel('RMSE (m)')
        plt.grid(True)
        ofile = 'fig_scatter_par_base_0_rmse.png'
    elif plot_option == 1: # Hist plot
        counts, bins ,patches = plt.hist(fitness, bins, density=False,
            edgecolor='black', linewidth=0.5, alpha = 0.5)
        plt.ylabel('Frequency')
        plt.xlabel('RMSE (m)')
        ax.grid(True)
        #plt.show()

    elif plot_option == 2:
        # Density Plot and Histogram 

        sns.distplot(fitness, hist=True, kde=True, 
                    bins=bins, color = 'darkblue', 
                    hist_kws={'edgecolor':'black'},
                    kde_kws={'linewidth': 1})
#        sns.distplot(fitness, bins=n_bins,
#             hist_kws=dict(cumulative=True),
#             kde_kws=dict(cumulative=True))                    

#        sns.distplot(fitness, hist = False, kde = True,
#                        kde_kws = {'shade': True, 'linewidth': 2}, 
#                        label = airline)
        plt.ylabel('Density')
        plt.xlabel('RMSE (m)')

        ofile = 'fig_dens_par_base_0_ss_rmse.png'

        ax.grid(True)
        #plt.show()



# https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

        # We'll color code by height, but you could use any scalar
        #fracs = N / N.max()

        # we need to normalize the data to 0..1 for the full range of the colormap
        #norm = colors.Normalize(fracs.min(), fracs.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        #for thisfrac, thispatch in zip(fracs, patches):
        #    color = plt.cm.viridis(norm(thisfrac))
        #    thispatch.set_facecolor(color)

        # Now we format the y-axis to display percentage
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        
        ##ax.set_xlim([0, 100])
        ##ax.yaxis.set_major_formatter(FuncFormatter('{0:.0}'.format))
        

        
        

    
    fig.savefig(ofile, dpi=200, transparent=False, bbox_inches='tight')

    plt.show()

    
if generate_oc_file==True:
    n_stress_period = 104
    f1 = open('temp.oc', 'w')
    for i in range(n_stress_period):
        f1.write(f'PERIOD {str(i+1)} STEP 1\n')
        f1.write(f'   PRINT BUDGET \n')        
    f1.close()

if read_out_file:
    ifile = 'TR_BLM_update_FINAL_FINAL_2016.out'
    print(f'Reading MODFLOW out file ')

    # Read file, line-by-line to find the starting row of data
    n_stress_period = 104
    id_in = range(2,n_stress_period*4,4)  # Each repeated twice (in and out), MF outputed at t = 1 and t=12 of each stress period.
    id_out = range(3,n_stress_period*4,4)
    id2 = range(1,n_stress_period*2,2)

    STO = []
    GHB = [] 
    DRT = []
    ET = []
    Total_In = []
    Total_Out = []
    WEL = []
    count=0
    with open(ifile) as f:
        for line in f:          
            data = line.split('=')
            if '             STORAGE =' in line:                
                STO.append(float(data[2]))
            elif '     HEAD DEP BOUNDS =' in line:
                GHB.append(float(data[2]))
            elif '        DRAINS (DRT) =' in line:               
                DRT.append(float(data[2]))
            elif '                  ET =' in line:               
                ET.append(float(data[2]))
            elif '            TOTAL IN =' in line:               
                Total_In.append(float(data[2]))                                 
            elif '           TOTAL OUT =' in line:               
                Total_Out.append(float(data[2]))                                 
            elif '               WELLS =' in line:               
                WEL.append(float(data[2]))   

#    df = pd.DataFrame({'STO_IN':STO, 'GHB_IN':GHB, 'DRT_IN':DRT, 'ET_IN':ET})
#    df.to_csv('budget_all.csv')

    # BUDGET IN
    STO_IN = []
    GHB_IN = []
    DRT_IN = []
    ET_IN = []
    WEL_IN = []
    for i in id_in:
        STO_IN.append(STO[i])
        GHB_IN.append(GHB[i])
        DRT_IN.append(DRT[i])
        ET_IN.append(ET[i])
        WEL_IN.append(WEL[i])
#    df_IN = pd.DataFrame({'STO_IN':STO_IN, 'GHB_IN':GHB_IN, 'DRT_IN':DRT_IN, 'ET_IN':ET_IN})
#    df_IN.to_csv('budget_in.csv')

    # BUDGET OUT    
    STO_OUT = []
    GHB_OUT = []
    DRT_OUT = []
    ET_OUT = []
    WEL_OUT = []
    for i in id_out:
        STO_OUT.append(STO[i])
        GHB_OUT.append(GHB[i])
        DRT_OUT.append(DRT[i])
        ET_OUT.append(ET[i])
        WEL_OUT.append(WEL[i])

    #
    Total_In_NEW = []
    Total_Out_NEW = []
    for i in id2:
        Total_In_NEW.append(Total_In[i])
        Total_Out_NEW.append(Total_Out[i])        
    df = pd.DataFrame({'STO_IN':STO_IN, 'GHB_IN':GHB_IN, 'DRT_IN':DRT_IN, 'ET_IN':ET_IN,
                       'STO_OUT':STO_OUT, 'GHB_OUT':GHB_OUT, 'DRT_OUT':DRT_OUT, 'ET_OUT':ET_OUT,
                       'Total_In':Total_In_NEW, 'Total_Out': Total_Out_NEW, 
                       'WEL_IN':WEL_IN, 'WEL_OUT':WEL_OUT})
    df.to_csv('budget.csv')

if plot_errorbar:
    print(f'Generating errorbar plot ...')
    sub_fig = [['(a)', '(b)'],['(c)','(d)']]
    fsize = 12
    # Re-arrange pname
    pname = ["GHB_13", "GHB_14", "GHB_15", "HFB_16", "HFB_17", "HFB_18", "HFB_19", "HFB_20", "HFB_21", "HFB_22", "HFB_23", "HFB_24", 
            "HK_1", "HK_2", "HK_3", "HK_4", "HK_5", "HK_6", "HK_7", "HK_8", "HK_9", "HK_10", "HK_60", "VANI_200"]  
    if gwmodel == 'ss':
        list_csv = [['delsa_out_rmse.csv', 'delsa_out_southwest.csv'],
                    ['delsa_out_drt.csv','delsa_out_evt.csv']]
    else:    
        list_csv = [['delsa_out_rmse.csv', 'delsa_out_GHB_OUT.csv'],
                    ['delsa_out_DRT_OUT.csv','delsa_out_ET_OUT.csv']]

    fig, axs  = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True)
    fig.set_size_inches(11, 7)
    
    for i in range(2):
        for j in range(2):
            df=pd.read_csv(list_csv[i][j])        
            
            df=df[pname]
            mu = df.mean(skipna=True)
            stdv = df.std(skipna=True)
            stderr = df.sem(skipna=True)

            # Plotting the error bars
            kwargs = dict(ecolor='k', color='k', capsize=2,
                    elinewidth=1, linewidth=0.6, ms=5, alpha=0.95)
            plt.rc('grid', linestyle="dotted", color='blue', alpha = 0.15)

            # plt.xticks(rotation=90)
            axs[i,j].errorbar(pname, mu, yerr=stdv, fmt='o', mfc='r', **kwargs)
            axs[i,j].set_title(sub_fig[i][j])
            axs[i,0].set_ylabel('DELSA (-)', fontsize=fsize)
            # axs[1,j].set_xlabel('DELSA (-)', fontsize=fsize)
            #axs[i,j].set_ylim(-0.1, 1.1)
            axs[i,j].set_ylim(-0.2, 1.1)
            axs[i,j].grid(True)
            for tick in axs[i,j].get_xticklabels():
                tick.set_rotation(90)        


            # Adding plotting parameters
            
    # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
    # axarr[0,0].set_title(sub_fig[k])
    # axarr[0,1].set_title(sub_fig[k])
    # axarr[1,0].set_title(sub_fig[k])
    # axarr[1,1].set_title(sub_fig[k])
    # ax0.set_xlabel('X axis', fontsize=fsize)
    # axarr[1,0].set_ylabel('DELSA (-)', fontsize=fsize)
    # ax0.set_xlim(0, 12)
    # axarr[0,0].set_ylim(-0.2, 0.8)
    fig.tight_layout()
    ofile = 'delsa_' + gwmodel + '_.png'
    fig.savefig(ofile, dpi=200, transparent=False, bbox_inches='tight')
    plt.show()


if combine_csv_df_files:
    df1=pd.read_csv('hed_1.csv')
    
    # Load the second dataset and drop column hobs
    df2=pd.read_csv('hed_2.csv')
    df2=df2.drop("hobs", axis=1)
    
    df3=pd.read_csv('hed_3.csv')
    df3=df3.drop("hobs", axis=1)

    df4=pd.read_csv('hed_4.csv')
    df4=df4.drop("hobs", axis=1)

    df5=pd.read_csv('hed_5.csv')
    df5=df5.drop("hobs", axis=1)


    df_all = pd.concat([df1, df2, df4, df3, df5], axis=1)
    #df_all.to_csv('hed_all.csv', index=None)
    
    # Calculate RMSE
    nobs = df_all.shape[0]
    ncurr_runs = df_all.shape[1] -1
    rmse = []
    for irun in range(ncurr_runs):
        col_name = 'r' + str(irun+1)
        rmse.append(math.sqrt((sum(df_all[col_name] - df_all.hobs)**2)/nobs))
    run_id = range(1,ncurr_runs+1,1)
    df_rmse = pd.DataFrame({'run_id':run_id, 'rmse':rmse})
    df_rmse.to_csv('rmse_tr_all_pars.csv', index=None)


#
cal_delsa = False
if cal_delsa == True:    
    if gwmodel =='tr':
        # [1] 

        # [2] Get input for delsa
        ifile = 'rmse_tr_npars.csv'
        #get_input_for_delsa(ifile)
    else:
        print(f'Nothing run ... passing ...')
        pass

#
#
# Read csv file produced by tr models
if read_out_of_tr_runs:
    print(f'\nReading csv files and save output to csv files.')
    for i in range(nos_files_start, nos_files_stop, 1):
        ifile = 'all_out_files/out_' + str(i+1) + '.csv'

        if i % 100 == 0:
            print(f'Reading os file {ifile}')
        if os.path.isfile(ifile):  # File ._os is available
            df = pd.read_csv(ifile)
        else:
            print(f'Warning: file {ifile} does not exist!')    
        
        # add hcal and hobs to dataframe
        if i == nos_files_start:
            nobs = df.shape[0]
            df_new=df.mean(axis=0)
            df2 = df_new
        else:  
#            new_name = 'r' + str(i+1)
#            df_new=df.rename({'hobs':'hobs', 'hcal':new_name}, axis=1)
#            df_new=df_new[new_name] # get hcal  
            df_new=df.mean(axis=0)          
            df2 = pd.concat([df2, df_new], axis=1)
    df_final=df2.T
    run_id = range(nos_files_start+1,nos_files_stop+1,1)
    df_final['run_id'] = run_id # add a new column to df

    ifile = 'ccf_tr_' + str(npars) + 'pars_' + str(nsamples) + 'samples.csv'
    df_final.to_csv(ifile, index=None)    
  