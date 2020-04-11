import os
import numpy as np
import numpy.ma as ma
import flopy.utils.binaryfile as bf
from shutil import copyfile, move
import pandas as pd
from scipy.interpolate import interp1d
import math
import subprocess


# USE this file FOR UQ and Sens.

# Generate a pval file, run MODFLOW and process ._os + .out files
# Use this file for reading results of sensitivity analysis and uncertainty analysis
# Update 08062019
# ...


# Update these pars before running ...
npars = 148
id_cali = list(range(0, npars, 1))
parameter_range = [0, 10]  # CMA-ES range [0, 10]; pysot range [-5.12, 5.12]

n_budget_components = 7
mname_prefix = "TR_BLM_update_FINAL_FINAL_2066_HKparams"
n_stress_period = 154  # 104 for tr_model, 154 for 50 years pred model
n_ts_in_one_stress_period = 12
return_coeficient = 0.3

# List of input files
ifile_loc_spring_cells = 'list_of_drt_cells.csv'
ifile_offpring = 'offspring.dat'
ifile_par_range = 'par_range.csv'
ifile_ts = 'list_of_ts.csv'
#ifile_obs_flow_rate = 'spring_flow_rate.csv'


#
run_MODFLOW = True  # []
read_ccf = True    # Read ccf and get spring flow?
read_out_file = True  # .out file to get values of flow budget components

opt_genpval = True  # Gen a MODFLOW pval file?
scale_to_mf = True  # True if using cma/pysot/UQ.

move_ofiles = True  # For sensitity analysis only
sensitivity_anslysis = True  # Using this file for sens. anal. / UQ?

if sensitivity_anslysis:  # First columns is ID of run
    x = np.loadtxt(ifile_offpring, delimiter=',')
    rid = int(x[0])
    # List of OUTPUT files
    #ofile_ccf = "out_" + str(rid) + ".ccf"
    ofile_os = "out_" + str(rid) + "._os"
    ofile_hed = "out_" + str(rid) + ".hed"
    ofile_drw = "out_" + str(rid) + ".drw"
    ofile_budget = "out_wbud_" + str(rid) + ".csv"
    ofile_springflow = "out_sprflow_" + str(rid) + ".csv"
else:
    x = np.loadtxt(ifile_offpring, delimiter=',')

# par_name = ["HK_2", "HK_4", "HK_3", "HK_6", "HK_5", "HK_7", "HK_8", "HK_1", "HK_9", "HK_10", "HK_60", "GHB_14", "GHB_13",
#            "GHB_15", "HFB_16", "HFB_22", "HFB_18", "HFB_24", "HFB_21", "HFB_17", "HFB_19", "HFB_20", "HFB_23", "VANI_200"]

# par_name = ["GHB_14", "GHB_13", "GHB_15", "HFB_16", "HFB_22", "HFB_18", "HFB_24", "HFB_21", "HFB_17", "HFB_19",
#            "HFB_20", "HFB_23", "SY_2", "SY_4", "SY_3", "SY_6", "SY_5", "SY_7", "SY_8", "SY_1", "SY_9", "SY_10", "SY_60"]

# Susie tr model, 23 pars
# par_name = ["SY_2", "SY_4", "SY_3", "SY_6", "SY_5", "SY_7", "SY_8", "SY_1", "SY_9", "SY_10", "SY_60", "GHB_14",
#            "GHB_13", "GHB_15", "HFB_16", "HFB_22", "HFB_18", "HFB_24", "HFB_21", "HFB_17", "HFB_19", "HFB_20", "HFB_23"]

# HP's modified Susie_tr_model HK+SS+VANI. TOtal: 36 pars. Sequence must be the same as in par_range.csv
#par_name = ["HK_101", "HK_102", "HK_103", "HK_104", "HK_105", "HK_106", "HK_107", "HK_108", "HK_109", "HK_110", "HK_111",
#            "SY_1", "SY_2", "SY_3", "SY_4", "SY_5", "SY_6", "SY_7", "SY_8", "SY_9", "SY_10", "SY_60",
#            "GHB_13", "GHB_14", "GHB_15", "HFB_16", "HFB_17", "HFB_18", "HFB_19", "HFB_20", "HFB_21", "HFB_22", "HFB_23", "HFB_24",
#            "SS_201", "VANI_901"]

# Susie_tr_model 148 pars with ppoints. Sequence must be the same as in par_range.csv
par_name = ["HK_2","HK_4","HK_3","HK_6","HK_5","HK_7","pp1_1","pp1_2","pp1_3","pp1_4","pp1_5","pp1_6","pp1_7","pp1_8","pp1_9","pp1_10","pp1_11","pp1_12","pp1_13","pp1_14","pp1_15","pp1_16","pp1_17","pp1_18","pp1_19","pp1_20","pp1_21","pp1_22","pp1_23","pp1_24","pp1_25","pp1_26","pp1_27","pp1_28","pp1_29","pp1_30","pp1_31","pp1_32","pp1_33","pp1_34","pp1_35","pp1_36","pp1_37","pp1_38","pp1_39","pp1_40","pp1_41","pp1_42","pp1_43","pp1_44","pp1_45","pp1_46","pp1_47","pp1_48","pp1_49","pp1_50","pp1_51","pp1_52","pp1_53","pp1_54","pp1_55","pp1_56","pp1_57","pp1_58","pp1_59","pp1_60","HK_60","HK_1","pp2_1","pp2_2","pp2_3","pp2_4","pp2_5","pp2_6","pp2_7","pp2_8","pp2_9","pp2_10","pp2_11","pp2_12","pp2_13","pp2_14","pp2_15","pp2_16","pp2_17","pp2_18","pp2_19","pp2_20","pp2_21","pp2_22","pp2_23","pp2_24","pp2_25","pp2_26","pp2_27","pp2_28","pp2_29","pp2_30","pp2_31","pp2_32","pp2_33","pp2_34","pp2_35","pp2_36","pp2_37","pp2_38","pp2_39","pp2_40","pp2_41","pp2_42","pp2_43","pp2_44","pp2_45","pp2_46","pp2_47","pp2_48","pp2_49","pp2_50","pp2_51","pp2_52","pp2_53","pp2_54","pp2_55","HK_10","VANI_200","SY_32","SY_34","SY_33","SY_36","SY_35","SY_37","SY_38","SY_360","SY_31","SY_39","SY_310","GHB_14","GHB_13","GHB_15","HFB_16","HFB_22","HFB_18","HFB_24","HFB_21","HFB_17","HFB_19","HFB_20","HFB_23"]

def gen_pval_MODFLOW_file(mname_prefix, npars, df, par):
    # Gen pval file (work with both python2 and python3)
    # For python2+ ==============================================================
    ifile = mname_prefix + '.pval'
    f1 = open(ifile, 'w')
    f1.write("%d \n" % (npars))
    for i in range(npars):
        f1.write("%9s %9.6e \n" % (df['Name'].iloc[i], par[i]))
    f1.close()


if opt_genpval:
    # Read parameter ranges from par_range.inp
    df = pd.read_csv('par_range.csv')
    # Scale cmaes/pysot values to MODFLOW values
    if scale_to_mf:
        par = []
        x2 = np.delete(x, 0)  # Delete the first column
        for i in range(0, len(id_cali)):
            k = id_cali[i]    # Only change par values of npar_cali parameters.
            par.append(
                float(interp1d(parameter_range, [df['Min'].iloc[i], df['Max'].iloc[i]])(x2[i])))
    else:
        par = np.delete(x, 0)  # Delete the first column [for sens. ana. only]
    # Call function to generate pval
    gen_pval_MODFLOW_file(mname_prefix, npars, df, par)

# [2] Run MODFLOW =================================================================
# Run MODFLOW
if run_MODFLOW:
    cmd = './mfnwt ' + mname_prefix + '.mfn > omf.txt'
    #print(f'\nRunning MODFLOW NWT ...')
    subprocess.call(cmd, shell=True)
    #os.system('./mfnwt TR_BLM_update_052019.mfn > omf.txt')
    #p = subprocess.Popen(['mfnwt', 'TR_BLM_update_052019.mfn'], stdout=None, stderr=None)
    # p.wait()
    #    ifile = mname_prefix + '.mfn'
    #os.system('./mfnwt TR_BLM_update_052019.mfn > omf.txt')
    #    cmd = './mfnwt ' + ifile + ' > omf.txt'
    #    subprocess.call(cmd, shell=True)
    #os.system('mfnwt TR_BLM_hpham5.mfn > omf.txt')


# ==============================================================================
# Read *.out file from MODFLOW
if read_out_file:
    ifile = mname_prefix + '.out'
#    print(f'Reading MODFLOW out file ')

    # Read file, line-by-line to find the starting row of data

    # Each repeated twice (in and out), MF outputed at t = 1 and t=12 of each stress period.
    id_in = range(2, n_stress_period*4, 4)
    id_out = range(3, n_stress_period*4, 4)
    id2 = range(1, n_stress_period*2, 2)

    STO = []
    GHB = []
    DRT = []
    ET = []
    Total_In = []
    Total_Out = []
    WEL = []
    count = 0
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
    df = pd.DataFrame({'STO_IN': STO_IN, 'GHB_IN': GHB_IN, 'DRT_IN': DRT_IN, 'ET_IN': ET_IN,
                       'STO_OUT': STO_OUT, 'GHB_OUT': GHB_OUT, 'DRT_OUT': DRT_OUT, 'ET_OUT': ET_OUT,
                       'Total_In': Total_In_NEW, 'Total_Out': Total_Out_NEW,
                       'WEL_IN': WEL_IN, 'WEL_OUT': WEL_OUT})
    df.to_csv('budget.csv', index=None)

if read_ccf:  # Read ccf file to get simulated spring flows
    ifile = mname_prefix + '.ccf'
    fsize = os.path.getsize(ifile)
    if fsize > 4000000000:   # make sure model is converged (ccf > 4GB)
        cbb = bf.CellBudgetFile(ifile)
        # list_unique_records = cbb.list_unique_records()  # Print all RECORDS

        # ncols = 3  # make sure you change this
        #data_out = np.empty([n_stress_period, ncols])

        # Load data

        drt_cell = np.loadtxt(ifile_loc_spring_cells,
                              delimiter=',', dtype=int)  # use "" for python2
        n_drt_cells = len(drt_cell)
        drt_cell = drt_cell - 1  # IMPORTANT -  Python index is from 0
        Q_Bennetts = []
        Q_Manse = []
    #    fid = open('flow_at_cells.csv', 'w')
        for ts in range(n_stress_period):
            for month in range(n_ts_in_one_stress_period):
                DRT = cbb.get_data(kstpkper=(month, ts),
                                   text='DRAINS (DRT)', full3D=True)[0]
                DRT = np.where(DRT.mask, 0, DRT)

                #
                Q_springs = []
                for k in range(n_drt_cells):  # Cell k
                    # print(k)
                    # print(DRT.sum())
                    Q_springs.append(
                        DRT[drt_cell[k, 0], drt_cell[k, 1], drt_cell[k, 2]])
    #            fid.write(f'{ts+1}, {month+1}, {Q_springs} \n')
                # Two first cells are for Bennetts
                Q_Bennetts.append(sum(Q_springs[0:2])/(1-return_coeficient))
                Q_Manse.append(sum(Q_springs[2:5])/(1-return_coeficient))
    #    fid.close()
        # Load time step
        df1 = pd.read_csv(ifile_ts)
        df1.Date = pd.to_datetime(df1.Date)
        # Simulated flow df2 are:
        df2 = pd.DataFrame(
            {'Date': df1.Date, 'Qcal_Bennetts': Q_Bennetts, 'Qcal_Manse': Q_Manse})
        df2.to_csv('simulated_flow.csv')
    else:  # small ccf (meaning fail converged)
        pass

# Copy files
if move_ofiles:
    # copyfile('data_out.txt', ofile_ccf)  # Rename files
    #
    cmd = 'mv ' + mname_prefix + '._os' + ' ../all_os_files/' + ofile_os
    subprocess.call(cmd, shell=True)
    #
    cmd = 'mv ' + mname_prefix + '.hed' + ' ../all_hed_files/' + ofile_hed
    subprocess.call(cmd, shell=True)
    #
#    cmd = 'mv ' + mname_prefix + '.drw' + ' ../all_drw_files/' + ofile_drw
#    subprocess.call(cmd, shell=True)
    #
    cmd = 'mv budget.csv ../all_wbud_files/' + ofile_budget
    subprocess.call(cmd, shell=True)
    #
    cmd = 'mv simulated_flow.csv ../all_spring_flow_files/' + ofile_springflow
    subprocess.call(cmd, shell=True)
