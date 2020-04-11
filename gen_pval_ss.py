import os
import numpy as np
#from scipy.interpolate import interp1d
from shutil import copyfile

npars = 24
data = np.loadtxt('offspring.dat', delimiter=',')

rid = int(data[0])
ofile_ccf = "out_" + str(rid) + ".ccf"
ofile_os = "out_" + str(rid) + "._os"
ofile_out = "out_" + str(rid) + ".out"

par = np.delete(data, 0)  # Delete the first column

par_name = ["HK_2", "HK_4", "HK_3", "HK_6", "HK_5", "HK_7", "HK_8", "HK_1", "HK_9", "HK_10", "HK_60", "GHB_14", "GHB_13",
            "GHB_15", "HFB_16", "HFB_22", "HFB_18", "HFB_24", "HFB_21", "HFB_17", "HFB_19", "HFB_20", "HFB_23", "VANI_200"]

# For python3 ===============================================================
with open("TR_BLM_hpham5.pval", "w") as text_file:
    print("{}".format(npars), file=text_file)
    for i in range(npars):
        print("{:9s}    {:12.9e}".format(par_name[i], par[i]), file=text_file)


# Run MODFLOW
os.system('./mfnwt TR_BLM_hpham5.mfn > omf.txt')

# Copy files
copyfile('TR_BLM_hpham5.ccf', ofile_ccf)  # Rename files
copyfile('TR_BLM_hpham5._os', ofile_os)  # Rename files
copyfile('TR_BLM_hpham5.out', ofile_out)  # Rename files
os.system('mv out*.ccf ../all_ccf_files')  # Move to another folders
os.system('mv out*._os ../all_os_files')  # Move to another folders
os.system('mv *.out ../all_out_files')  # Move to another folders
