import os
from shutil import copyfile
curr_dir = os.getcwd()

nrun_folders = 12500
for i in range(nrun_folders):
    curr_run_folder = 'run_' + str(i+1)
    os.chdir(curr_run_folder)
    rid = i+1+12500
    ofile_ccf = "out_" + str(rid) + ".ccf"
#    ofile_os = "out_" + str(rid) + "._os"
    ofile_out = "out_" + str(rid) + ".out"

    # Copy files
    copyfile('TR_BLM_hpham5.ccf', ofile_ccf)  # Rename files
#    copyfile('TR_BLM_hpham5._os', ofile_os)  # Rename files
    copyfile('TR_BLM_hpham5.out', ofile_out)  # Rename files
    #
    os.system('mv out*.ccf ../all_ccf_files_2')  # Move to another folders
#    os.system('mv out*._os ../all_os_files_2')  # Move to another folders
    os.system('mv out*.out ../all_out_files_2')  # Move to another folders
    print(f'Copying file {ofile_ccf}')
    os.chdir(curr_dir)
