#!/usr/bin/env python3

'''
[1] Main functions ============================================================

func_mf.py
- Generate a MODFLOW .oc file for MODFLOW (generate_oc_file)
- Read a MODFLOW *.out file and save budget information (read_out_file)
- Read a MODFLOW *.ccf file for a SS model (read_ccf_ss)
- Read a MODFLOW *.ccf file for a TR model (read_ccf_tr)
- Read MF *.ccf file & get info @givencells, aTR model (read_ccf_tr_at_cells)
- Prepare input files for DELSA sensitivity analysis (get_inp_for_delsa)
- Read a single MODFOW *.os file (read_one_os_file)
- Read mult. MODFLOW *._os files & cal RMSE (ss model) (read_os_and_cal_rmse)
- Read mul MF *._os ile and save all heds (MCMC sim) (read_os_and_save_hed)
- Plot MODFLOW ccf output (plot_ccf_output)
- combine_csv_df_files
- For sensitivity analysis ONLY
    - Get of number of pars (out of n pars) (read_slice_parset)
    - Plot DELSA results: par vals vs. fitness (plot_fitness_per_pars)
    - Plot DELSA results: Par_val vs fitness/outputs (plot_fitness_base_par)
    - Plot DELSA results: plot_errorbar


pv_UQ_main.py



MFfunc.py [List of functions]
- read_os()
- read_multi_os_files()
- process_mult_gwlevels()
- MapPlotPV()
- plot_spring_flow()

[2] Run sequence




'''
