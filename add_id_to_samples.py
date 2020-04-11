import numpy as np

transform_data = True
select_pars = False

# Notes: manually copy par_base_0.txt to all_par_pertu.txt


if transform_data:
    data = np.loadtxt('all_par_pertu_.txt')
    nrealizations = 1000
    nsamples = data.shape[0]
    npars = data.shape[1]
    #id = np.linspace(nsamples+1, nsamples+nsamples, 1)
    id = np.linspace(1, nsamples, nsamples)
    data_new = np.zeros((nsamples, npars+1))
    data_new[:, 0] = id
    data_new[:, 1:] = data
    np.savetxt('all_par_pertu.dat', data_new, delimiter=',')


# choose priority parameters to run
if select_pars:
    # par_id = [4, 5, 6, 8, 9, 12, 13, 14, 15,
    #          17, 19, 20, 21, 24, 25, 31, 32, 34]
    par_id = range(1, npars, 1)
    id_to_run = np.empty([nrealizations, len(par_id)], dtype=int)
    id_to_run_reshaped = np.empty([len(par_id)*nrealizations], dtype=int)

    for i in range(len(par_id)):
        print(i)
        start_id = (par_id[i]-1)*nrealizations + 1
        print(start_id)
        id_to_run[:, i] = np.linspace(
            start_id, start_id + nrealizations-1, nrealizations) - 1
    id_to_run_reshaped = np.reshape(np.transpose(
        id_to_run), [len(par_id)*nrealizations])
    data_new2 = data_new[id_to_run_reshaped, :]
    np.savetxt('all_par_pertu.dat', data_new2, delimiter=',')
