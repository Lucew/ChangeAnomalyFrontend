import pandas as pd
from changepoynt.algorithms.sst import SST
from changepoynt.algorithms.esst import ESST
import h5py
import numpy as np
import datetime
import time
import plotly.express as px


def read_data():
    # time the read (and use a print to make sure the cache works)
    start = time.perf_counter()

    # read the pressure data -------------------------------------------------------------------------------------------

    # get the data into memory as a dict of numpy arrays
    file_path = "./HS2/Pressure_monitoring/HS2_Pressure_Monitoring.mat"
    with h5py.File(file_path) as f:
        dat = {k: np.array(v) for k, v in f.items()}

    # time array for inj dat
    Nt = len(dat['time_inj'][0])
    t_inj_dat = np.zeros(Nt)
    for i in range(Nt):
        t_inj_dat[i] = datetime.timedelta(dat['time_inj'][0, i] - dat['time_inj'][0, 0]).seconds
    t_inj_dat /= 3600

    # get the timings from the recordings
    t0 = dat['time_inj'][0, 0]
    tend = dat['time_inj'][0, -1]

    # read the flow data -----------------------------------------------------------------------------------------------
    file_path = "./HS2/Injection_protocol/20170208_HS2inj.dat"

    vol_dat = pd.read_csv(file_path)
    id0_vol = np.argmin(np.abs(vol_dat['time_datenum'] - t0))
    idend_vol = np.argmin(np.abs(vol_dat['time_datenum'] - tend))
    Nt = len(vol_dat['time_datenum'][id0_vol:idend_vol])
    t_vol = np.zeros(Nt)
    for i in range(id0_vol, idend_vol):
        t_vol[i-id0_vol] = datetime.timedelta(vol_dat['time_datenum'][i] - vol_dat['time_datenum'][id0_vol]).seconds

    # get the flow and pressure signal of interest
    flow_time = t_vol/3600
    flow_signal = vol_dat['cleaned_flow_lpm'][id0_vol:idend_vol].to_numpy()
    pressure_time = t_inj_dat[:-1]
    pressure_signal = dat['pressure'][0][:-1]

    # end the time
    print(f"Data is in memory (and cached) and loading took: {time.perf_counter()-start} s.")
    return flow_time, flow_signal, pressure_time, pressure_signal


def compute_change_score(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    scorer = SST(window_length=window_size, scoring_step=step_size, method="rsvd")
    change_score = scorer.transform(signal)
    return change_score

