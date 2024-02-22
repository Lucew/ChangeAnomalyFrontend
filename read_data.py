import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from changepoynt.algorithms.sst import SST
from changepoynt.algorithms.esst import ESST


# read pressure data
file_path = "./HS2/Pressure_monitoring/HS2_Pressure_Monitoring.mat"

f = h5py.File(file_path)
print(f.keys())
dat = {}
for k, v in f.items():
    dat[k] = np.array(v)

prp_dat = {}
for k, v in f['Data'].items():
    prp_dat[k] = np.array(v)

t0 = dat['time_inj'][0,0]
tend = dat['time_inj'][0,-1]

# time array for inj dat
Nt = len(dat['time_inj'][0])
t_inj_dat = np.zeros(Nt)
for i in range(Nt):
    t_inj_dat[i] = datetime.timedelta(dat['time_inj'][0, i] - dat['time_inj'][0, 0]).seconds
t_inj_dat /= 3600
    
# time array for prp dat
id0_prp = np.argmin(np.abs(prp_dat['Datenum'][0] - t0))
idend_prp = np.argmin(np.abs(prp_dat['Datenum'][0] - tend))
Nt = len(prp_dat['Datenum'][0, id0_prp:idend_prp])
t_prp_dat = np.zeros(Nt)
for i in range(id0_prp, idend_prp):
    t_prp_dat[i-id0_prp] = datetime.timedelta(prp_dat['Datenum'][0,i] - prp_dat['Datenum'][0,id0_prp]).seconds
t_prp_dat /= 3600
    
# time array for logger_obs dat
id0_lobs = np.argmin(np.abs(dat['time_logger_obs'][0] - t0))
idend_lobs = np.argmin(np.abs(dat['time_logger_obs'][0] - tend))
Nt = len(dat['time_logger_obs'][0,id0_lobs:idend_lobs])
t_logger_obs_dat = np.zeros(Nt)
for i in range(id0_lobs, idend_lobs):
    t_logger_obs_dat[i-id0_lobs] = datetime.timedelta(
        dat['time_logger_obs'][0,i] - dat['time_logger_obs'][0,id0_lobs]).seconds
t_logger_obs_dat /= 3600
    
fig, ax = plt.subplots(nrows=5, ncols=1)
fig.text(0, 0.5, 'Pressure / MPa', rotation='vertical', va='center', ha='center')

ax[0].plot(t_inj_dat, dat['pressure'][0])
ax[0].set_ylabel('INJ1')
ax[0].set_xticks([])
ax[1].plot(t_prp_dat, prp_dat['PRP1_1'][0,id0_prp:idend_prp]/1000)
ax[1].plot(t_prp_dat, prp_dat['PRP1_2'][0,id0_prp:idend_prp]/1000)
ax[1].plot(t_prp_dat, prp_dat['PRP1_3'][0,id0_prp:idend_prp]/1000)
ax[1].set_xticks([])
ax[1].set_ylabel('PRP1')
ax[2].plot(t_prp_dat, prp_dat['PRP2_1'][0,id0_prp:idend_prp]/1000)
ax[2].plot(t_prp_dat, prp_dat['PRP2_2'][0,id0_prp:idend_prp]/1000)
ax[2].set_ylabel('PRP2')
ax[2].set_xticks([])
ax[3].plot(t_prp_dat, prp_dat['PRP3_1'][0,id0_prp:idend_prp]/1000)
ax[3].plot(t_prp_dat, prp_dat['PRP3_2'][0,id0_prp:idend_prp]/1000)
ax[3].set_ylabel('PRP3')
ax[3].set_xticks([])
ax[4].plot(t_inj_dat, dat['pressure_obs'][0])
ax[4].plot(t_logger_obs_dat, dat['pressure_logger_obs'][0,id0_lobs:idend_lobs])
ax[4].set_xlabel('Time / h')
ax[4].set_ylabel('INJ2')
fig.tight_layout()

# read strain data
file_path = "./HS2/FBG_strain_data/FBG_FBS1_strain_all_experiments.mat"

f = h5py.File(file_path)
print(f.keys())
fbs1_dat = {}
for k, v in f.items():
    fbs1_dat[k] = np.array(v)
    
id0_fbs1 = np.argmin(np.abs(fbs1_dat['time'][:,0] - t0))
idend_fbs1 = np.argmin(np.abs(fbs1_dat['time'][:,0] - tend))
Nt = len(fbs1_dat['time'][id0_fbs1:idend_fbs1,0])
t_fbs1_dat = np.zeros(Nt)
for i in range(id0_fbs1, idend_fbs1):
    t_fbs1_dat[i-id0_fbs1] = datetime.timedelta(
        fbs1_dat['time'][i,0] - fbs1_dat['time'][id0_fbs1,0]).seconds
    
fig2, ax2 = plt.subplots(nrows=1, ncols=1)
ax2.plot(t_fbs1_dat/3600, fbs1_dat['STRAIN1'][id0_fbs1:idend_fbs1,:])
ax2.set_xlabel('Time / h')
ax2.set_ylabel('Strain / $\mu\epsilon$')
fig2.tight_layout()

# read volumetric flow rate data
file_path = "./HS2/Injection_protocol/20170208_HS2inj.dat"

vol_dat = pd.read_csv(file_path)
id0_vol = np.argmin(np.abs(vol_dat['time_datenum'] - t0))
idend_vol = np.argmin(np.abs(vol_dat['time_datenum'] - tend))
Nt = len(vol_dat['time_datenum'][id0_vol:idend_vol])
t_vol = np.zeros(Nt)
for i in range(id0_vol, idend_vol):
    t_vol[i-id0_vol] = datetime.timedelta(
        vol_dat['time_datenum'][i] - vol_dat['time_datenum'][id0_vol]).seconds

fig3, ax3 = plt.subplots(nrows=1, ncols=1)
ax3.plot(t_vol/3600, vol_dat['cleaned_flow_lpm'][id0_vol:idend_vol])
ax3.set_xlabel('Time / h')
ax3.set_ylabel('Flow rate / L min$^{-1}$')
# plt.show()

# get the flow and pressure signal of interest
flow_time = t_vol/3600
flow_signal = vol_dat['cleaned_flow_lpm'][id0_vol:idend_vol].to_numpy()
pressure_time = t_inj_dat[:-1]
pressure_signal = dat['pressure'][0][:-1]
print(type(flow_signal), type(pressure_time))
print(flow_time.shape, pressure_time.shape)

# check whether they are synchronous
assert np.all(np.equal(flow_time, pressure_time)), "The time steps are not equal -> resample."

# compute the change score over both signals
scorer = SST(30, method="rsvd")
flow_change = scorer.transform(flow_signal)
pressure_change = scorer.transform(pressure_signal)
fig4, ax4 = plt.subplots(1, 1)
ax4.plot(flow_change, pressure_change, 'kx')
ax4.set_xlabel("Flow Change")
ax4.set_ylabel("Pressure Change")
plt.show()