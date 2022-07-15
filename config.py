
path = 'C:\\Users\\sweaton\\OneDrive - Sandia National Laboratories\\Desktop\\Projects\\MINOS Venture\\Fan Acoustics Project\\Infrasound Paper\\software for upload\\'
filename = 'WACO_Station_Acoustic_Data.parquet'
filepath = path + filename

gt_filename = 'Tachometer_Ground_Truth.parquet'
gt_filepath = path + gt_filename

samples_per_second = 100
window = 90
overlap = 0.5

start = '2018-09-16T14:00:00'
stop = '2018-09-16T18:00:00'

freq_lims = [10, 42]
spl_lims = [40, 70]

gear_ratio = 11.14
blades = 8
span = 3
fan_number_guess = 3
total_fans = 4
max_fan_bpf = 21.54
total_filter_height = 5
mid_filter_height = 3
harmonics = [1, 2]
