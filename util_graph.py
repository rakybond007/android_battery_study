from matplotlib import pyplot as plt
import numpy as np
import os
import json
for i in range(1):
    file_nanme = f"util_logs/log_util_twitch_6_1"
    util_file = open(file_nanme, 'r')
    utils = util_file.readlines()
    print(len(utils))
    start_time = float(utils[0].split('[')[1].split(']')[0].split(',')[-2])
    time_x = []
    cpu_util_y = []
    gpu_util_y = []
    for j in range(len(utils)):
        line_list = utils[j].split('[')[1].split(']')[0].split(',')
        time_x.append(float(line_list[-2]) - start_time)
        cpu_util_y.append(float(line_list[0]))
        gpu_util_y.append(float(line_list[1].split('\'')[1]))
    plt.plot(time_x, cpu_util_y, 'b', label='cpu util')
    plt.plot(time_x, gpu_util_y, 'r', label='gpu util')
    plt.xlabel('time')
    plt.ylabel('utils')
    plt.title(f"util_twitch{i}")
    plt.legend(loc="upper right")
    plt.show()
    util_file.close()

def draw_batterystats_power_change(exp_num, app_name):
    log_list = os.listdir(f"batterystats_logs/{exp_num}")
    app_y = []
    screen_y = []
    idx_x = []
    for log in log_list:
        f = open(f"batterystats_logs/{exp_num}/{log}", "r")
        lines = f.readlines()
        powers = []
        for line in lines :
            #print(line)
            if line[0] != '{':
                line = line[1:]
                #print(line)
            data = json.loads(line.strip('\"').strip('!').strip('#').strip())
            #print(data)
            powers.append(data)
        powers = sorted(powers, key=lambda k: k.get('totalPowerMah',0), reverse=True)
        idx = int(log.split('_')[1])
        idx_x.append(idx)
        for power in powers:
            print(power)
            if "package_name" in power:
                if app_name in power['package_name']:
                    app_y.append(float(power['totalPowerMah']))
                elif "Screen" in power['package_name']:
                    screen_y.append(float(power['totalPowerMah']))
                else:
                    continue
    print(idx_x)
    plt.plot(idx_x, app_y, 'b', label='batterystats twitch_app')
    plt.xlabel('idx')
    plt.ylabel('totalPower')
    plt.title(f"twitch batterystats app")
    plt.legend(loc="upper right")
    plt.show()
    plt.plot(idx_x, screen_y, 'b', label='batterystats twitchscreen')
    plt.xlabel('idx')
    plt.ylabel('totalPower')
    plt.title(f"twitch batterystats screen")
    plt.legend(loc="upper right")
    plt.show()

draw_batterystats_power_change(3, "twitch")

#cvt
file_nanme = f"cvt_logs/log_cvt_twitch_6_1"
cvt_file = open(file_nanme, 'r')
cvts = cvt_file.readlines()
#print(len(utils))
start_time = float(cvts[0].split(' ')[-2])
time_x = []
current_y = []
temp_y = []
voltage_y = []
cnt = 0
current_tmp = []
time_tmp = []
current_avgs = []
time_avgs = []
for j in range(len(cvts)):
    line_list = cvts[j].split(' ')
    if line_list[0] == '':
        continue
    time_x.append(float(line_list[-2]) - start_time)
    current_y.append(float(line_list[0]))
    voltage_y.append(float(line_list[1]))
    print(line_list)
    temp_y.append(float(line_list[2]))
    cnt += 1 
    current_tmp.append(float(line_list[0]))
    time_tmp.append(float(line_list[-2]) - start_time)
    if cnt == 1000:
        cnt = 0
        time_avg = sum(time_tmp) / len(time_tmp)
        current_avg = sum(current_tmp) / len(time_tmp)
        if current_avg < 2550000:
            time_avgs.append(time_avg)
            current_avgs.append(current_avg)
        time_tmp = []
        current_tmp = []

plt.plot(time_x, current_y, 'b', label='current')
plt.xlabel('time')
plt.ylabel('current')
plt.title(f"twitch_current")
plt.legend(loc="upper right")
plt.show()
plt.plot(time_avgs, current_avgs, 'b', label='current_avgs')
plt.xlabel('time')
plt.ylabel('current')
plt.title(f"twitch_current")
plt.legend(loc="upper right")
plt.show()
plt.plot(time_x, voltage_y, 'r', label='')
plt.xlabel('time')
plt.ylabel('voltage')
plt.title(f"twitch_voltage")
plt.legend(loc="upper right")
plt.show()
plt.plot(time_x, temp_y, 'g', label='')
plt.xlabel('time')
plt.ylabel('temp')
plt.title(f"twitch_temp")
plt.legend(loc="upper right")
plt.show()
cvt_file.close()