import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np 
import random

def train_model_from_files(file_list):
    #path_dir = "cvt_with_colors"
    #file_list = os.listdir(path_dir)
    model_x = []
    model_y_power = []
    model_y_current = []
    for info in file_list:
        app_name = info[0]
        device_idx = info[1]
        exp_idx = info[2]
        repeat_idx = info[3]
        file_name_cvt = f"cvt_logs{device_idx}/log_cvt_{app_name}_{exp_idx}_{repeat_idx}"
        file_name_util = f"util_logs{device_idx}/log_util_{app_name}_{exp_idx}_{repeat_idx}"
        cvt_file = open(file_name_cvt, 'r')
        cvts = cvt_file.readlines()
        util_file = open(file_name_cvt, 'r')
        cvts = cvt_file.readlines()
        for j in range(len(cvts)):
            line_list = cvts[j].split(' ')
            if line_list[0] == '':
                continue
            if len(line_list) < 8:
                continue
            if float(line_list[1]) > 0:
                continue
            # r g b battery temp brightness
            model_x_tmp = []
            model_x_tmp.append(float(line_list[-3]) / 255)
            model_x_tmp.append(float(line_list[-2]) / 255)
            model_x_tmp.append(float(line_list[-1]) / 255)
            model_x_tmp.append(float(line_list[2]) / 100)
            model_x_tmp.append(float(line_list[4]) / 1000)
            model_x_tmp.append(float(line_list[5]) / 1000)
            model_x.append(model_x_tmp)
            model_y_current.append(float(line_list[0]) / 1000000)
            model_y_power.append(((float(line_list[0]) / 1000000) * (float(line_list[3]) / 1000000)))
    print(sum(model_y_power) / len(model_y_power), sum(model_y_current) / len(model_y_current))
    print(max(model_y_power), max(model_y_current))
    for i in range(10):
        print(model_x[i*100])
    mlr_color_current = LinearRegression()
    mlr_color_current.fit(model_x, model_y_current)
    mlr_color_power = LinearRegression()
    mlr_color_power.fit(model_x, model_y_power)
    return mlr_color_current, mlr_color_power

def train_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    print("model ceof", model.coef_, model.intercept_)
    return model

def predict(model, test_x):
    y_predict = model.predict(test_x)
    return y_predict

def calculate_section_avg(predict, real, num):
    predict_avgs = []
    real_avgs = []
    idx = 0
    predict_tmp = []
    real_tmp = []
    for i in range(len(predict)):
        predict_tmp.append(predict[i])
        real_tmp.append(real[i])
        idx += 1
        if idx == num:
            idx = 0
            predict_avgs.append(sum(predict_tmp) / num)
            real_avgs.append(sum(real_tmp) / num)
            predict_tmp = []
            real_tmp = []
    return predict_avgs, real_avgs
    
def calculate_error(predict, real):
    error = 0
    for predict_val, real_val in zip(predict, real):
        error += abs(predict_val - real_val) / real_val
    error /= len(predict)
    mean_val = sum(real) / len(real)
    return error, mean_val

    #return np.sqrt(sum([pow(p-r, 2)  for p, r in zip(predict, real)]))

def split_data_cpu_old(file_list, interval, frequency):
    train_x_each_power = []
    train_x_total_power = []
    train_x_each_current = []
    train_x_total_current = []
    test_x_each_power = []
    test_x_total_power = []
    test_x_each_current = []
    test_x_total_current = []
    train_y_power = []
    train_y_current = []
    test_y_power = []
    test_y_current = []
    for info in file_list:
        app_name = info[0]
        device_idx = info[1]
        exp_idx = info[2]
        repeat_idx = info[3]
        file_name_cvt = f"cvt_logs{device_idx}/log_cvt_{app_name}_{exp_idx}_{repeat_idx}"
        file_name_util = f"util_logs{device_idx}/log_util_{app_name}_{exp_idx}_{repeat_idx}"
        cvt_file = open(file_name_cvt, 'r')
        cvts = cvt_file.readlines()
        util_file = open(file_name_util, 'r')
        utils = util_file.readlines()
        util_start_time, cvt_start_time = float(utils[0].split('[')[1].split(']')[0].split(',')[-2]), float(cvts[0].split(' ')[-2])
        util_end_time, cvt_end_time = float(utils[-1].split('[')[1].split(']')[0].split(',')[-1]), float(cvts[-1].split(' ')[-1])
        log_end_time = min(util_end_time, cvt_end_time)
        log_start_time = max(util_start_time, cvt_start_time)
        util_datas = [[] for tmp in range(int((log_end_time - log_start_time) // frequency) + 1)]
        cvt_datas = [[] for tmp in range(int((log_end_time - log_start_time) // frequency) + 1)]
        print((int((log_end_time - log_start_time) // frequency)))
        for j in range(len(utils)):
            line_list = utils[j].split('[')[1].split(']')[0].split(',')
            util_start_time_tmp = float(utils[j].split('[')[1].split(']')[0].split(',')[-2])
            util_end_time_tmp = float(utils[j].split('[')[1].split(']')[0].split(',')[-1])
            if util_start_time_tmp < log_start_time:
                continue
            if util_end_time > log_end_time:
                continue
            idx_from = int((util_start_time_tmp - log_start_time) // frequency)
            idx_to = idx_from + int((util_end_time_tmp - util_start_time_tmp) // frequency)
            for k in range(idx_from, idx_to+1):
                if not util_datas[k]:
                    util_datas[k].append(float(line_list[0]) / 100.0)
                    for m in range(3):
                        util_datas[k].append(float(line_list[2+m]) / 100.0)
                        util_datas[k].append(float(line_list[5+m].split('\'')[1]) / 10000000.0)
        for j in range(len(cvts)):
            line_list = cvts[j].strip().split(' ')
            cvt_start_time_tmp = float(line_list[-2])
            cvt_end_time_tmp = float(line_list[-1])
            if cvt_start_time_tmp < log_start_time:
                continue
            if cvt_end_time_tmp > log_end_time:
                continue
            idx_from = int((cvt_start_time_tmp - log_start_time) // frequency)
            idx_to = idx_from + int((cvt_end_time_tmp - cvt_start_time_tmp) // frequency)
            for k in range(idx_from, idx_to+1):
                if not cvt_datas[k]:
                    for data_tmp in line_list:
                        cvt_datas[k].append(float(data_tmp))
        line_list_past = []
        for i in range(int((log_end_time - log_start_time) // frequency)):
            if util_datas[i] and cvt_datas[i]:
                if float(cvt_datas[i][0]) < 0 or float(cvt_datas[i][1]) > 0:
                    continue
                idxtmp = i-interval
                line_list_past = []
                while (not line_list_past) and idxtmp >= 0:
                    if cvt_datas[idxtmp] and util_datas[idxtmp]:
                        line_list_past = cvt_datas[idxtmp]
                        break
                    else:
                        idxtmp -= 1
                if not line_list_past:
                    line_list_past = cvt_datas[i]
                random_pass = random.randrange(0,4)
                x_each = []
                x_total = []
                #모든 util, freq 쌍
                #x_each.extend(list(map(float,util_datas[i][1:7])))
                #util * freq 쌍
                #util_freq_sum = 0
                for j in range(3):
                    #util_freq_sum += float(util_datas[i][1+j]) * float(util_datas[i][4+j])
                    x_each.append(float(util_datas[i][1+j]) * float(util_datas[i][4+j]))
                #x_each.append(util_freq_sum)
                x_total.extend([float(util_datas[i][1])])
                x_each.extend([float(cvt_datas[i][2]) / 100, float(cvt_datas[i][4]) / 1000])
                x_total.extend([float(cvt_datas[i][2]) / 100, float(cvt_datas[i][4]) / 1000])
                x_each_current = x_each.copy()
                x_each_power = x_each.copy()
                x_total_current = x_total.copy()
                x_total_power = x_total.copy()
                x_each_current.append(float(line_list_past[0]) / 1000000)
                x_each_power.append(((float(line_list_past[0]) / 1000000) * (float(line_list_past[3]) / 1000000)))
                x_total_current.append(float(line_list_past[0]) / 1000000)
                x_total_power.append(((float(line_list_past[0]) / 1000000) * (float(line_list_past[3]) / 1000000)))
                if random_pass == 0:
                    test_x_each_power.append(x_each_power)
                    test_x_each_current.append(x_each_current)
                    test_x_total_power.append(x_total_power)
                    test_x_total_current.append(x_total_current)
                    test_y_current.append(float(cvt_datas[i][0]) / 1000000)
                    test_y_power.append(((float(cvt_datas[i][0]) / 1000000) * (float(cvt_datas[i][3]) / 1000000)))
                else:
                    train_x_each_power.append(x_each_power)
                    train_x_each_current.append(x_each_current)
                    train_x_total_power.append(x_total_power)
                    train_x_total_current.append(x_total_current)
                    train_y_current.append(float(cvt_datas[i][0]) / 1000000)
                    train_y_power.append(((float(cvt_datas[i][0]) / 1000000) * (float(cvt_datas[i][3]) / 1000000)))
                #cpu each util, freq, 
    return test_x_each_current, train_x_each_current, test_x_total_current, train_x_total_current, test_x_each_power, train_x_each_power, test_x_total_power, train_x_total_power, test_y_current, train_y_current, test_y_power, train_y_power

def make_test_set(path_dir, file_list):
    model_x = []
    model_y_power = []
    model_y_current = []
    for file in file_list:
        print(file)
        file_name = path_dir + '/' + file
        cvt_file = open(file_name, 'r')
        cvts = cvt_file.readlines()
        for j in range(len(cvts)):
            line_list = cvts[j].split(' ')
            if line_list[0] == '':
                continue
            if len(line_list) < 10:
                continue
            if float(line_list[1]) > 0:
                continue
            # r g b battery temp brightness, 이전 정보?
            model_x_tmp = []
            model_x_tmp.append(float(line_list[-3]) / 255)
            model_x_tmp.append(float(line_list[-2]) / 255)
            model_x_tmp.append(float(line_list[-1]) / 255)
            model_x_tmp.append(float(line_list[2]) / 100)
            model_x_tmp.append(float(line_list[4]) / 1000)
            model_x_tmp.append(float(line_list[5]) / 1000)
            model_x.append(model_x_tmp)
            model_y_current.append(float(line_list[0]) / 1000000)
            model_y_power.append(((float(line_list[0]) / 1000000) * (float(line_list[3]) / 1000000)))
    return model_x, model_y_current, model_y_power


test_x_each_current, train_x_each_current, test_x_total_current, train_x_total_current, test_x_each_power, train_x_each_power, test_x_total_power, train_x_total_power, test_y_current, train_y_current, test_y_power, train_y_power = split_data_cpu_old([["cpu_task", 3, 0, 0]], 1, 0.3)
mlr_each_current = train_model(train_x_each_current, train_y_current)
mlr_each_power = train_model(train_x_each_power, train_y_power)
mlr_total_current = train_model(train_x_total_current, train_y_current)
mlr_total_power = train_model(train_x_total_power, train_y_power)

predict_each_current = mlr_each_current.predict(test_x_each_current)
error, mean = calculate_error(predict_each_current, test_y_current)
print(error, mean)
predict_each_power = mlr_each_power.predict(test_x_each_power)
error, mean = calculate_error(predict_each_power, test_y_power)
predict_total_power = mlr_total_power.predict(test_x_total_power)
print(error, mean)
time_x = [i for i in range(len(predict_each_power))]
plt.figure()
plt.plot(time_x, test_y_power, 'r')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/cpu_each_power_real.png')
plt.figure()
plt.plot(time_x, predict_each_power, 'b')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/cpu_each_power_predict.png')

predict_avgs, real_avgs = calculate_section_avg(predict_each_power, test_y_power, 100)
time_x = [i for i in range(len(predict_avgs))]
plt.figure()
plt.plot(time_x, predict_avgs, 'b')
plt.plot(time_x, real_avgs, 'r')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/cpu_each_power_avgs2.png')
avgs_error = list(map(lambda x,y : abs(x-y) / max(x,y), predict_avgs, real_avgs))
plt.figure()
plt.plot(time_x, avgs_error, 'b')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/cpu_each_power_avg_errors2_withpast.png')
print("avgs0 error avg : ", sum(avgs_error) / len(avgs_error))

predict_avgs, real_avgs = calculate_section_avg(predict_total_power, test_y_power, 100)
time_x = [i for i in range(len(predict_avgs))]
plt.figure()
plt.plot(time_x, predict_avgs, 'b')
plt.plot(time_x, real_avgs, 'r')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/cpu_total_power_avgs2.png')
avgs_error = list(map(lambda x,y : abs(x-y) / max(x,y), predict_avgs, real_avgs))
plt.figure()
plt.plot(time_x, avgs_error, 'b')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/cpu_total_power_avg_errors2_withpast.png')
print("avgs0 error avg : ", sum(avgs_error) / len(avgs_error))

predict_total_current = mlr_total_current.predict(test_x_total_current)
error, mean = calculate_error(predict_total_current, test_y_current)
print(error, mean)
error, mean = calculate_error(predict_total_power, test_y_power)
print(error, mean)
errors = list(map(lambda x,y : abs(x-y) / max(x,y), predict_total_power, test_y_power))
print(sum(errors) / len(errors), sum(test_y_power) / len(test_y_power))
errors = list(map(lambda x,y : abs(x-y) / max(x,y), predict_each_power, test_y_power))
print(sum(errors) / len(errors), sum(test_y_power) / len(test_y_power))

exit(1)

path_dir = "cvt_with_colors"
file_list = ["displaytest__4_1", "displaytest__4_2", "displaytest__4_3", "log_cvt_displaytest_0_4_4", "log_cvt_displaytest_0_4_5", "log_cvt_displaytest_0_4_6", "log_cvt_displaytest_0_4_7"]
#file_list = os.listdir(path_dir)
test_x_current, train_x_current, test_x_power, train_x_power, test_y_current, train_y_current, test_y_power, train_y_power = split_data(path_dir, file_list, 1)
mlr_color_current = train_model(train_x_current, train_y_current)
mlr_color_power = train_model(train_x_power, train_y_power)
predict_y_power = mlr_color_power.predict(test_x_power)
predict_y_current = mlr_color_current.predict(test_x_current)
error, mean = calculate_error(predict_y_power,test_y_power)
print(error, mean)
error, mean = calculate_error(predict_y_current,test_y_current)
print(error, mean)
power_errors = list(map(lambda x,y : abs(x-y), predict_y_power, test_y_power))
predict_avgs, real_avgs = calculate_section_avg(predict_y_power, test_y_power, 4, 4, 16)
time_x = [i for i in range(len(predict_avgs))]
plt.figure()
plt.plot(time_x, predict_avgs, 'b')
plt.plot(time_x, real_avgs, 'r')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_avgs0.png')
avgs_error = list(map(lambda x,y : abs(x-y) / max(x,y), predict_avgs, real_avgs))
plt.figure()
plt.plot(time_x, avgs_error, 'b')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_avg_errors0.png')
print("avgs0 error avg : ", sum(avgs_error) / len(avgs_error))
time_x = [i for i in range(len(test_y_power))]
plt.figure()
plt.plot(time_x, test_y_power, 'b', label='power real')
plt.plot(time_x, predict_y_power, 'r', label='power predict')
plt.title(f"predict and real")
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_predict0.png')


file_list = ["displaytest_2_4_1", "displaytest_2_4_2", "displaytest_2_4_3"]
#file_list = os.listdir(path_dir)
test_x_current, train_x_current, test_x_power, train_x_power, test_y_current, train_y_current, test_y_power, train_y_power = split_data(path_dir, file_list, 1)
mlr_color_current = train_model(train_x_current, train_y_current)
mlr_color_power = train_model(train_x_power, train_y_power)
predict_y_power = mlr_color_power.predict(test_x_power)
predict_y_current = mlr_color_current.predict(test_x_current)
error, mean = calculate_error(predict_y_power,test_y_power)
print(error, mean)
error, mean = calculate_error(predict_y_current,test_y_current)
print(error, mean)
power_errors = list(map(lambda x,y : abs(x-y), predict_y_power, test_y_power))
predict_avgs, real_avgs = calculate_section_avg(predict_y_power, test_y_power, 4, 4, 16)
time_x = [i for i in range(len(predict_avgs))]
plt.figure()
plt.plot(time_x, predict_avgs, 'b')
plt.plot(time_x, real_avgs, 'r')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_avgs2.png')
avgs_error = list(map(lambda x,y : abs(x-y) / max(x,y), predict_avgs, real_avgs))
plt.figure()
plt.plot(time_x, avgs_error, 'b')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_avg_errors2.png')

file_list = ["displaytest_3_4_1", "displaytest_3_4_2", "displaytest_3_4_3"]
#file_list = os.listdir(path_dir)
test_x_current, train_x_current, test_x_power, train_x_power, test_y_current, train_y_current, test_y_power, train_y_power = split_data(path_dir, file_list, 1)
mlr_color_current = train_model(train_x_current, train_y_current)
mlr_color_power = train_model(train_x_power, train_y_power)
predict_y_power = mlr_color_power.predict(test_x_power)
predict_y_current = mlr_color_current.predict(test_x_current)
error, mean = calculate_error(predict_y_power,test_y_power)
print(error, mean)
error, mean = calculate_error(predict_y_current,test_y_current)
print(error, mean)
power_errors = list(map(lambda x,y : abs(x-y), predict_y_power, test_y_power))
predict_avgs, real_avgs = calculate_section_avg(predict_y_power, test_y_power, 4, 4, 16)
time_x = [i for i in range(len(predict_avgs))]
plt.figure()
plt.plot(time_x, predict_avgs, 'b')
plt.plot(time_x, real_avgs, 'r')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_avgs3.png')
avgs_error = list(map(lambda x,y : abs(x-y) / max(x,y), predict_avgs, real_avgs))
plt.figure()
plt.plot(time_x, avgs_error, 'b')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_avg_errors3.png')

file_list = ["displaytest_2_4_3"]
file_list2 = ["displaytest_2_4_3"]
mlr_color_current, mlr_color_power = train_model_from_files(path_dir, file_list)
test_x, test_y_current, test_y_power = make_test_set(path_dir, file_list2)
test_x2, test_y_current2, test_y_power2 = make_test_set(path_dir, file_list)

predict_y_power = mlr_color_power.predict(test_x)
predict_y_current = mlr_color_current.predict(test_x)

error, mean = calculate_error(predict_y_power,test_y_power)
print(error, mean)
error2, mean2 = calculate_error(predict_y_current, test_y_current)
print(error2, mean2)
power_errors = list(map(lambda x,y : abs(x-y), predict_y_power, test_y_power))

count_tmp = 0
for error in power_errors:
    if error < 0.4:
        count_tmp += 1
print("ERROR(COUNT)",count_tmp, len(power_errors))
'''
#print("Average Compare", sum(test_y_power) / len(test_y_power), sum(test_y_power2) / len(test_y_power2))
#print("Average Compare", sum(test_y_current) / len(test_y_current), sum(test_y_current2) / len(test_y_current2))
#time_x = [i for i in range(len(test_y_current))]
'''
time_x = [i for i in range(len(test_y_current))]
plt.figure()
plt.plot(time_x, power_errors, 'b', label='abs errors')
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_errors.png')

plt.figure()
plt.plot(time_x, test_y_current, 'b', label='current real')
plt.plot(time_x, predict_y_current, 'r', label='current predict')
plt.title(f"predict and real")
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_current_predict1.png')

plt.figure()
plt.plot(time_x, test_y_power, 'b', label='power real')
plt.plot(time_x, predict_y_power, 'r', label='power predict')
plt.title(f"predict and real")
plt.savefig(f'/home/hj/BatteryAOSP10/figures/display_power_predict1.png')
