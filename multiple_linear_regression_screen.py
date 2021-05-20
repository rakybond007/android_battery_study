import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np 
import random

a = [[1,0,0],[1,0,0],[1,0,0]]
b = [2,3,4]
df = pd.DataFrame([x for x in zip(a, b)])
print(df)
print(df[0])
mlr = LinearRegression()
mlr.fit(a, b)
y_predict = mlr.predict([[1,0,0]])
print(y_predict)

def train_model_from_files(path_dir, file_list):
    #path_dir = "cvt_with_colors"
    #file_list = os.listdir(path_dir)
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

def calculate_section_avg(predict, real, change_num, brightness_num, color_num):
    predict_avgs = []
    real_avgs = []
    for i in range(change_num):
        each_change_num = len(predict) // change_num
        for j in range(brightness_num):
            each_brightness_data_num = each_change_num // brightness_num
            for k in range(color_num):
                each_color_set_num = each_brightness_data_num // color_num
                section_start_idx = i*each_change_num+j*each_brightness_data_num+k*each_color_set_num
                section_end_idx = i*each_change_num+j*each_brightness_data_num+(k+1)*each_color_set_num
                predict_avg = sum(predict[section_start_idx:section_end_idx]) / each_color_set_num
                real_avg = sum(real[section_start_idx:section_end_idx]) / each_color_set_num
                predict_avgs.append(predict_avg)
                real_avgs.append(real_avg)
    return predict_avgs, real_avgs

def calculate_error(predict, real):
    error = 0
    for predict_val, real_val in zip(predict, real):
        error += abs(predict_val - real_val)
    error /= len(predict)
    mean_val = sum(real) / len(real)
    return error, mean_val

    #return np.sqrt(sum([pow(p-r, 2)  for p, r in zip(predict, real)]))

def split_data(path_dir, file_list, interval):
    train_x_power = []
    train_x_current = []
    test_x_power = []
    test_x_current = []
    train_y_power = []
    train_y_current = []
    test_y_power = []
    test_y_current = []
    for file in file_list:
        file_name = path_dir + '/' + file
        cvt_file = open(file_name, 'r')
        cvts = cvt_file.readlines()
        for j in range(len(cvts)):
            line_list = cvts[j].split(' ')
            if j-interval >= 0:
                line_list_past = cvts[j-interval].split(' ')
            else:
                line_list_past = line_list
            if line_list[0] == '':
                continue
            if len(line_list) < 10:
                continue
            if float(line_list[1]) > 0:
                continue
            # r g b battery temp brightness past_val
            model_x_tmp = []
            #model_x_tmp.append(float(line_list[-3]) / 255)
            #model_x_tmp.append(float(line_list[-2]) / 255)
            #model_x_tmp.append(float(line_list[-1]) / 255)
            model_x_tmp.append(float(line_list[2]) / 100)
            model_x_tmp.append(float(line_list[4]) / 1000)
            model_x_tmp.append(float(line_list[5]) / 250)
            model_x_current = model_x_tmp.copy()
            model_x_power = model_x_tmp.copy()
            model_x_current.append(float(line_list_past[0]) / 1000000)
            model_x_power.append(((float(line_list_past[0]) / 1000000) * (float(line_list_past[3]) / 1000000)))
            random_pass = random.randrange(0,4)
            if random_pass == 0:
                test_x_current.append(model_x_current)
                test_x_power.append(model_x_power)
                test_y_current.append(float(line_list[0]) / 1000000)
                test_y_power.append(((float(line_list[0]) / 1000000) * (float(line_list[3]) / 1000000)))
            else:
                train_x_current.append(model_x_current)
                train_x_power.append(model_x_power)
                train_y_current.append(float(line_list[0]) / 1000000)
                train_y_power.append(((float(line_list[0]) / 1000000) * (float(line_list[3]) / 1000000)))
    return test_x_current, train_x_current, test_x_power, train_x_power, test_y_current, train_y_current, test_y_power, train_y_power

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
