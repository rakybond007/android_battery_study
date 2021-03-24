import time
import os
from threading import Thread
import random
cpu_cores = 8
device_name = "4"
Finish_flag = False
idx_count = 0
last_cpu_infos = [[0 for i in range(10)] for j in range(cpu_cores)]
last_cpu_total = [0 for i in range(10)]
cpu_utilization = []
app_webtoon = "com.nhn.android.webtoon/.splash.SplashActivity"
app_snow = "com.campmobile.snow/com.linecorp.b612.android.activity.ActivityCamera"
app_twitch = "tv.twitch.android.app/.core.LandingActivity"
def exec(command):
  global device_name
  os.system(f"adb -s $PIXEL4_wifi shell \"" + command +"\"")

def exec_with_output(command):
  out = os.popen(f"adb -s $PIXEL4_wifi shell \"" + command +"\"").read()
  ret_status = out.strip()
  return ret_status

def check_screen_on():
  status = exec_with_output("dumpsys display | grep mScreenState")
  status = status.strip().split('=')[1]
  if status == "OFF":
    return False
  else:
    return True

def wakeup():
  exec("input keyevent 26;input swipe 500 1000 300 300")

def launch_webtoon():
  exec(f"am start -a android.intent.action.VIEW -n {app_webtoon}")

def launch_snow():
    exec(f"am start -a android.intent.action.VIEW -n {app_snow}")

def select_filter():
    # open effects
    exec("input tap 400 2470")
    time.sleep(1)
    # Select category
    exec("input tap 650 1880")
    time.sleep(1)
    # new effects
    filters = [(160, 2340), (440, 2340), (730, 2340), (1020, 2340), (1280, 2340)]
    #which_effect_y = random.randrange(205, 250) * 10
    #which_effect_x = random.randrange(15,130) * 10
    filter_idx = random.randrange(0,5)
    which_effect_y = filters[filter_idx][1]
    which_effect_x = filters[filter_idx][0]
    exec(f"input tap {which_effect_x} {which_effect_y}")
    time.sleep(2)
    # cancel button
    exec("input tap 320 2970")
    time.sleep(2)

def shot(num):
    global idx_count
    for i in range(num):
        if i % 10 == 0:
            select_filter()
        # shot button
        exec("input tap 710 2500")
        time.sleep(2)
        # back button
        exec("input tap 150 2500")
        # ready
        idx_count += 1
        time.sleep(4)

def webtoon_read():
  # 금요일
  exec("input tap 850 900")
  time.sleep(2)
  # 웹툰 썸네일 클릭
  exec("input tap 700 1400")
  time.sleep(2)
  exec("input tap 1420 190")
  time.sleep(2)
  exec("input tap 1100 330")
  # 웹툰 켜져 있을 거임
  for read in range(30):
    for i in range(10):
      drag()
    exec("input tap 700 1500")
    time.sleep(0.5)
    exec("input tap 1350 2800")

def webtoon_read_cuttoon():
    # 일요일
    exec("input tap 1110 900")
    time.sleep(2)
    # 대학일기
    exec("input tap 270 2650")
    time.sleep(2)
    # 첫화보기
    exec("input tap 1420 190")
    time.sleep(2)
    exec("input tap 1100 330")
    # 예고편 패스
    time.sleep(2)
    exec("input tap 1350 2800")
    for read in range(30):
        for i in range(10):
            drag_right()
        exec("input tap 1350 2800")

def drag():
  exec("input swipe 720 2300 1050 1050")
  time.sleep(2)

def drag_right():
    exec("input swipe 1000 1400 400 1400")
    time.sleep(2)

def init_cpu_util():
    global last_cpu_infos
    global last_cpu_total
    procstat = exec_with_output("cat /proc/stat").split('\n')
    procstat_lines = procstat[1:9]
    total_time = list(map(int, procstat[0].strip().split(' ')[2:]))
    last_cpu_total = total_time
    for idx in range(cpu_cores):
        times = list(map(int, procstat_lines[idx].strip().split(' ')[1:]))
        last_cpu_infos[idx] = times

def read_cpu_util():
    global last_cpu_infos
    global cpu_cores
    #global cpu_utilization
    global last_cpu_total
    cpu_infos = []
    cpu_utilization = []
    #f = open("/proc/stat", 'r')
    procstat = exec_with_output("cat /proc/stat").split('\n')
    procstat_lines = procstat[1:9]
    #for each_core in procstat_lines:
    total_time = list(map(int, procstat[0].strip().split(' ')[2:]))
    total_diff = 0
    idle_diff = 0
    for j in range(8):
        diff = total_time[j] - last_cpu_total[j]
        total_diff += diff
        # idle iowait
        if j == 3 or j == 4:
            idle_diff += diff
        last_cpu_total[j] = total_time[j]
    cpu_utilization_total = 100.0 - (idle_diff / total_diff * 100)
    for idx in range(cpu_cores):
        # user nice system idle iowait irq softirq steal guest guest_nice
        #times = list(map(int, each_core.strip().split(' ')[1:]))
        times = list(map(int, procstat_lines[idx].strip().split(' ')[1:]))
        total_diff = 0
        idle_diff = 0
        for j in range(8):
            #diff = times[j] - last_cpu_infos[procstat_lines.index(each_core)][j]
            diff = times[j] - last_cpu_infos[idx][j]
            total_diff += diff
            # idle iowait
            if j == 3 or j == 4:
                idle_diff += diff
        if total_diff == 0:
            cpu_utilization.append(0)
        else:
            cpu_utilization.append(100.0 - (idle_diff / total_diff * 100))
        last_cpu_infos[idx] = times
    #print(cpu_utilization)          
    return cpu_utilization_total, cpu_utilization

def read_gpu_util():
    #f = open("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage", 'r')
    gpu_util = exec_with_output("cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage").strip().split(' ')[0]
    #val = int(f.read().strip().split(' ')[0])
    #print("gpu util : " + str(val))
    return gpu_util

def read_cpu_frequency():
    cluster0_file_path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq"
    cluster1_file_path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_cur_freq"
    cluster2_file_path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_cur_freq"
    f0 = open(cluster0_file_path, 'r')
    val0 = int(f0.read().strip().split(' ')[0])
    f1 = open(cluster1_file_path, 'r')
    val1 = int(f1.read().strip().split(' ')[0])
    f2 = open(cluster2_file_path, 'r')
    val2 = int(f2.read().strip().split(' ')[0])
    #print(val0, val1, val2)
    return (val0,val1,val2)

def read_frequency():
    cluster0 = exec_with_output("cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq").strip()
    cluster1 = exec_with_output("cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_cur_freq").strip()
    cluster2 = exec_with_output("cat /sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_cur_freq").strip()
    gpu = exec_with_output("cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")
    return [cluster0,cluster1,cluster2,gpu]

def read_max_brightness():
    f = open("/sys/class/backlight/panel0-backlight/max_brightness", 'r')
    val = int(f.read().strip())
    print("max brightness : " + str(val))
    f.close()

def read_brightness():
    f = open("/sys/class/backlight/panel0-backlight/brightness", 'r')
    val = int(f.read().strip())
    print("brightness : " + str(val))
    f.close()

def read_current():
    f = open("/sys/class/power_supply/battery/current_now", 'r')
    val = int(f.read().strip())
    f.close()
    return val

def read_temp():
    f = open("/sys/class/power_supply/battery/temp", 'r')
    val = int(f.read().strip())
    f.close()
    return val

def read_current_pc():
    current = exec_with_output("cat /sys/class/power_supply/battery/current_now")
    return current

def read_voltage_ocv_pc():
    voltage = exec_with_output("cat /sys/class/power_supply/battery/voltage_ocv")
    return voltage

def read_temp_pc():
    temp = exec_with_output("cat /sys/class/power_supply/battery/temp")
    return temp

def read_battery_level():
    battery_level = exec_with_output("dumpsys battery").split('\n')[10].split(':')[1]
    return int(battery_level)

def log_current_pc(frequency, log_num):
    global Finish_flag
    current_file = open(f"log_current_{log_num}", 'a+')
    while not Finish_flag:
        current = read_current_pc()
        current_file.write(str(current)+'\n')
        time.sleep(frequency)
    current_file.close()

def log_voltage_pc(frequency, log_num):
    global Finish_flag
    voltage_file = open(f"log_voltage_{log_num}", 'a+')
    while not Finish_flag:
        voltage = read_voltage_ocv_pc()
        voltage_file.write(str(voltage)+'\n')
        time.sleep(frequency)
    voltage_file.close()

def runtime_log_current_voltage_temp(frequency, log_num, log_num2, app_name):
    global Finish_flag
    cvt_file = open(f"cvt_logs/log_cvt_{app_name}_{log_num}_{log_num2}", 'a+')
    while not Finish_flag:
        start = time.time()
        voltage = read_voltage_ocv_pc()
        current = read_current_pc()
        temp = read_temp_pc()
        end = time.time()
        cvt_file.write(current + ' ' + voltage + ' ' + temp + ' ' + str(start) + ' ' + str(end) + '\n')
        time.sleep(max(0, frequency - (end - start)))
    cvt_file.close()

def runtime_log_util_and_freq(frequency, log_num, log_num2, app_name):
    global Finish_flag
    util_file = open(f"util_logs/log_util_{app_name}_{log_num}_{log_num2}", 'a+')
    while not Finish_flag:
        start = time.time()
        cpu_util_total, utils = read_cpu_util()
        gpu_util = read_gpu_util()
        freqs = read_frequency()
        utils.insert(0, cpu_util_total)
        utils.insert(1, gpu_util)
        utils.extend(freqs)
        end = time.time()
        utils.extend([start,end])
        util_file.write(str(utils)+'\n')
        time.sleep(max(0, frequency - (end - start)))
    util_file.close()


def runtime_log(frequency, log_num):
    global Finish_flag
    current_file = open(f"log_current_{log_num}", 'a+')
    battery_level_file = open(f"log_battery_level_{log_num}", 'a+')
    start_level = read_battery_level()
    battery_level_file.write(str(start_level)+'\n')
    while not Finish_flag:
        current = read_current_pc()
        current_file.write(str(current)+'\n')
        time.sleep(frequency)
    end_level = read_battery_level()
    battery_level_file.write(str(end_level)+'\n')
    current_file.close()
    battery_level_file.close()

def runtime_log_with_voltage(frequency, log_num):
    global Finish_flag
    battery_level_file = open(f"log_battery_level_{log_num}", 'a+')
    start_level = read_battery_level()
    battery_level_file.write(str(start_level)+'\n')
    th_current = Thread(target=log_current_pc, args=(frequency, log_num, 2, "snow",))
    th_voltage = Thread(target=log_voltage_pc, args=(frequency, log_num, 2, "snow",))
    th_current.start()
    th_voltage.start()
    th_voltage.join()
    th_current.join()
    end_level = read_battery_level()
    battery_level_file.write(str(end_level)+'\n')
    battery_level_file.close()

def log():
  os.system("adb -s $PIXEL4_wifi shell dumpsys batterystats")

def save(num, repeat_num):
  global log_dir
  os.system(f"adb -s $PIXEL4_wifi pull /data/local/tmp/log batterystats_logs/{num}/log_{repeat_num}")

def batterystats_log_save(num):
    global Finish_flag
    global idx_count
    os.system(f"mkdir batterystats_logs/{num}")
    while not Finish_flag:
        if idx_count % 100 == 0:
            log()
            save(num, (idx_count // 100))
        time.sleep(6)
        idx_count += 1

def experiment_webtoon():
    global Finish_flag
    launch_webtoon()
    time.sleep(5)
    webtoon_read()
    Finish_flag = True
    quit_webtoon()

def experiment_webtoon_cuttoon():
    global Finish_flag
    launch_webtoon()
    time.sleep(5)
    webtoon_read_cuttoon()
    Finish_flag = True
    quit_webtoon()

def quit_webtoon():
  exec("am force-stop com.nhn.android.webtoon")

def quit_twitch():
    exec("am force-stop tv.twitch.android.app")

def experiment_snow(shot_num):
    global Finish_flag
    launch_snow()
    time.sleep(5)
    select_filter()
    shot(shot_num)
    Finish_flag = True
    quit_snow()

def experiment_sleep(howmuch):
    global Finish_flag
    time.sleep(howmuch)
    Finish_flag = True
    quit_twitch()

def quit_snow():
    exec("am force-stop com.campmobile.snow")

'''
for i in range(100):
    current = read_current_pc()
    print("current = ", current)
    time.sleep(0.1)
exit(1)
'''

'''
for i in range(2):
    Finish_flag = False
    th_util = Thread(target=runtime_log_util_and_freq, args=(0.3,i,))
    th_cvt = Thread(target=runtime_log_current_voltage_temp, args=(0.3,i,))
    th_cvt.start()
    th_util.start()
    time.sleep(10)
    Finish_flag = True
    th_util.join()
    th_cvt.join()

exit(1)
'''

init_cpu_util()
time.sleep(1)

for i in range(1):
    chk = check_screen_on()
    if not chk:
        wakeup()
    Finish_flag = False
    #shot_num = 2500
    th_experiment = Thread(target=experiment_sleep, args=(22000,))
    th_util = Thread(target=runtime_log_util_and_freq, args=(0.3,6,1,"twitch",))
    th_cvt = Thread(target=runtime_log_current_voltage_temp, args=(0.3,6,1,"twitch",))
    th_batterystats = Thread(target=batterystats_log_save, args=(3,))
    th_experiment.start()
    th_util.start()
    th_cvt.start()
    th_batterystats.start()
    th_batterystats.join()
    th_cvt.join()
    th_util.join()
    th_experiment.join()

time.sleep(10)
'''
init_cpu_util()
time.sleep(1)
for i in range(5):
    chk = check_screen_on()
    if not chk:
        wakeup()
    Finish_flag = False
    th_experiment = Thread(target=experiment_webtoon_cuttoon)
    th_util = Thread(target=runtime_log_util_and_freq, args=(0.3,23+i,))
    th_cvt = Thread(target=runtime_log_current_voltage_temp, args=(0.3,23+i,))
    th_experiment.start()
    th_util.start()
    th_cvt.start()
    th_cvt.join()
    th_util.join()
    th_experiment.join()
    time.sleep(5)
'''
'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Execute the experiment')
  parser.add_argument('--time', '-t', type=int, 
                          help='time to play the video', default=10)

  parser.add_argument('--video_link', '-l', type=str, 
      help='time to play the video', default="https://www.youtube.com/watch?v=sj4FmBseC7M")

  parser.add_argument('--brightness', '-b', type=int,
      help="brightness 0-250", default=200)

  parser.add_argument('--resolution', '-r', type=int,
      help="resolution", default=1080)

  parser.add_argument('--volume', '-v', type=int,
      help="volume", default=0);

  parser.add_argument('--repeat', type=int,
      help="# of repeating", default=1)
      #res = os.system("adb shell dumpsys display | grep \"mScreenState\"")
      #print(res)
      #exit(1)
      args = parser.parse_args()
      read_cpu_frequency()
      exit(1)
      for i in range(20):
          time.sleep(2)
          read_cpu_util()
          #read_max_brightness()
          #read_brightness()
          #read_gpu_busy()
'''