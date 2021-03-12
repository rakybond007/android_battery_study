import time
import os
from threading import Thread
import random
cpu_cores = 8
device_name = "4"
Finish_flag = False
last_cpu_infos = [[0 for i in range(10)] for j in range(cpu_cores)]
cpu_utilization = []
app_webtoon = "com.nhn.android.webtoon/.splash.SplashActivity"
app_snow = "com.campmobile.snow/com.linecorp.b612.android.activity.ActivityCamera"
def exec(command):
  global device_name
  os.system(f"adb shell \"" + command +"\"")

def exec_with_output(command):
  out = os.popen(f"adb shell \"" + command +"\"").read()
  ret_status = out.strip()
  return ret_status

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
    which_effect_y = random.randrange(205, 250) * 10
    which_effect_x = random.randrange(15,130) * 10
    exec(f"input tap {which_effect_x} {which_effect_y}")
    time.sleep(1)
    # cancel button
    exec("input tap 320 2970")
    time.sleep(1)

def shot(num):
    for i in range(num):
        if i % 5 == 0:
            select_filter()
        # shot button
        exec("input tap 710 2500")
        time.sleep(1)
        # back button
        exec("input tap 150 2500")
        # ready
        time.sleep(5)

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
  for read in range(300):
    for i in range(20):
      drag()
    exec("input tap 700 1500")
    time.sleep(0.5)
    exec("input tap 1350 2800")

def drag():
  exec("input swipe 720 2300 1050 1050")
  time.sleep(2)

def read_cpu_util():
    global last_cpu_infos
    global cpu_cores
    global cpu_utilization
    cpu_infos = []
    f = open("/proc/stat", 'r')
    procstat_lines = f.readlines()[1:9]
    for each_core in procstat_lines:
        # user nice system idle iowait irq softirq steal guest guest_nice
        times = list(map(int, each_core.strip().split(' ')[1:]))
        total_diff = 0
        idle_diff = 0
        for j in range(8):
            diff = times[j] - last_cpu_infos[procstat_lines.index(each_core)][j]
            total_diff += diff
            # idle iowait
            if j == 3 or j == 4:
                idle_diff += diff
        if last_cpu_infos[0][0] != 0:
            cpu_utilization.append(100.0 - (idle_diff / total_diff * 100))
        else:
            cpu_utilization.append(0)
        last_cpu_infos[procstat_lines.index(each_core)] = times
    print(cpu_utilization)          

def read_gpu_util():
    f = open("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage", 'r')
    val = int(f.read().strip().split(' ')[0])
    print("gpu util : " + str(val))

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
    print(val0, val1, val2)

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

def read_current_pc():
    current = exec_with_output("cat /sys/class/power_supply/battery/current_now")
    return current

def read_battery_level():
    battery_level = exec_with_output("dumpsys battery").split('\n')[10].split(':')[1]
    return int(battery_level)

def runtime_log(frequency):
    global Finish_flag
    current_file = open("/data/local/tmp/log_current", 'w')
    battery_level_file = open("/data/local/tmp/log_battery_level", 'w')
    start_level = read_battery_level()
    battery_level_file.write(str(start_level)+'\n')
    while not Finish_flag:
        current = read_current()
        current_file.write(str(current)+'\n')
        time.sleep(frequency)
    end_level = read_battery_level()
    battery_level_file.write(str(end_level)+'\n')
    current_file.close()
    battery_level_file.close()

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

def experiment_webtoon():
    global Finish_flag
    launch_webtoon()
    time.sleep(5)
    webtoon_read()
    Finish_flag = True
    quit_webtoon()

def quit_webtoon():
  exec("am force-stop com.nhn.android.webtoon")

def experiment_snow(shot_num):
    global Finish_flag
    launch_snow()
    time.sleep(5)
    select_filter()
    shot(shot_num)
    Finish_flag = True
    quit_snow()

def quit_snow():
    exec("am force-stop com.campmobile.snow")

for i in range(100):
    current = read_current_pc()
    print("current = ", current)
    time.sleep(0.1)
exit(1)

wakeup()
for i in range(2):
    Finish_flag = False
    shot_num = 300
    th_experiment = Thread(target=experiment_snow, args=(shot_num,))
    th_log = Thread(target=runtime_log, args=(0.3*(i+1), 6+i,))
    th_experiment.start()
    th_log.start()
    th_log.join()
    th_experiment.join()
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