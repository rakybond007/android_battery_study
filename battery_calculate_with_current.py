f = open("log_current_18", 'r')
currents = list(map(int, f.readlines()))

current_sum = sum(currents) * (1 / 10)

battery_capacity = 3712000

f2 = open("log_battery_level_18", 'r')
start_level = int(f2.readline())
end_level = int(f2.readline())

actual_val = battery_capacity * 3600 * ((start_level-end_level) / 100)
print(current_sum, actual_val)
