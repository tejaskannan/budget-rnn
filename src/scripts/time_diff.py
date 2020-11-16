from datetime import datetime, timedelta
from typing import List


sample_times = ['02:58:42', '00:35:14', '05:45:01', '00:52:06', '01:31:18', '05:42:47', '01:58:20']
skip_times = ['07:44:54', '01:32:27', '04:57:22', '01:29:30', '01:49:37', '17:18:57', '08:20:04']
phased_times = ['09:17:07', '01:30:43', '09:13:55', '02:16:28', '03:53:04', '18:04:21', '06:32:57']

sample_iters = [297.5, 146.4, 559.3, 176.5, 446.3, 511.2, 274.6]
skip_iters = [724.8, 320.9, 664.8, 311.4, 699.1, 1559.8, 988.3]
phased_iters = [803.6, 293.4,  1139.9, 461.2, 1355.2, 1591.9, 734.0]


def time_avg_diff(xs: List[str], ys: List[str]) -> float:
    x_times = list(map(lambda t: datetime.strptime(t, '%H:%M:%S'), xs))
    y_times = list(map(lambda t: datetime.strptime(t, '%H:%M:%S'), ys))

    base = datetime.strptime('00:00:00', '%H:%M:%S')

    diffs: List[float] = []
    for x, y in zip(x_times, y_times):
        x_sec = (x - base).total_seconds()
        y_sec = (y - base).total_seconds()

        diffs.append(y_sec / x_sec)

    return sum(diffs) / len(diffs)


def iters_avg_diff(xs: List[float], ys: List[float]) -> float:
    diffs: List[float] = []

    for x, y in zip(xs, ys):
        diff = y / x
        diffs.append(diff)

    return sum(diffs) / len(diffs)


skip_time_diff = time_avg_diff(sample_times, skip_times)
print('Skip Avg % Greater Time: {0:.3f}'.format(skip_time_diff))

phased_time_diff = time_avg_diff(sample_times, phased_times)
print('Phased Avg % Greater Time: {0:.3f}'.format(phased_time_diff))

skip_iters_diff = iters_avg_diff(sample_iters, skip_iters)
print('Skip Avg % Greater Iters: {0:.3f}'.format(skip_iters_diff))

phased_iters_diff = iters_avg_diff(sample_iters, phased_iters)
print('Phased Avg % Greater Iters: {0:.3f}'.format(phased_iters_diff))
