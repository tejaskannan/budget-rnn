import re

STYLE = 'seaborn-ticks'
LINEWIDTH = 2
MARKER_SIZE = 6
CAPSIZE = 3
TINY_FONT = 9
SMALL_FONT = 12
NORMAL_FONT = 14
LARGE_FONT = 18

LABEL_REGEX = re.compile(r'.*model-optimized-([^-]+)-([0-9\.]*)-.*test-log.*')
