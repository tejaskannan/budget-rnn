import re

STYLE = 'fast'
LINEWIDTH = 2
MARKER_SIZE = 5

LABEL_REGEX = re.compile(r'.*model-optimized-([^-]+)-([0-9\.]*)-.*test-log.*')
LABEL_FORMAT = '{0}, $\lambda = {1}$'
