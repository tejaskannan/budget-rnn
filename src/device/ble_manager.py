import pexpect
import time
import re
import subprocess
import signal
from collections import namedtuple
from enum import Enum, auto
from typing import Optional, List


DEFAULT_TIMEOUT = 5
MAX_RETRIES = 10
RETRY_WAIT = 0.1

DATA_CHAR = 'D'
PRED_CHAR = 'P'
CONTROL_CHAR = 'C'
START_CHAR = 'S'

NOTIFICATION_REGEX = re.compile('Notification handle = ([0-9x]+) value: ([0-9a-f ]+).*')
QueryResponse = namedtuple('QueryResponse', ['response_type', 'value'])
ResponseValue = namedtuple('ResponseValue', ['num_levels', 'stride', 'to_execute', 'prediction', 'voltage'])


class ResponseType(Enum):
    CONTROL = auto()
    PREDICTION = auto()


class BLEManager:

    def __init__(self, mac_addr: str, handle: int, hci_device: str = 'hci0'):
        self._mac_addr = mac_addr
        self._rw_handle = handle
        self._hci_device = hci_device

        self._is_connected = False
        self._gatt = None
        self._connection_handle = None

    @property
    def mac_address(self) -> str:
        return self._mac_addr

    @property
    def rw_handle(self) -> int:
        return self._rw_handle

    @property
    def connection_handle(self) -> Optional[int]:
        return self._connection_handle

    @property
    def hci_device(self) -> str:
        return self._hci_device

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def start(self, timeout: float = DEFAULT_TIMEOUT) -> bool:
        if self._is_connected:
            return True

        # Start the gatttool session
        init_cmd = 'gatttool -b {0} -i {1} -I'.format(self.mac_address, self.hci_device)

        self._gatt = pexpect.spawn(init_cmd, ignore_sighup=False)
        assert self._gatt is not None, 'Could not spawn process'

        self._gatt.expect(r'\[LE\]', timeout=timeout)
        self._gatt.delaybeforesend = None

        # Open the connection
        self._gatt.sendline('connect')

        retry_count = 0
        did_connect = False
        while not did_connect and retry_count < MAX_RETRIES:
            try:
                self._gatt.sendline('connect')
                self._gatt.expect(r'.*Connection successful.*\[LE\]>', timeout)
                did_connect = True
            except pexpect.TIMEOUT as ex:
                print('Connection timeout after {0:.3f} seconds. Reason: {1}'.format(timeout, ex))

                retry_count += 1
                time.sleep(RETRY_WAIT)

        if not did_connect:
            self._gatt.send('exit')
            return False

        # Get the hci handle
        hci_result = subprocess.check_output(['sudo', 'hcitool', '-i', self.hci_device, 'con'])
        hci_output = hci_result.decode()

        pattern = 'Connections:\n.*< LE {0} handle ([0-9]+) state.*'.format(self.mac_address)
        handle_match = re.match(pattern, hci_output)
        assert handle_match is not None, 'Could not match: {0}'.format(hci_output)

        self._connection_handle = int(handle_match.group(1))
        self._is_connected = True
        return True

    def stop(self):
        """
        Tears down the connection and exits the gatttool session.
        """
        if not self._is_connected:
            return

        assert self.connection_handle is not None and self._gatt is not None, 'Must call start() first'

        # We kill the connection using hcitool. For some reason, gatttool keeps the connection
        # open for a few seconds after we initiate the disconnect. We do not want this behavior
        # in our application. We use the hcitool connection handle here.
        subprocess.check_output(['sudo', 'hcitool', '-i', self.hci_device, 'ledc', str(self.connection_handle)])

        # Shutdown the gatttool session
        if self._gatt.isalive():
            self._gatt.sendline('exit')
            self._gatt.close()

        self._gatt = None
        self._is_connected = False
        self._connection_handle = None

    def send(self, value: str, timeout: float = DEFAULT_TIMEOUT):
        assert self._is_connected and self._gatt is not None, 'Must call start() first'

        retry_count = 0
        did_send = False
        while not did_send and retry_count < MAX_RETRIES:
            try:
                hex_string = ''.join('{0:02x}'.format(ord(char)) for char in value)
                write_cmd = 'char-write-cmd 0x{0:02x} {1}'.format(self.rw_handle, hex_string)

                self._gatt.sendline(write_cmd)
                self._gatt.expect(r'.*\[LE\]>', timeout)

                did_send = True
            except pexpect.TIMEOUT as ex:
                print('Write timeout after {0} seconds. Command: {1}. Reason: {2}'.format(timeout, write_cmd, ex))

                retry_count += 1
                time.sleep(RETRY_WAIT)

    def reset_device(self):
        self.send(value='R')

    def query(self, value: str, is_first: bool, timeout: float = DEFAULT_TIMEOUT) -> QueryResponse:
        assert self._is_connected and self._gatt is not None, 'Must call start() first'

        # Send the data
        header_char = START_CHAR if is_first else DATA_CHAR
        data_string = '{0}{1}'.format(header_char, value)
        self.send(value=data_string)

        # Receive the response. This either contains the number of levels OR
        # contains the prediction for the sequence. This determination is
        # based on the leading character in the response
        response: Optional[List[int]] = None
        retry_count = 0

        while response is None and retry_count < MAX_RETRIES:
            try:
                self._gatt.expect('Notification handle = .*? \r', timeout)
                response_string = self._gatt.after.decode()

                match = NOTIFICATION_REGEX.match(response_string)
                if match is None:
                    value = ResponseValue(num_levels=None,
                                          stride=None,
                                          to_execute=None,
                                          prediction=0,
                                          voltage=None)
                    return QueryResponse(response_type.PREDICTION, value=value)

                tokens = match.group(2).split()

                response = list(map(lambda t: int(t, 16), tokens))
            except pexpect.TIMEOUT as ex:
                print('Read timeout after {0} seconds. Reason: {1}'.format(timeout, ex))

                retry_count += 1
                time.sleep(RETRY_WAIT)

        # If we never receive anything, then we assume that the sequence is ended and the prediction is
        # This is a design decision--it allows the system to recover from transient failures.
        if response is None:
            value = ResponseValue(num_levels=None,
                                  stride=None,
                                  to_execute=None,
                                  prediction=0,
                                  voltage=None)
            return QueryResponse(response_type.PREDICTION, value=value)

        # Unpack the response
        type_char = chr(response[0])
        response_type = ResponseType.CONTROL if type_char == CONTROL_CHAR else ResponseType.PREDICTION

        if response_type == ResponseType.CONTROL:
            value = ResponseValue(num_levels=response[1],
                                  stride=response[2],
                                  to_execute=response[3],
                                  prediction=None,
                                  voltage=None)
        else:
            upper = '{0:08b}'.format(response[2])
            lower = '{0:08b}'.format(response[3])
            nAdc = int('{0}{1}'.format(upper, lower), 2)
            voltage = 2 * (nAdc * 2500) / 4096  # Device voltage in Volts

            value = ResponseValue(num_levels=None,
                                  stride=None,
                                  to_execute=None,
                                  prediction=response[1],
                                  voltage=voltage)

        return QueryResponse(response_type=response_type, value=value)
