import time
import threading
import subprocess
import re
import numpy as np


def integrate_trapezoid(y, x) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x=x))
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return float(np.sum((x_arr[1:] - x_arr[:-1]) * (y_arr[1:] + y_arr[:-1]) / 2.0))


class INA219HardwareDriver:
    """
    Driver I2C calibrado a 32V/2A basado exactamente en la implementación de ina219.py.
    """
    def __init__(
        self,
        i2c_bus: int = 1,
        addr: int = 0x40,
        bus_num: int = None,
        address: int = None,
        **kwargs
    ):
        self.addr = address if address is not None else addr
        self.bus_num = bus_num if bus_num is not None else i2c_bus
        self._cal_value = 4096
        self._power_lsb = 0.002  # 2mW por bit
        self._current_lsb = 0.1
        self.is_connected = False

        try:
            try:
                import smbus
                self.bus = smbus.SMBus(self.bus_num)
            except ImportError:
                import smbus2
                self.bus = smbus2.SMBus(self.bus_num)

            self.set_calibration_32V_2A()
            self.is_connected = True
        except Exception:
            self.is_connected = False

    def write_i2c_block(self, address: int, data: int):
        temp = [0, 0]
        temp[1] = data & 0xFF
        temp[0] = (data & 0xFF00) >> 8
        self.bus.write_i2c_block_data(self.addr, address, temp)

    def read_i2c_block(self, address: int) -> int:
        data = self.bus.read_i2c_block_data(self.addr, address, 2)
        return (data[0] << 8) | data[1]

    def set_calibration_32V_2A(self):
        self.write_i2c_block(0x05, self._cal_value)
        config = (0x01 << 13) | (0x03 << 11) | (0x0D << 7) | (0x0D << 3) | 0x07
        self.write_i2c_block(0x00, config)

    def get_bus_voltage_v(self) -> float:
        if not self.is_connected:
            return 0.0
        try:
            self.write_i2c_block(0x05, self._cal_value)
            return (self.read_i2c_block(0x02) >> 3) * 0.004
        except Exception:
            return 0.0

    def get_power_w(self) -> float:
        if not self.is_connected:
            return 0.0
        try:
            self.write_i2c_block(0x05, self._cal_value)
            val = self.read_i2c_block(0x03)
            if val > 32767:
                val -= 65536
            return max(0.0, val * self._power_lsb)
        except Exception:
            return 0.0

    def read_power_watts(self) -> float:
        return self.get_power_w()

    def read_power(self) -> float:
        return self.get_power_w()


class RPi5PMICReader:
    """
    Lector de potencia del PMIC Renesas en Raspberry Pi 5 via vcgencmd.
    """
    def read_soc_power_watts(self) -> float:
        try:
            out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], stderr=subprocess.DEVNULL).decode("utf-8")
            currents, voltages = {}, {}
            for line in out.splitlines():
                m_c = re.search(r'([A-Z0-9_]+)_A current\(\d+\)=([0-9\.]+)A', line)
                if m_c:
                    currents[m_c.group(1)] = float(m_c.group(2))
                m_v = re.search(r'([A-Z0-9_]+)_V volt\(\d+\)=([0-9\.]+)V', line)
                if m_v:
                    voltages[m_v.group(1)] = float(m_v.group(2))

            watts = 0.0
            for rail in ["VDD_CORE", "DDR_VDD2", "DDR_VDDQ", "1V8_SYS", "1V1_SYS", "0V8_SW"]:
                if rail in currents and rail in voltages:
                    watts += currents[rail] * voltages[rail]
            return round(watts, 3)
        except Exception:
            return 0.0

    def read_power_watts(self) -> float:
        return self.read_soc_power_watts()


class HardwarePowerProfiler:
    """
    Profiler HIL con soporte para Context Manager (with HardwareProfiler() as profiler:).
    """
    def __init__(
        self,
        power_watts: float = 7.0,
        ch1_addr: int = 0x40,
        ch2_addr: int = 0x41,
        sample_rate_hz: int = 50,
        **kwargs
    ):
        self.power_watts = power_watts
        self.fallback_power_w = power_watts
        self.sensor_ch1 = INA219HardwareDriver(addr=ch1_addr)
        self.sensor_ch2 = INA219HardwareDriver(addr=ch2_addr)
        self.sample_interval = 1.0 / sample_rate_hz

        self._running = False
        self._thread = None
        self.time_history = []
        self.power_history = []
        self.start_time = time.time()
        self.end_time = time.time()
        self.telemetry = {}

    def _sampling_loop(self):
        while self._running:
            now = time.time()
            rel_time = now - self.start_time

            p1 = self.sensor_ch1.get_power_w()
            p2 = self.sensor_ch2.get_power_w()
            total_w = p1 + p2

            if total_w <= 0.0:
                total_w = self.fallback_power_w

            self.time_history.append(rel_time)
            self.power_history.append(total_w)
            time.sleep(self.sample_interval)

    def start(self):
        self.time_history.clear()
        self.power_history.clear()
        self.start_time = time.time()
        self.end_time = self.start_time
        self._running = True
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        return self.stop_and_get_telemetry()

    def stop_and_get_telemetry(self) -> dict:
        self._running = False
        self.end_time = time.time()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        dur = max(0.001, self.end_time - self.start_time)

        if len(self.power_history) < 2:
            total_j = self.fallback_power_w * dur
            avg_p = self.fallback_power_w
        else:
            total_j = integrate_trapezoid(self.power_history, self.time_history)
            avg_p = float(np.mean(self.power_history))

        self.telemetry = {
            "total_joules": total_j,
            "energy_consumed_j": total_j,
            "energy_total_j": total_j,
            "avg_power_w": avg_p,
            "duration_s": dur,
            "instant_p1_w": self.sensor_ch1.get_power_w(),
            "instant_p2_w": self.sensor_ch2.get_power_w()
        }
        return self.telemetry

    def get_telemetry(self) -> dict:
        if not self.telemetry or self._running:
            return self.stop_and_get_telemetry()
        return self.telemetry

    def estimate_energy_joules(self) -> float:
        tel = self.get_telemetry()
        return float(tel.get("total_joules", 0.0))

    def get_energy_joules(self) -> float:
        return self.estimate_energy_joules()

    @property
    def total_joules(self) -> float:
        return self.estimate_energy_joules()

    @property
    def energy_joules(self) -> float:
        return self.estimate_energy_joules()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_and_get_telemetry()


HardwareProfiler = HardwarePowerProfiler
INA219Reader = INA219HardwareDriver
