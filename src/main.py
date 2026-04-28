from utils.logger import logger
import utils.data_load as data_load
import utils.plot_utils as plt
if __name__ == "__main__":
    file_path = "E:/dataSet/DroneRFa/T0001_D00_S0000.mat"
    iq_two_ch = data_load.load_iq_signal(file_path)
    iq_ch0 = iq_two_ch[0]
    iq_ch1 = iq_two_ch[1]
    plt.plot_iq_frequency_domain(iq_ch0, 0, 150000000)

