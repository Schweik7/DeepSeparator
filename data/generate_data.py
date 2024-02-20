import numpy as np
import math
import os

# Author: Haoming Zhang
# The code here not only include data importing,
# but also data standardization and the generation of analog noise signals

#  定义获取RMS（均方根值）的函数
def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

# 定义函数用于随机打乱信号，增加数据多样性
def random_signal(signal, combin_num):
    random_result = []
    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])  # 重新排列给定数组
        shuffled_dataset = signal[random_num, :]  # 单纯随机乱序EEG而已
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])  # 从结果来看和上一段代码没什么区别
        random_result.append(shuffled_dataset)

    random_result = np.array(random_result)  # combin有多少，最后的结果就重复了多少次的乱序拼接

    return random_result

# 定义标准化函数
def standardize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    standardized_data = (data - mu) / sigma
    return standardized_data

# 定义数据准备和处理函数
def data_prepare(EEG_all, noise_all, combin_num, train_num, val_num, test_num):
    # 分割数据集
    split1 = train_num
    split2 = train_num + val_num
    eeg_train, eeg_val, eeg_test = np.split(EEG_all, [split1, split2])
    noise_train, noise_val, noise_test = np.split(noise_all, [split1, split2])
    
    # 应用随机信号函数
    EEG_train = random_signal(eeg_train, combin_num)
    EEG_val = random_signal(eeg_val, combin_num)
    EEG_test = random_signal(eeg_test, combin_num)
    NOISE_train = random_signal(noise_train, combin_num)
    NOISE_val = random_signal(noise_val, combin_num)
    NOISE_test = random_signal(noise_test, combin_num)
    
    # 生成有噪声的EEG数据
    def generate_noisy_data(EEG_data, NOISE_data, SNR_dB_range=(-7, 2)):
        noisy_data_list = []
        clean_data_list = []
        for i in range(EEG_data.shape[0]):
            eeg = EEG_data[i].reshape(-1)
            noise = NOISE_data[i].reshape(-1)
            SNR_dB = np.random.uniform(SNR_dB_range[0], SNR_dB_range[1])
            SNR = 10 ** (SNR_dB / 10)
            rms_eeg = get_rms(eeg)
            rms_noise = get_rms(noise)
            scaled_noise = noise * (rms_eeg / (rms_noise * SNR))
            noisy_eeg = eeg + scaled_noise
            noisy_data_list.append(noisy_eeg)
            clean_data_list.append(eeg)
        return np.array(noisy_data_list), np.array(clean_data_list)
    
    # 处理训练、验证和测试集
    train_input, train_output = generate_noisy_data(EEG_train, NOISE_train)
    val_input, val_output = generate_noisy_data(EEG_val, NOISE_val)
    test_input, test_output = generate_noisy_data(EEG_test, NOISE_test)
    
    # 标准化数据
    train_input = standardize(train_input)
    val_input = standardize(val_input)
    test_input = standardize(test_input)
    
    # 返回处理后的数据集
    return train_input, train_output, val_input, val_output, test_input, test_output


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    
    train_num = 3000  # 训练集大小
    val_num = 500    # 验证集大小
    test_num = 500   # 测试集大小
    combin_num = 10  # 混合次数

    # 加载EEG和EOG数据
    EEG_all = np.load("EEG_all_epochs.npy")
    EOG_all = np.load("EOG_all_epochs.npy")
    
    # 调用data_prepare函数准备数据
    train_input, train_output, val_input, val_output, test_input, test_output = data_prepare(EEG_all, EOG_all, combin_num, train_num, val_num, test_num)
    
    # 保存数据到文件
    np.save("./train_input.npy", train_input)
    np.save("./train_output.npy", train_output)
    np.save("./val_input.npy", val_input)
    np.save("./val_output.npy", val_output)
    np.save("./test_input.npy", test_input)
    np.save("./test_output.npy", test_output)


