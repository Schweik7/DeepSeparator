import numpy as np
import math
import os
os.chdir(os.path.dirname(__file__))

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


# 混合EEG和EOG的信号作为有噪声的EEG数据
def data_prepare(EEG_all, noise_all, combin_num, train_num, test_num):
    # The code here not only include data importing,
    # but also data standardization and the generation of analog noise signals

    eeg_train, eeg_test = EEG_all[0:train_num, :], EEG_all[train_num:train_num + test_num, :]
    noise_train, noise_test = noise_all[0:train_num, :], noise_all[train_num:train_num + test_num, :]

    # 难道combin是为了一个EEG数据混合多次EOG实现一个样本的多种降噪？
    EEG_train = random_signal(eeg_train, combin_num).reshape(combin_num * eeg_train.shape[0], eeg_train.shape[1])
    NOISE_train = random_signal(noise_train, combin_num).reshape(combin_num * noise_train.shape[0],
                                                                 noise_train.shape[1])

    # print(EEG_train.shape)
    # print(NOISE_train.shape)

    #################################  simulate noise signal of training set  ##############################

    # create random number between -10dB ~ 2dB
    # SNR is signal-to-noise ratio
    SNR_train_dB = np.random.uniform(-7, 2, (EEG_train.shape[0]))
    # print(SNR_train_dB.shape)
    SNR_train = 10 ** (0.1 * (SNR_train_dB))

    # combin eeg and noise for training set 
    noiseEEG_train = []
    NOISE_train_adjust = []
    for i in range(EEG_train.shape[0]):
        eeg = EEG_train[i].reshape(EEG_train.shape[1])
        noise = NOISE_train[i].reshape(NOISE_train.shape[1])

        coe = get_rms(eeg) / (get_rms(noise) * SNR_train[i])
        noise = noise * coe
        neeg = noise + eeg  # 直接相加添加噪声

        NOISE_train_adjust.append(noise)
        noiseEEG_train.append(neeg)

    noiseEEG_train = np.array(noiseEEG_train)
    NOISE_train_adjust = np.array(NOISE_train_adjust)

    # variance for noisy EEG
    EEG_train_end_standard = []
    noiseEEG_train_end_standard = []

    for i in range(noiseEEG_train.shape[0]):
        # Each epochs divided by the standard deviation
        eeg_train_all_std = EEG_train[i] / np.std(noiseEEG_train[i])
        EEG_train_end_standard.append(eeg_train_all_std)

        noiseeeg_train_end_standard = noiseEEG_train[i] / np.std(noiseEEG_train[i])
        noiseEEG_train_end_standard.append(noiseeeg_train_end_standard)

    noiseEEG_train_end_standard = np.array(noiseEEG_train_end_standard)
    EEG_train_end_standard = np.array(EEG_train_end_standard)

    #################################  simulate noise signal of test  ##############################

    SNR_test_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_test = 10 ** (0.1 * (SNR_test_dB))

    eeg_test = np.array(eeg_test)
    noise_test = np.array(noise_test)

    # combin eeg and noise for test set 
    EEG_test = []
    noise_EEG_test = []
    for i in range(10):

        noise_eeg_test = []
        for j in range(eeg_test.shape[0]):
            eeg = eeg_test[j]
            noise = noise_test[j]

            coe = get_rms(eeg) / (get_rms(noise) * SNR_test[i])
            noise = noise * coe
            neeg = noise + eeg

            noise_eeg_test.append(neeg)

        EEG_test.extend(eeg_test)
        noise_EEG_test.extend(noise_eeg_test)

    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)

    # std for noisy EEG
    EEG_test_end_standard = []
    noiseEEG_test_end_standard = []
    std_VALUE = []
    for i in range(noise_EEG_test.shape[0]):
        # store std value to restore EEG signal
        std_value = np.std(noise_EEG_test[i])
        std_VALUE.append(std_value)

        # Each epochs of eeg and neeg was divide by the standard deviation
        eeg_test_all_std = EEG_test[i] / std_value
        EEG_test_end_standard.append(eeg_test_all_std)

        noiseeeg_test_end_standard = noise_EEG_test[i] / std_value
        noiseEEG_test_end_standard.append(noiseeeg_test_end_standard)

    std_VALUE = np.array(std_VALUE)
    noiseEEG_test_end_standard = np.array(noiseEEG_test_end_standard)
    EEG_test_end_standard = np.array(EEG_test_end_standard)

    # In order to facilitate the training and application of neural networks,
    # normalized the noisy EEG signal

    return noiseEEG_train_end_standard, EEG_train_end_standard, noiseEEG_test_end_standard, EEG_test_end_standard, std_VALUE


epochs = 1  # training epoch
batch_size = 1000  # training batch size
train_num = 3000  # how many trails for train
test_num = 400  # how many trails for test

# 主要为了增加数据数量，不同的EEG和噪声混合会产生更多的数据，同时也可增强模型鲁棒性，引入不同的噪声仍可还原信号
combin_num = 10  # combin EEG and noise ? times


EEG_all = np.load('EEG_all_epochs.npy')
EOG_all = np.load('EOG_all_epochs.npy')
EMG_all = np.load('EMG_all_epochs.npy')

EOGEEG_train_input, EOGEEG_train_output, EOGEEG_test_input, EOGEEG_test_output, test_std_VALUE = data_prepare(EEG_all, EOG_all, combin_num,
                                                                                  train_num, test_num)

EMGEEG_train_input, EMGEEG_train_output, EMGEEG_test_input, EMGEEG_test_output, test_std_VALUE = data_prepare(EEG_all, EMG_all, combin_num,
                                                                                  train_num, test_num)

train_input = np.vstack((EOGEEG_train_input, EMGEEG_train_input))
train_output = np.vstack((EOGEEG_train_output, EMGEEG_train_output))
test_input = np.vstack((EOGEEG_test_input, EMGEEG_test_input))
test_output = np.vstack((EOGEEG_test_output, EMGEEG_test_output))

np.save('./train_input.npy', train_input)
np.save('./train_output.npy', train_output)
np.save('./test_input.npy', test_input)
np.save('./test_output.npy', test_output)

np.save('./EOG_EEG_test_input.npy', EOGEEG_test_input)
np.save('./EOG_EEG_test_output.npy', EOGEEG_test_output)
np.save('./EMG_EEG_test_input.npy', EMGEEG_test_input)
np.save('./EMG_EEG_test_output.npy', EMGEEG_test_output)


