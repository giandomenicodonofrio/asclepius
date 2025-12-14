from src.utils.ecg_math_generation import ecg_wav_generator
import matplotlib.pyplot as plt
import numpy as np 

def load():
    wave = np.load("./resources/synthetic_ecg.npy")
    plt.plot(range(len(wave)),wave)
    plt.show()

def main():
    tse=ecg_wav_generator(hrate=1.5,
                APwav=0.25,DPwav=0.09,TPwav=0.16,
                ATwav=0.35, DTwav=0.142,TTwav=0.2,
                AUwav=0.035, DUwav=0.0476, TUwav=0.433,
                AQwav=0.025, DQwav=0.066,TQwav=0.166,
                ASwav=0.25, DSwav=0.066,TSwav=0.08,
                AQRSwav=1.6, DQRSwav=0.11)    
    
    scale=0.4
    tsh=[x*scale for x in tse[0]]
    tsv=[x*scale for x in tse[1]]
    plt.plot(tsv,tsh)
    plt.show()

    np.save("./resources/synthetic_ecg.npy", np.asanyarray(tsh))


if __name__ == "__main__":
    main()
