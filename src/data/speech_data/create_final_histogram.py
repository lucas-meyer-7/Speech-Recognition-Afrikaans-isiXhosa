import pickle
import numpy as np
import matplotlib.pyplot as plt

def create_stuff(language):
    with open (f'duration_lists/fleurs_durations_{language}.ob', 'rb') as fp:
        fleurs = pickle.load(fp)
    with open (f'duration_lists/nchlt_durations_{language}.ob', 'rb') as fp:
        nchlt = pickle.load(fp)
    with open (f'duration_lists/hqtts_durations_{language}.ob', 'rb') as fp:
        hqtts = pickle.load(fp)

    all_durations = []
    all_durations.extend(fleurs)
    all_durations.extend(hqtts)
    all_durations.extend(nchlt)

    num_seconds = np.sum(all_durations)
    num_hours = (num_seconds/60.0)/60.0
    print(f"Total {language} hours: {num_hours}")

    plt.figure()
    #plt.xlim(0, 20)
    #plt.ylim(0, 350)
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.hist(all_durations, bins=50)
    plt.savefig(f"final_histogram_{language}.pdf")

create_stuff("af")
create_stuff("xh")
plt.show()
