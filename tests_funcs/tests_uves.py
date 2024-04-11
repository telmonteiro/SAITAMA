import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, glob
from util_funcs import read_bintable

stars_dic = {
    'HD209100': {'up': 0, 'down': 0.14},
    'HD160691': {'up': 0.1012, 'down': 1},
    'HD115617': {'up': 0.106, 'down': 1},
    'HD46375': {'up': 1, 'down': 0},  # no points
    'HD22049': {'up': 1, 'down': 0},
    'HD102365': {'up': 0, 'down': 0.1},
    'HD1461': {'up': 0.103, 'down': 1},
    'HD16417': {'up': 0.103, 'down': 1},
    'HD10647': {'up': 1, 'down': 0},
    'HD13445': {'up': 1, 'down': 0},
    'HD142A': {'up': 1, 'down': 0},
    'HD108147': {'up': 1, 'down': 0},
    'HD16141': {'up': 0.101, 'down': 1},
    'HD179949': {'up': 1, 'down': 0},
    'HD47536': {'up': 1, 'down': 0}
}

stars = stars_dic.keys()

for star in stars:
    file_uves = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_UVES/", f"df_stats_{star}.fits"))
    if file_uves == []:
        continue
    df_uves, hdr = read_bintable(file_uves[0], print_info=False)
    file_harps = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_HARPS/", f"df_stats_{star}.fits"))
    df_harps, hdr_harps = read_bintable(file_harps[0], print_info=False)

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    fig.suptitle(star, fontsize=14)

    outlier_up = stars_dic[star]['up']
    outlier_down = stars_dic[star]['down']

    mask = (df_uves['I_Ha06'] < outlier_down) & (df_uves['I_Ha06'] > outlier_up)

    ax1.scatter(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'SNR'], label=f'I_Ha06 >= {outlier_down}')
    ax1.scatter(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'SNR'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax1.set_xlabel("BJD $-$ 2450000 [days]"); ax1.set_ylabel("SNR"); ax1.legend()

    ax2.scatter(df_uves.loc[~mask, 'SNR'], df_uves.loc[~mask, 'rv'], label=f'I_Ha06 >= {outlier_down}')
    ax2.scatter(df_uves.loc[mask, 'SNR'], df_uves.loc[mask, 'rv'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax2.set_xlabel("SNR"); ax2.set_ylabel("rv"); ax2.legend()

    ax3.errorbar(df_uves.loc[~mask, 'SNR'], df_uves.loc[~mask, 'I_Ha06'], df_uves.loc[~mask, 'I_Ha06_err'], fmt='k.', label=f'I_Ha06 >= {outlier_down}')
    ax3.errorbar(df_uves.loc[mask, 'SNR'], df_uves.loc[mask, 'I_Ha06'], df_uves.loc[mask, 'I_Ha06_err'], fmt='r.', label=f'I_Ha06 < {outlier_down}')
    ax3.set_xlabel("SNR"); ax3.set_ylabel("I_Ha06"); ax3.legend()

    ax4.errorbar(df_harps['bjd'] - 2450000, df_harps['I_Ha06'], df_harps['I_Ha06_err'], fmt='k.', label="HARPS")
    ax4.errorbar(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'I_Ha06'], df_uves.loc[~mask, 'I_Ha06_err'], fmt='b.', label='UVES')
    ax4.errorbar(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'I_Ha06'], df_uves.loc[mask, 'I_Ha06_err'], fmt='r.',label="UVES outlier")
    ax4.legend()
    ax4.set_ylabel("I_Ha06"); ax4.set_xlabel("BJD $-$ 2450000 [days]")

    plt.tight_layout()
    plt.savefig(f"uves_tests_fig/{star}.pdf",overwrite=True, format = 'pdf', dpi=300)

'''
HD1461: 1 outlier com RV perto de 0. o BJD é muito menor que os outros (8/2005)
HD16141: apenas 1 ponto não é outlier. outliers todos no mesmo dia (2004). RV simbad = -51, RV outlier = -22, não outlier = -76
HD102365: todos os outliers concentrados num dia, onde apenas 1 ponto não é outlier (6/2017, o penultimo dia)
HD115617: os outliers estão concentrados em 100<SNR<250, inconclusivo

todas as restantes não tem outliers (HD10647, HD13445, HD22049, HD47536, HD108147, HD179949), 
ou são todos outliers (HD102365, HD160691), ou é inconclusivo (HD209100, HD16417).

no caso de HD179949, os pontos estão no espaço de 2 dias

HD46375 não tem pontos nenhuns.

'''