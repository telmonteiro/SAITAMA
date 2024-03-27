import os, glob, numpy as np, matplotlib.pyplot as plt, pandas as pd

def stats_indice(star,cols,df):
    """
    Return pandas data frame with statistical data on the indice(s) given: max, min, mean, median, std and N (number of spectra)
    """
    df_stats = pd.DataFrame(columns=["star","indice","max","min","mean","median","std","time_span","N_spectra"])
    if len(cols) == 1:
            row = {"star":star,"column":cols,
                "max":max(df[cols]),"min":min(df[cols]),
                "mean":np.mean(df[cols]),"median":np.median(df[cols]),
                "std":np.std(df[cols]),"time_span":max(df["bjd"])-min(df["bjd"]),
                "N_spectra":len(df[cols])}
            df_stats.loc[len(df_stats)] = row
    elif len(cols) > 1:
        for i in cols:
            indices = df[df[i+"_Rneg"] < 0.001].index
            data = df.loc[indices, i]
            row = {"star": star, "indice": i,
                   "max": max(data), "min": min(data),
                   "mean": np.mean(data), "median": np.median(data),
                   "std": np.std(data), "time_span": max(df["bjd"]) - min(df["bjd"]),
                   "N_spectra": len(data)}
            df_stats.loc[len(df_stats)] = row

    else:
        print("ERROR: No columns given")
        df_stats = None
    
    return df_stats


target_save_name = "HD85512"
instr = "HARPS"
path = f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/df_{target_save_name}_{instr}.csv"

df = pd.read_csv(path)

cols = ['I_CaII', 'I_Ha06', 'I_NaI',"rv"]
stats_df = stats_indice(target_save_name,cols,df)
print(stats_df)
#stats_df.to_csv(folder_path+f"stats_{target_save_name}.csv")       

