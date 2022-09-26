import pandas as pd 
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


from datetime import datetime, timedelta

## different attribution models
class AttributionModels:
    
    def __init__(self, df, time_col, conv_col):
        self.df = df.copy()
        self.time_col = time_col
        self.conv_col = conv_col

        
    def get_models(self,):
        return [func for func in dir(self, ) if callable(getattr(self, func)) and not func.startswith("__") and func.startswith("model")]
    
        
    def apply_models(self, *models):
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        
        for model in models:
            self.df[model] = self.df.groupby(self.conv_col)[self.time_col].apply(getattr(self, model))
        
        return self.df
     
    
    def model_first_click(self, timestamp):
        first_click_attribution = (timestamp == timestamp.min())
        return first_click_attribution.astype(int)

    def model_last_click(self, timestamp ):
        last_click_attribution = (timestamp == timestamp.max())
        return last_click_attribution.astype(int)

    def model_u_shape(self, timestamp,  u_val = 0.4):
        
        length_df = len(timestamp)
        
        first = (timestamp == timestamp.min())
        last =  (timestamp == timestamp.max())
        first_and_last = first + last 
        
        if length_df >2:
            first_and_last.where(lambda x: x==0, u_val, inplace = True)
            first_and_last.where(lambda x: x>0, (1-(2*u_val))/(length_df-2), inplace = True)

        if length_df ==2:
            first_and_last.where(lambda x: x==0, 0.5, inplace = True)

        if length_df ==1:
               first_and_last.where(lambda x: x==0, 1, inplace = True)

        return first_and_last.astype(float)

    def model_time_decay(self, timestamp ):
        weight_decay = (timestamp - timestamp.min()).dt.total_seconds()/(3600*24)
        weight_decay = weight_decay/weight_decay.sum()
        weight_decay.where(~weight_decay.isna(), 1, inplace = True)
        return weight_decay.astype(float)
    

    
## plots
def plot_ihc_scores(df_ihc_scores, show_nb = False):
    
    fig, ax = plt.subplots(figsize = (18,6))

    df_ihc_scores.plot(kind = "bar", ax =ax, width = 0.5, cmap ="flare", alpha = 0.9)

    ax.axhline(1, ax.get_xlim()[0], ax.get_xlim()[1], linestyle = "--", color = "brown", alpha = 0.7, lw = 2.5)

    ax.tick_params(rotation =0, labelsize = 14)
    ax.set_title("Channel ", fontsize= 18)
    ax.set_xlabel("")
    ax.legend(ncol = 3, prop = {"size":14}, framealpha = 0, loc = (0,1))
    
    if not show_nb:
        plt.close()
    else:
        plt.show()
   
    return fig


def attribution_distribution(df, channel, value, show_nb  = False):
    fig, ax = plt.subplots(figsize = (15,5))
        
    
    sns.histplot(x = df[df.channel_label == channel][value],
                stat = "percent",
                bins = 30,
                alpha= 0.4,
                kde =True,
                kde_kws = {"bw_adjust":.8},
                line_kws = {"lw":2, "alpha": 1},
                color = f'C{len(channel)}',  
                ax = ax,
                legend=  False,
               )

    ax.set_xlim(0,1)

    ax.tick_params(labelsize = 14)
    ax.set_yticks(np.arange(5, ax.get_yticks().max(), 5))

    ax.set_xlabel(f"{value.upper()} Value", fontsize =16)
    ax.set_ylabel("Percentage", fontsize =16)

    ax.set_title(f"{value.upper()} Attribution Distribution for {channel}", fontsize =18)

    if not show_nb:
        plt.close()
    else:
        plt.show()
   
    return fig

def attribution_distribution_compare(df, channel, *model, show_nb  = False):
    fig, ax = plt.subplots(figsize = (15,5))
    model_columns = list(model)

    sns.histplot(data = df[df.channel_label == channel][model_columns].melt(),
                 x = "value",
                 hue = 'variable',
                stat = "percent",
                bins = 50,
                alpha= 0.5,
                kde =True,
                kde_kws = {"bw_adjust":.8},
                line_kws = {"lw":2, "alpha": 1},
                color = f'C{len(channel)}',  
                ax = ax,
               )

    ax.set_xlim(0,1)

    ax.tick_params(labelsize = 14)
    ax.set_yticks(np.arange(5, ax.get_yticks().max(), 5))

    ax.legend(list(reversed(model)), prop = {"size":14}, framealpha =0)
    ax.set_ylabel("Percentage", fontsize =16)
    ax.set_xlabel("Value", fontsize =16)

    ax.set_title(f"Attribution Distribution for {channel} Channel", fontsize =18)
    
    if not show_nb:
        plt.close()
    else:
        plt.show()
   
    return fig

def attribution_distribution_difference(df, channel, model_1, model_2, show_nb  = False):
    fig, ax = plt.subplots(figsize = (15,5))
    
    to_plot=  df[df.channel_label == channel]

    sns.histplot(x = (to_plot[model_1] - to_plot[model_2]) ,
                stat = "percent",
                bins = 100,
                alpha= 0.5,
                kde =True,
                kde_kws = {"bw_adjust":1},
                line_kws = {"lw":2, "alpha": 1},
                color = f'C{len(channel)}',  
                ax = ax,
               )
    

    ax.tick_params(labelsize = 14)
    ax.set_yticks(np.arange(5, ax.get_yticks().max(), 5))

    ax.set_ylabel("Percentage", fontsize =16)
    ax.set_xlabel("", fontsize =16)

    ax.set_title(f"Attributed Conversion Difference for {channel} Channel \n Models: {model_1}, {model_2} ", fontsize =18)
    
    if not show_nb:
        plt.close()
    else:
        plt.show()
   
    return fig




