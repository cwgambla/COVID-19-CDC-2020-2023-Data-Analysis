#Importing libraries
import pandas as pd
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_predict


def show_max_deaths_in_month(month_df):
    month_df_copy = month_df
    month_df_copy['Start Date'] = pd.to_datetime(month_df['Start Date'])
    month_df_copy['End Date'] = pd.to_datetime(month_df['End Date'])
    filtered_month = month_df_copy[month_df_copy['COVID-19 Deaths'] == 105565]
    print("Max Deaths from Covid in a month of", "105565 from:" , filtered_month["Start Date"].iloc[0].strftime("%Y-%m-%d"), "to", filtered_month["End Date"].iloc[0].strftime("%Y-%m-%d"))
    
def empty_cells(total_df, year_df, month_df):
    #number of empty cells
    empty_cells = total_df['COVID-19 Deaths'].isnull().sum()
    print(empty_cells)
    empty_cells = year_df['COVID-19 Deaths'].isnull().sum()
    print(empty_cells)
    empty_cells = month_df['COVID-19 Deaths'].isnull().sum()
    print(empty_cells)

def clean_data(total_df, year_df, month_df):
    #Data Cleaning
    total_df['COVID-19 Deaths'] = pd.to_numeric(total_df['COVID-19 Deaths'], errors='coerce')
    
    # Use transform to fill NaN values with the mean of the group
    total_df['COVID-19 Deaths'] = total_df.groupby(['Sex', 'Age Group'])['COVID-19 Deaths'].transform(lambda x: x.fillna(0))
    
    
    year_df['COVID-19 Deaths'] = pd.to_numeric(year_df['COVID-19 Deaths'], errors='coerce')
    
    # Use transform to fill NaN values with the mean of the group
    year_df['COVID-19 Deaths'] = year_df.groupby(['Start Date','Sex', 'Age Group'])['COVID-19 Deaths'].transform(lambda x: x.fillna(0))
    
    
    month_df['COVID-19 Deaths'] = pd.to_numeric(month_df['COVID-19 Deaths'], errors='coerce')
    
    # Use transform to fill NaN values with the mean of the group
    month_df['COVID-19 Deaths'] = month_df.groupby(['Start Date','Sex', 'Age Group'])['COVID-19 Deaths'].transform(lambda x: x.fillna(0))

    return total_df, year_df, month_df

def basic_breakdowns_total_deaths(total_age_df):
    #basic breakdown of total deaths

    # Create a boolean mask for rows that do not have "51-55" or "65-69" in the "Age Range" column
    mask = ~total_age_df["Age Group"].isin(["Under 1 year","1-4 years", "5-14 years", "15-24 years","25-34 years","35-44 years"
                                           ,"45-54 years", "55-64 years", "All Ages"])
    
    # Apply the mask to the DataFrame to filter out the rows
    total_basic = total_age_df[mask]
    
    sns.barplot(x = total_basic['Age Group'], y = total_basic['COVID-19 Deaths'])
    # Rotate x-axis labels
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=80)
    
    # plt.xlabel('Age Group')
    plt.ylabel('COVID-19 Deaths (Millions)')
    plt.title("Total Deaths due to COVID-19")
    # Display the plot
    plt.show()
    #more detailed breakdown of total deaths by group
    
    # Create a boolean mask for rows that do not have "51-55" or "65-69" in the "Age Range" column
    mask = ~total_age_df["Age Group"].isin(["0-17 years", "18-29 years", "30-39 years","40-49 years","50-64 years", "All Ages"])
    
    # Apply the mask to the DataFrame to filter out the rows
    total_detailed = total_age_df[mask]
    
    sns.barplot(x = total_detailed['Age Group'], y = total_detailed['COVID-19 Deaths'], order=['Under 1 year', '1-4 years', '5-14 years', '15-24 years', '25-34 years', '35-44 years', '45-54 years', '55-64 years', '65-74 years', '75-84 years', '85 years and over'])
    # Rotate x-axis labels
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=80)
    
    plt.ylabel('COVID-19 Deaths (Millions)')
    plt.title("Total Deaths due to COVID-19")
    # Display the plot
    plt.show()

def line_graphs_yearly_deaths(year_age_df):
    mask = ~year_age_df["Age Group"].isin(["Under 1 year","1-4 years", "5-14 years", "15-24 years","25-34 years","35-44 years"
                                       ,"45-54 years", "55-64 years", "All Ages"])

    # Apply the mask to the DataFrame to filter out the rows
    basic_yearly = year_age_df[mask]
    
    sns.lineplot(x='Start Date', y='COVID-19 Deaths', data=basic_yearly, hue = "Age Group")
    
    plt.title("Yearly Deaths due to COVID-19")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45)
    
    plt.show()
    # Create a boolean mask for rows that do not have "51-55" or "65-69" in the "Age Range" column
    mask = ~year_age_df["Age Group"].isin(["0-17 years", "18-29 years", "30-39 years","40-49 years","50-64 years", "All Ages"])
    
    # Apply the mask to the DataFrame to filter out the rows
    detailed_yearly = year_age_df[mask]
    plt.title("Yearly Deaths due to COVID-19")
    sns.lineplot(x='Start Date', y='COVID-19 Deaths', data=detailed_yearly, hue = "Age Group")
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

def line_graphs_monthly_deaths(month_age_df):
    mask = ~month_age_df["Age Group"].isin(["0-17 years", "18-29 years", "30-39 years","40-49 years","50-64 years", "All Ages"])
    
    # Apply the mask to the DataFrame to filter out the rows
    detailed_monthly = month_age_df[mask]
    plt.title("Monthly Deaths due to COVID-19")
    hue_order = ["Under 1 year","1-4 years", "5-14 years", "15-24 years","25-34 years","35-44 years"
                                       ,"45-54 years", "55-64 years",'65-74 years', '75-84 years', '85 years and over']
    sns.lineplot(x='Start Date', y='COVID-19 Deaths', data=detailed_monthly, hue = "Age Group", hue_order=hue_order)
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

def ARMA(month_age_df):
    # month_age_df.dtypes
    from statsmodels.tsa.arima.model import ARIMA 
    predictions = ["Under 1 year","1-4 years", "5-14 years", "15-24 years","25-34 years","35-44 years"
                ,"45-54 years", "55-64 years","65-74 years","75-84 years","85 years and over","All Ages"]
    results = []
    forecasts = []
    # basic_month.set_index('Start Date', inplace=True)
    for x in predictions:
        
        mask = month_age_df["Age Group"].isin([x])
    
        # Apply the mask to the DataFrame to filter out the rows
        basic_month = month_age_df[mask]
        basic_month.dtypes
    
        basic_month = basic_month.drop(['Age Group'], axis=1)
    #     print(basic_month)
        basic_month.set_index('Start Date', inplace=True)
        model = ARIMA(basic_month[0:40], order=(4, 0, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)
        forecasts.append(forecast)
        mae = mean_absolute_error(forecast, basic_month[40:45])
        mse = mean_squared_error(forecast, basic_month[40:45])
        result = x +  " MAE: " + str(mae) + " MSE: " + str(mse)
        results.append(result)
#     print(forecasts)
    return results,forecasts

def ARIMA(month_age_df):
    # month_age_df.dtypes
    from statsmodels.tsa.arima.model import ARIMA 
    predictions = ["Under 1 year","1-4 years", "5-14 years", "15-24 years","25-34 years","35-44 years"
                ,"45-54 years", "55-64 years","65-74 years","75-84 years","85 years and over","All Ages"]
    results = []
    forecasts = []
    # basic_month.set_index('Start Date', inplace=True)
    for x in predictions:
        
        mask = month_age_df["Age Group"].isin([x])
    
        # Apply the mask to the DataFrame to filter out the rows
        basic_month = month_age_df[mask]
        basic_month.dtypes
    
        basic_month = basic_month.drop(['Age Group'], axis=1)
    #     print(basic_month)
        basic_month.set_index('Start Date', inplace=True)
        model = ARIMA(basic_month[0:40], order=(4, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)
        forecasts.append(forecast)
        mae = mean_absolute_error(forecast, basic_month[40:45])
        mse = mean_squared_error(forecast, basic_month[40:45])
        result = x +  " MAE: " + str(mae) + " MSE: " + str(mse)
        results.append(result)
    return results,forecasts        

def graph_results(models,month_age_df):
    sns.set_palette("magma")
    

    predictions = ["Under 1 year","1-4 years", "5-14 years", "15-24 years","25-34 years","35-44 years"
                ,"45-54 years", "55-64 years","65-74 years","75-84 years","85 years and over","All Ages"]
    for x in range(0,int(len(predictions)),4):
        mask = month_age_df["Age Group"].isin([predictions[x]])
        # Assuming month_age_df and mask are defined elsewhere in your code
        basic_month = month_age_df[mask]
        basic_month.dtypes
        basic_month = basic_month.drop(['Age Group'], axis=1)
        basic_month.set_index('Start Date', inplace=True)

        # Create a figure and a 1x2 grid of subplots
        fig, axs = plt.subplots(1, 4, figsize=(16, 5))

        #Plotting the first model's predictions
        basic_month.loc['Jan 2020':].plot(ax=axs[0])
        axs[0].set_ylim(0, basic_month['COVID-19 Deaths'].max() + basic_month['COVID-19 Deaths'].mean()*2)
        # axs[0].set_ylim(0, basic_month['COVID-19 Deaths'].max() + basic_month['COVID-19 Deaths'].mean())
        plot_predict(models[x], start='March 2023', end='September 2023', ax=axs[0])
        axs[0].set_title('AIRMA Predictions For ' + predictions[x])
        
        mask = month_age_df["Age Group"].isin([predictions[x+1]])
        basic_month = month_age_df[mask]
        basic_month.dtypes
        basic_month = basic_month.drop(['Age Group'], axis=1)
        basic_month.set_index('Start Date', inplace=True)

     
        # Plotting the second model's predictions
        basic_month.loc['Jan 2020':].plot(ax=axs[1])
        axs[1].set_ylim(0, basic_month['COVID-19 Deaths'].max() + basic_month['COVID-19 Deaths'].mean()*2)
        plot_predict(models[x+1], start='March 2023', end='September 2023', ax=axs[1])
        axs[1].set_title('ARIMA Predictions For ' + predictions[x+1])

        # Adjust the layout to ensure there's enough space between the plots
        plt.tight_layout()

        # plt.show()

        mask = month_age_df["Age Group"].isin([predictions[x+2]])
        basic_month = month_age_df[mask]
        basic_month.dtypes
        basic_month = basic_month.drop(['Age Group'], axis=1)
        basic_month.set_index('Start Date', inplace=True)

        

        basic_month.loc['Jan 2020':].plot(ax=axs[2])
        axs[2].set_ylim(0, basic_month['COVID-19 Deaths'].max() + basic_month['COVID-19 Deaths'].mean()*5)
        plot_predict(models[x+2], start='March 2023', end='September 2023', ax=axs[2])
        axs[2].set_title('ARMA Predictions For ' + predictions[x + 2])
        
        mask = month_age_df["Age Group"].isin([predictions[x+3]])
        basic_month = month_age_df[mask]
        basic_month.dtypes
        basic_month = basic_month.drop(['Age Group'], axis=1)
        basic_month.set_index('Start Date', inplace=True)
        
        
        # Plotting the second model's predictions
        basic_month.loc['Jan 2020':].plot(ax=axs[3])
        axs[3].set_ylim(0, basic_month['COVID-19 Deaths'].max() + basic_month['COVID-19 Deaths'].mean()*5)
        plot_predict(models[x+3], start='March 2023', end='September 2023', ax=axs[3])
        axs[3].set_title('ARIMA Predictions For ' + predictions[x + 3])

        # Adjust the layout to ensure there's enough space between the plots
        plt.tight_layout()

        plt.show()