import pandas as pd

def load_car_scout():
    df_auto_scout_car = pd.read_csv("progetto/auto_scout/auto_scout_car.csv")
    df_auto_scout_car = df_auto_scout_car.drop(columns=['make_model', 'vat','Comfort_Convenience','Entertainment_Media','Extras','Safety_Security'])
    return df_auto_scout_car

def clean_car_scout(df):
    columns_to_convert = ['Upholstery_type', 'Paint_Type', 'Gearing_Type', 'Fuel', 'Drive_chain', 'body_type']
    df[columns_to_convert] = df[columns_to_convert].astype("category")

    condition_mapping = {'New': 5,'Pre-registered': 4,'Demonstration': 3,"Employee's car": 2,'Used': 1}
    df['Type'] = df['Type'].map(condition_mapping)
    return df