import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carichiamo CSV
df_listings = pd.read_csv("progetto/Turam/listings.csv")
df_calendar = pd.read_csv("progetto/Turam/calendar.csv")
df_reviews = pd.read_csv("progetto/Turam/reviews_scompressa.csv")

# Puliamo i dataframe
df_listings = df_listings.drop(columns=['license'])
df_calendar['price'] = df_calendar['price'].str.replace("$", "").str.replace(",", "").astype(float)

df_calendar_mean = df_calendar.groupby('listing_id')['price'].mean()
df_listings['price'] = df_listings['id'].map(df_calendar_mean)
df_listings = df_listings.drop(columns=['name','host_name','last_review','reviews_per_month','latitude','longitude'])

df_listings = df_listings.dropna(subset=['price'])

# Rinominiamo le colonne
df_listings = df_listings.rename(columns={
    'id': 'ID_Annuncio',
    'host_id': 'ID_Host',
    'neighbourhood_group': 'Gruppo_Quartieri',
    'neighbourhood': 'Quartiere',
    'room_type': 'Tipo_Stanza',
    'price': 'Prezzo_Per_Notte',
    'minimum_nights': 'Notti_Minime',
    'number_of_reviews': 'Totale_Recensioni',
    'calculated_host_listings_count': 'Numero_Annunci_Host',
    'availability_365': 'Disponibilità_365_Giorni',
    'number_of_reviews_ltm': 'Recensioni_Ultimi_12_Mesi'
})



