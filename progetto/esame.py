from AutoScoutCarData import load_car_scout,clean_car_scout


df = load_car_scout()
df_clean = clean_car_scout(df)

print(df_clean.info())
print(df_clean)
print(df_clean.head())

# model = xgb.XGBRegressor(
#     tree_method="hist",        # Use 'hist' for efficient training
#     enable_categorical=True,   # Enable categorical support
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1
# )