# %%
import pandas as pd
import pgeocode

# Données avec Lat Long
df = pd.read_csv("DatasetChurn.csv")
df = df[["Zip Code", "Lat Long"]].drop_duplicates()

# Extraction de la latitude et de la longitude
df["Lat"] = df["Lat Long"].str.split(", ").str[0].astype(float)
df["Long"] = df["Lat Long"].str.split(", ").str[1].astype(float)
df.drop(columns="Lat Long", inplace=True)

# Liste exhaustive des Zip Code de Californie
zc = pd.read_csv("ZipCodes.csv")
zc = zc[["Zip Code"]].query("`Zip Code` > 90000")

# Merge
zc = zc.merge(df, how="left", on="Zip Code")

# Retrouver les coordonnées manquantes
geocoder = pgeocode.Nominatim("us")


def get_coordinates(zip_code):
    location = geocoder.query_postal_code(zip_code)
    return pd.Series({"Lat": location.latitude, "Long": location.longitude})


missing_coords_index = zc.loc[zc.isna().any(axis=1)].index
zc.loc[missing_coords_index, ["Lat", "Long"]] = zc.loc[
    missing_coords_index, "Zip Code"
].apply(get_coordinates)

# Sauvegarde
zc.to_csv("ZipCodesAndCoords.csv", index=False)
