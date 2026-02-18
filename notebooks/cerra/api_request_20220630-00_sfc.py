import cdsapi

dataset = "reanalysis-cerra-single-levels"
request = {
    "variable": [
        "orography",
        "surface_pressure"
    ],
    "level_type": "surface_or_atmosphere",
    "data_type": ["reanalysis"],
    "product_type": "analysis",
    "year": ["2022"],
    "month": ["06"],
    "day": ["30"],
    "time": ["00:00"],
    "data_format": "grib"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

