"""
Initial predictor variables for the GLP-1 Use Predictor.

These variables are selected because they may plausibly be associated
with GLP-1 medication use and are available before the reported
GLP-1 medication outcome.
"""

PREDICTOR_COLUMNS = [
    # Health conditions
    "CHR_OBEV",
    "DIB_DIBEV",

    # Healthcare access
    "INSURED",
    "DOCVIS_P12M",
    "ACC_HTHUSUAL",

    # Demographics
    "P_AGE5YRS_R",
    "P_SEX",
    "DEM_RACEETH",
    "DEM_REGION",

    # Socioeconomic factors
    "P_EDUCATION_I_R",
    "P_POVERTY4_R",

    # Body mass index
    "BMICAT6",
]