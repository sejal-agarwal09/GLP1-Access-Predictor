library(survey)
options(survey.lonely.psu = "adjust")

# Paths
DATA_PATH <- "data/raw/cdc_round7.csv"
OUTPUT_PATH <- "data/processed/weighted_glp1_prevalence_ci.csv"

# Load CDC Round 7 data
df <- read.csv(DATA_PATH)

# Keep valid GLP-1 responses
df <- df[df$GLP_MED12M %in% c(0, 1), ]

# Define CDC complex survey design
design <- svydesign(
  ids = ~P_PSU_R,
  strata = ~P_STRATA_R,
  weights = ~WEIGHT,
  data = df,
  nest = TRUE
)

# Estimate weighted GLP-1 prevalence
glp1_prevalence <- svymean(
  ~I(GLP_MED12M == 1),
  design,
  na.rm = TRUE
)

# Extract estimate and 95% CI
estimate <- coef(glp1_prevalence)[2]
ci <- confint(glp1_prevalence)[2, ]

# Print results
cat("\nCDC ROUND 7 DESIGN-BASED GLP-1 PREVALENCE\n")
cat(paste0("=", strrep("=", 60)), "\n")
cat(sprintf("Valid respondents: %s\n", format(nrow(df), big.mark = ",")))
cat(sprintf("Weighted prevalence: %.4f%%\n", estimate * 100))
cat(sprintf("95%% CI: %.4f%% - %.4f%%\n",
            ci[1] * 100,
            ci[2] * 100))

# Save results
results <- data.frame(
  Measure = "Weighted GLP-1 prevalence",
  Estimate = estimate,
  CI_Lower = ci[1],
  CI_Upper = ci[2]
)

write.csv(results, OUTPUT_PATH, row.names = FALSE)

cat("\nSaved:\n")
cat(OUTPUT_PATH, "\n")