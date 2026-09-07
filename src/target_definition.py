"""
Target variable definition for the GLP-1 Use Predictor.

Source:
CDC/NCHS Round 7 Rapid Surveys

Target:
GLP_MED12M

Definition:
Whether the respondent took an oral or injectable GLP-1 medication
for diabetes or weight loss during the past 12 months.

Coding:
0 = No
1 = Yes
-5 = Edited response / outlier
-6 = Skipped / implied refusal
-7 = Explicit refusal
-8 = Question not asked
-9 = Don't know
"""

TARGET_COLUMN = "GLP_MED12M"

VALID_TARGET_VALUES = [0, 1]

TARGET_LABELS = {
    0: "No GLP-1 use",
    1: "GLP-1 use",
}