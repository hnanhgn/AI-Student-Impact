import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_path = os.path.join(base_dir, "data", "ai_student_impact_dataset (1).csv")

cleaned_path = os.path.join(base_dir, "data", "data_cleaned.csv")

processed_dir = os.path.join(base_dir, "data")

# target
target = "Post_Semester_GPA"

# features
features = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Tool_Diversity",
    "Paid_Subscription",
    "Perceived_AI_Dependency",
]

id = "Student_ID"

binary_feature = ["Paid_Subscription"]

nominal_feature = [
    "Major_Category",
    "Primary_Use_Case",
    "Institutional_Policy",
]

ordinal_feature = ["Year_of_Study", "Prompt_Engineering_Skill", "Burnout_Risk_Level"]

ordinal_category = {
    "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
    "Burnout_Risk_Level": ["Low", "Medium", "High"],
}

stratify = "Year_of_Study"

random_state = 42
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15