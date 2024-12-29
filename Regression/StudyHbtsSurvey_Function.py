from stdInfo_Function import Student
from faker import Faker
import random
import pandas as pd

# Setting a seed, so that results will be consistent across different runs
FAKER_SEED = 43
RANDOM_SEED = 43

# Library for generating synthetic names
fake = Faker('en_PH')
Faker.seed(FAKER_SEED)

random.seed(RANDOM_SEED)

class S_H_Survey(Student):
    def __init__(self, students, total_respondents, respondents_ans, questions_per_category=3):
        super().__init__(students)  # Assuming Student class expects 'students' as input
        self.total_respondents = total_respondents
        self.respondents_ans = respondents_ans
        self.questions_per_category = questions_per_category
        self.category = [
            "Homework",
            "Time Allocation",
            "Reading and Note Taking",
            "Study Period Procedures",
            "Examination",
            "Teachers Consultation"
        ]
        self.hw_fct_1 = ["Yes", "No"]
        self.ta_fct_2 = ["Poor", "Fair", "Good"]
        self.rn_fct_3 = ["Handwritten", "Digital"]
        self.spp_fct_4 = ["Summarizing", "Group Discussions", "Flashcards", "Pomodoro"]
        self.tc_fct_6 = ["Yes", "No"]

    def answers(self):
        # Storage for all respondents
        all_total_respondents = []

        for i in range(self.total_respondents):
            respondents = {}

            # If respondent is part of the answering group
            if i < self.respondents_ans:
                for category in self.category:
                    if category == "Homework":
                        response = random.choice(self.hw_fct_1)
                    elif category == "Time Allocation":
                        response = random.choice(self.ta_fct_2)
                    elif category == "Reading and Note Taking":
                        response = random.choice(self.rn_fct_3)
                    elif category == "Study Period Procedures":
                        response = random.choice(self.spp_fct_4)
                    elif category == "Examination":
                        response = random.randint(1,20)  
                    elif category == "Teachers Consultation":
                        response = random.choice(self.tc_fct_6)
                        
                    respondents[f"{category}"] = response
            else:
                # If respondent did not answer, assign 0
                for category in self.category:
                    respondents[f"{category}"] = None

            all_total_respondents.append(respondents)

        # Creating DataFrame from collected responses
        df = pd.DataFrame(all_total_respondents)
        return df
    
    def std_info_and_survey(self):
        # Generate student info
        student_info = self.std_info_dt()

        # Generate survey responses
        survey_answers = self.answers()

        # Ensure both DataFrames have the same number of rows
        if len(student_info) > len(survey_answers):
            survey_answers = pd.concat([survey_answers, pd.DataFrame([None] * (len(student_info) - len(survey_answers)))], ignore_index=True)

        # Combine student info and survey responses
        InfoXSurvey = pd.concat([student_info, survey_answers], axis=1)

        return InfoXSurvey