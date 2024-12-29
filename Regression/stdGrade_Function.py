import random
import pandas as pd
from faker import Faker
from stdInfo_Function import Student

# Setting a seed, so that results will consistent across different runs
FAKER_SEED = 43
RANDOM_SEED = 43

# Library for generating synthetic names 
fake = Faker('en_PH')
Faker.seed(FAKER_SEED)

random.seed(RANDOM_SEED)

# Inheriting Student attributes for student information 
class GrdSystem(Student): 
    # Define the year level curriculum
    curriculum = {
        1: {
            "subjects": 5,
            "units": [2, 3, 3, 3, 3]
        },
        2: {
            "subjects": 8,
            "units": [2, 3, 3, 3, 3, 3, 3, 3]
        },
        3: {
            "subjects": 7,
            "units": [3, 3, 3, 3, 3, 3, 3]
        },
        4: {
            "subjects": 5,
            "units": [3, 3, 3, 3, 3]
        }
    }
    
    def __init__(self, students):
        super().__init__(students)
        self.std_info = self.std_info_dt()  # Get student information
        self.grades = self.std_grade()  # Generate grades
        

    def std_grade(self):
        # Grade scale based on common grading systems
        grade_scale = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 4.00, 5.00]
        all_student_grades = []

        for i, row in self.std_info.iterrows():
            year = row["Year"]  # Identify the student's year
            std_year_curriculum = self.curriculum[year]  # Get curriculum for that year
            
            grades = {}
            tgp = 0  
            valid_units = 0  # Valid units that correspond to passing grades
            sub_failed = 0  # Track units of failed subjects

            for subs, units in enumerate(std_year_curriculum["units"]):  # Generate grades for subjects
                grade = random.choice(grade_scale)
                grades[f'Subject_{subs + 1}'] = grade  # Assign grades for each subject

                if grade < 6.00:  # Only count valid grades
                    tgp += grade * units
                    valid_units += units  # Only count units for passing grades
                
                if grade > 3.00:
                    sub_failed += 1  # Track failed subjects
                    
            status = "Irregular" if sub_failed > 0 else "Regular"

            final_grade = tgp / valid_units if valid_units > 0 else "N/A"
            
            all_student_grades.append({**grades, 
                                       "Final Grade": round(final_grade, 2) if final_grade != "N/A" else final_grade, 
                                       "Subjects Failed": sub_failed,
                                       "Status" : status})
  
        return pd.DataFrame(all_student_grades)  

    def overall_dt_stdGrades(self):
        # Concatenate student info and grades
        combined_df = pd.concat([self.std_info, self.grades], axis=1)
        
        # Move 'Final Grade' and 'Subjects Failed' to the right side
        columns_order = [col for col in combined_df.columns if col not in ['Final Grade', 'Subjects Failed']] + ['Final Grade', 'Subjects Failed']
        
        # Reorder the DataFrame columns
        combined_df = combined_df[columns_order]
        
        return combined_df