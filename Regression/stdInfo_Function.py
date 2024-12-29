from faker import Faker
import random
import pandas as pd

# Setting a seed for consistent results
FAKER_SEED = 43
RANDOM_SEED = 43

# Library for generating synthetic names
fake = Faker('en_PH')
Faker.seed(FAKER_SEED)
random.seed(RANDOM_SEED)

# List for randomly picking year level
yrLvl = [1, 2, 3, 4]


class Student:
    # Library for generating synthetic names
    fake = Faker('en_PH')
    Faker.seed(FAKER_SEED)
    random.seed(RANDOM_SEED)


    def __init__(self, students):
        self.students = students
        
    
    # Function to generate student number based on year level
    def std_gen_num(self, year):
        # Dictionary for prefixes, what will be the starting number of their student number based on their year level
        year_prefix = {
            1: "24",  # For 1st year
            2: "23",  # For 2nd year
            3: "22",  # For 3rd year
            4: "21"   # For 4th year
        }
        prefix = year_prefix[year]
        random_digits = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        return prefix + random_digits
    
    def std_info_dt(self):
        all_students = []
        
        for i in range(self.students):
            info = {}
            name = fake.name()
            year = random.choice(yrLvl)
            student_number = self.std_gen_num(year)
            
            info["Name"] = name
            info["Year"] = year
            info["Student Number"] = student_number
            
            all_students.append(info)
        
        return pd.DataFrame(all_students)


