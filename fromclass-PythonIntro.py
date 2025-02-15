#!/usr/bin/env python
# coding: utf-8

# # Introduction to Python for Data Science
# ## Dr. Clif Baldwin
# ## Machine Learning Fundamentals, Spring 2025
# January 21, 2025

# In[1]:


# Set some Python data structures
# Constant number
year = 2025 

# Constant string
MLprofessor = "Baldwin" 
DLprofessor = "Olsen"
DVprofessor = "Dunn"
DGprofessor = "Laurino"
DEprofessor = "Sokol"

# Vector of numbers
primes = [2,3,5,7,11,13,17,19,23,29] 

# List of Strings
classes = ["Introduction to Data", "Data Exploration", "Data Visualization",
           "Data Gathering", "Machine Learning", "Deep Learning",
           "Data Entrepreneurship"]

# List of string variables
professors = [DLprofessor, MLprofessor, DVprofessor, DGprofessor, DEprofessor] 

# List of Strings
classes = ["Introduction to Data", "Data Exploration", "Data Visualization",
           "Data Gathering", "Machine Learning", "Deep Learning",
           "Data Entrepreneurship"]

# List of lists
courses = [[classes[0], professors[0]], 
           [classes[1], professors[1]],
           [classes[1], professors[3]],
           [classes[2], professors[2]],
           [classes[3], professors[3]],
           [classes[4], professors[1]],
           [classes[5], professors[0]],
           [classes[6], professors[4]]]

# List of booleans
FallFilter = [True, True, True, False, False, False, False]


# In[2]:


professors


# In[3]:


courses


# In[4]:


classes[0:3]


# In[5]:


import numpy as np


# In[6]:


test = np.array([1, 2, 3, 4])
test


# In[7]:


names = np.array(["Bob", "Clif", "Keiana", "Melissa", "John"])
names


# In[8]:


classes = np.array(classes)
classes


# In[9]:


fall = classes[FallFilter]
fall


# <br>
# While R and other languages use brackets or parentheses to delimit blocks of code, Python uses indentation.<br>
# <br>

# In[10]:


print("The year is " + str(year))
if year < 2025:
    print("Completed")
else:
    print("Not finished yet")


# In[28]:


year = 2024
print("The year is " + str(year))
text = "Not finished yet"
if year < 2025:
    text = "Completed"

print(text)


# In[13]:


# Find the professors who teach classes that have "Data" in the title
for course in courses:
    if "Data" in course[0]:
        print(course[1])


# In[12]:


# Find all classes that have "Data" in the name
for c in classes:
    if "Data" in c:
        print(c)


# The Pythonic way to do For Loops is to use something called List Comprehension.

# In[24]:


# Non-Pythonic way
odds = []
for x in range(2, 25):
    if x % 2 != 0:
        odds.append(x)
print(odds)


# In[25]:


# Or the Pythonic way
odds = [x for x in range(2,25) if x % 2 != 0]
print(odds)


# ## Penguins Data

# In[1]:


import numpy as np
import pandas as pd

penguins = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv",delimiter=",")

# If you have the data saved locally
#penguins = pd.read_csv("penguins.csv",delimiter=",")

# If you load the Python library palmerpenguins
#from palmerpenguins import load_penguins
#penguins = load_penguins()

# In[2]:


# Explore the data 
penguins.head()


# In[3]:

# Explore the data 
penguins.info()


# From the list of column names, I am initially interested in 'species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', and 'body_mass_g'. I could add 'island' or any of the other variables, but at this time, I will just explore these five columns. I will save this subset dataset as ds.

# In[13]:


ds = penguins[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
ds.head()


# In[12]:

# Graph the data
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Palmer Penguins')
plt.xlabel('FlipperLength_mm')
plt.ylabel('BodyMass_g')
ax.scatter(ds.iloc[:,1].to_numpy(), ds.iloc[:,2].to_numpy())


# In[31]:


# Convert categories to numerical values
unique_categories = np.unique(penguins.iloc[:,0])
category_numbers = np.array([np.where(unique_categories == cat)[0][0] for cat in penguins.iloc[:,0]])

# Graph the data by Species
plt.scatter(x=penguins.iloc[:,2].to_numpy(), y=penguins.iloc[:,3].to_numpy(), c=category_numbers, cmap='viridis') 
plt.colorbar(ticks=np.arange(len(unique_categories)), format=lambda x, pos: unique_categories[int(x)])
plt.title('Palmer Penguins')
plt.xlabel('FlipperLength_mm')
plt.ylabel('BodyMass_g')
plt.show()


# # Homework
# Read chapters 2, 3, and 4  of <b>Data Science from Scratch</b>
# 
# ## Next Week
# Chapters 12 and 20 of <b>Data Science from Scratch</b>

# Assign Project 1 – Write a Python script to read in the Palmer Penguin dataset and explore it. 
# Document each step of the code to demonstrate you understand what each block of code does. 
# At a minimum, document what columns are in the dataset, what penguins are included, what islands, 
#   # how many penguins in each group, any outliers among the quantitative data, and any missing data.
# 
