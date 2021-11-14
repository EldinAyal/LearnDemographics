# This is a sample Python script.
import os
import csv
import shutil
from sklearn.model_selection import train_test_split
import re


import matplotlib.pyplot as plt
import pandas as pd
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
def create_data():
    header1 = ["patient", "Gender", "Age", "Weight", "Height"]
    header2 =["image", "Gender", "Age", "Weight", "Height"]
    with open('patients.csv', 'w') as f1,open('all_images.csv','w') as f2:
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer1.writerow(header1)
        writer2.writerow(header2)
        for patient in os.listdir("./C2_Demographics2"):
            if patient.startswith(".") or len(os.listdir(os.path.join("C2_Demographics2", patient)))<=1:
                continue
            labels = open(os.path.join("C2_Demographics2", patient, "demo.txt"), encoding='UTF-8').readline().split(
                ",")
            writer1.writerow([patient]+parse_labels(labels))
            unpack_images(patient,parse_labels(labels),writer2)

def create_data(file_name_input,file_name_output):
    header=["image", "Gender", "Age", "Weight", "Height"]
    with open(file_name_input, 'r') as input,open(file_name_output, 'w') as output:
        writer = csv.writer(output)
        writer.writerow(header)
        reader=csv.reader(input)
        for patient in reader:
            labels=patient[1:]
            unpack_images(patient[0],labels,writer)






def unpack_images(patient,labels,csv_writer):
    images_type=["Colon","SB","Stomach"]
    for image_type in images_type:
        path_to_images=os.path.join("C2_Demographics2", patient,image_type)
        if os.path.exists(path_to_images) is False:
            continue
        #write all images in csv file with corresponding label
        for image in os.listdir(path_to_images):
            csv_writer.writerow([os.path.join(patient,image_type,image)]+labels)


def plot_distrubtions(csv_file):
    df=pd.read_csv(csv_file)
    df.plot(x='patient',y='Age')
    df['Age'].plot(kind='hist')
    df.groupby(['Gender']).size().plot(kind='bar')
    plt.show()
def parse_labels(labels):
    labels = [''.join(label.split()) for label in labels]  # remove spaces from labels
    gender = labels[0].split(":")[1]
    age = labels[1].split(":")[1]
    weight = labels[2].split(":")[1] if len(labels[2].split(":")[1]) != 0 else None
    height = labels[3].split(":")[1] if len(labels[3].split(":")[1]) != 0 else None
    return [gender,age,weight,height]
if __name__ == '__main__':
    data=pd.read_csv('patients.csv')
    train_dataset, test_dataset = train_test_split(data, test_size=0.2,stratify=list(data['Gender']))
    train_dataset.to_csv('train_patients_csv.csv',index=False)
    test_dataset.to_csv('test_patients_csv.csv',index=False)
    create_data("train_patients_csv.csv","train_images.csv")
    create_data("test_patients_csv.csv", "test_images.csv")






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
