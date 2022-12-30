import sys
import pandas as pd
import numpy as np
import json
from sqlalchemy  import create_engine
import re

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df

def empty_url_fixer(x):
    global urls_to_fix
    
    for i in urls_to_fix:
        try:
            if i in x:
                unique_url = x.split(i,1)[1].split()[0]

                x = x.replace(unique_url,"")

                x = x.replace(i,i.replace(" ","://")+"/"+unique_url)
        except Exception as e:
            print(e)
            print(i)
    return x

def remove_html_tags(x):
    CLEANR = re.compile('<.*?>')
    x = re.sub(CLEANR, '', x)
    return x

def remove_urls(x):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, x)
    for url in detected_urls:
        x = x.replace(url, "urlplaceholder")
    return x


def clean_data(df):
    global urls_to_fix
    
    categories = pd.DataFrame(df['categories'].apply(lambda x: x.split(";")).tolist())
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df.drop('categories',axis=1,inplace=True)
    df.drop_duplicates('message',inplace=True)
    df = pd.concat([df,categories],axis=1)
    df['related'] = df['related'].apply(lambda x: 1 if x==2 else x)
    df.drop('child_alone',axis=1,inplace=True)  
    # clean messages
    df =df[df.message.isnull()==False]
    
    urls_to_fix = list(set([i.split("http ")[1].split()[:2][0] for i in df[df.message.str.contains("http ")]['message']]))
    urls_to_fix = ["http "+i for i in urls_to_fix]
    urls_to_fix.remove('http bi')
    df['message'] = df['message'].apply(lambda x: remove_html_tags(x))
    df['message'] = df['message'].apply(lambda x: empty_url_fixer(x))
    df['message'] = df['message'].apply(lambda x: remove_urls(x))
    
    return df



def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse_table', engine, index = False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()