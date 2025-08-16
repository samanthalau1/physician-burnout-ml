# functions for data visualizations

import matplotlib.pyplot as pyplot
import seaborn as sns

from data_utils import fix_hours, fix_years


# bar graph of burnout levels count
def level_count(clean_data):
    pyplot.rcParams['figure.dpi'] = 100
    pyplot.rcParams['savefig.dpi'] = 100
    pyplot.rcParams.update({'font.size': 12})

    fig1 = sns.countplot(x = 'Burnout Level', data = clean_data, order=['Low - 1', '2', 'Moderate - 3', '4', 'High - 5'])
    pyplot.title('Burnout Level Data Count')
    pyplot.show()


# graph of burnout correlated to hours on ehr, admin, and seeing patients
def hours_correlation(clean_data):
    graph_data = fix_hours(clean_data)

    melted = graph_data.melt(id_vars='Burnout Level', 
                         value_vars=['Patient Hours', 'EHR Hours', 'Admin Hours'],
                         var_name='Task Type', 
                         value_name='Hours')

    sns.lineplot(data=melted, x='Hours', y='Burnout Level', hue='Task Type', style='Task Type', errorbar=None)
    pyplot.title('Burnout Level vs. Time Spent on Tasks')
    pyplot.show()
    

# lineplot for years worked
def years_correlation(clean_data):
    graph_data = fix_years(clean_data)
    fig2 = sns.lineplot(data=graph_data, x='Years Worked', y='Burnout Level', errorbar=None)
    pyplot.title("Burnout Level vs. Years Worked")
    pyplot.show()