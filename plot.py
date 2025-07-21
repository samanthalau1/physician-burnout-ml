import preprocess_and_plot

# bar graph of burnout levels count
pyplot.rcParams['figure.dpi'] = 300
pyplot.rcParams['savefig.dpi'] = 300
pyplot.rcParams.update({'font.size': 12})

fig1 = sns.countplot(x = 'Burnout Level', data = clean_data, order=['Low - 1', '2', 'Moderate - 3', '4', 'High - 5'])
pyplot.title('Burnout Level Data Count')
burnout_count = fig1.get_figure()