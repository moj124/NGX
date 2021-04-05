import matplotlib.pyplot as plt
import pandas as pd
import datetime
import calendar


def findDay(date):
    """retrieves the day of the week given a date

    Parameters
    ----------
    date : str
        An string given in Y:m:d specifying the date
    """

    # retrieve the int day of the week
    day = datetime.datetime.strptime(date, '%Y:%m:%d').weekday()

    # return the day name of the week
    return calendar.day_name[day]


def load_graph(path):
    """load and update the graph plots of visitor flows

    Parameters
    ----------
    path : string
        An string of the output path to save and load graph plots

    """

    # load data from path
    data = pd.read_csv(path)

    # setup dictionary with each day each hour
    avg_days = {}
    days = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']

    for day in days:  # every day
        avg_days[day] = {}

        for j in range(24):  # every hour
            if j < 10:
                avg_days[day]['0' + str(j)] = []
            else:
                avg_days[day][str(j)] = []

    # retrieve each record from data and add the count to the respective day and hour of the data dictionary
    for dat in data['date'].unique():
        day = findDay(str(dat))
        for i in range(len(data)):

            hr = str(data.iloc[i]['time'][0:2])

            # where date matches each other
            if str(data.iloc[i]['date']) == str(dat):
                avg_days[day][hr].append(
                    data.iloc[i]['count'])

    # append the average of the counts from data dictionary
    for dat in data['date'].unique():
        day = findDay(str(dat))
        for time in avg_days[day].keys():

            # where data is in correct format
            if isinstance(avg_days[day][time], list):
                if len(avg_days[day][time]) > 0:
                    avg_days[day][time] = sum(
                        avg_days[day][time])/len(avg_days[day][time])
                else:
                    avg_days[day][time] = 0

    # for each day create a plot of average visitor flow
    for dat in data['date'].unique():
        # print(dat)
        day = findDay(str(dat))
        fig, ax = plt.subplots()
        ax.bar(avg_days[day].keys(),
               avg_days[day].values())
        ax.tick_params(axis='x', which='major', labelsize=5)
        ax.tick_params(axis='y', which='minor', labelsize=8)
        plt.suptitle(day+' Visitor Flow')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Occurence of People')
        plt.grid(True)
        # save to the respective folder paths
        plt.savefig('tests/'+day+'4_plot.pdf', bbox_inches='tight')
        print('saved to ' + 'tests/'+day+'4_plot.pdf')


if __name__ == '__main__':
    load_graph('tests/case4.csv')
