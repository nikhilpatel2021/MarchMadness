#Format Data for frontend data
import csv

confereceOrder = ("W", "X", "Z", "Y")
teams = {}
with open('kaggle_data/MTeams.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for line in csv_reader:
        teams[str(line[0])] = str(line[1])
        
def format(data):
    games = []
    for round in range(1,5):
        for conference in confereceOrder:
            for g in getOrder(round):
                game = "R" + str(round) + conference + str(g)
                row = data.loc[data['Slot'] == game]
                if round == 1 or g == 1:
                    games.append(row.iloc[0]["TeamID_x"])
                    games.append(row.iloc[0]["TeamID_y"])
                else:
                    games.append(row.iloc[0]["TeamID_y"])
                    games.append(row.iloc[0]["TeamID_x"])

    #Cross region games
    row = data.loc[data['Slot'] == "R5WX"]
    games.append(row.iloc[0]["TeamID_x"])
    games.append(row.iloc[0]["TeamID_y"])
    row = data.loc[data['Slot'] == "R5YZ"]
    games.append(row.iloc[0]["TeamID_y"])
    games.append(row.iloc[0]["TeamID_x"])
    row = data.loc[data['Slot'] == "R6CH"]
    games.append(row.iloc[0]["TeamID_x"])
    games.append(row.iloc[0]["TeamID_y"])
    games.append(row.iloc[0]["Winner"])

    return getTeamNames(games)


def getTeamNames(games):
    names = []
    for g in games:
        names.append(teams[str(int(g))])
    return names

def getOrder(round):
    if round == 1:
        return [1,8,5,4,6,3,7,2]
    elif round == 2:
        return [1,4,3,2]
    elif round == 3:
        return [1,2]
    else:
        return [1]
