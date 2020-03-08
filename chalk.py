import csv
from formatting import getTeamNames

firstSeedRankings = ("W", "X", "Z", "Y")

def chalk_simulator(year):
    teams = {}
    orderedTeams = []
    with open('kaggle_data/MNCAATourneySeeds.csv') as file:
        print("OPEN")
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        for line in csv_reader:
            if not first and (int(line[0]) == year and "b" not in str(line[1])):
                key = ""
                if "10" not in line[1]:
                    key = line[1].replace("a","").replace("0","")
                else:
                    key = line[1].replace("a","")
                print(key)
                teams[key] = line[2]
            else:
                first = False
    

    order1 = [1,8,5,4,6,3,7,2]
    order2 = [1,4,3,2]
    order3 = [1,2]
    order4 = [1]
    
    for region in firstSeedRankings:
        for seed in order1:
            team = region + str(seed)
            orderedTeams.append(teams[team])
            team = region + str(17 - seed)
            orderedTeams.append(teams[team])
    for region in firstSeedRankings:
        for seed in order1:
            team = region + str(seed)
            orderedTeams.append(teams[team])
    for region in firstSeedRankings:
        for seed in order2:
            team = region + str(seed)
            orderedTeams.append(teams[team])
    for region in firstSeedRankings:
        for seed in order3:
            team = region + str(seed)
            orderedTeams.append(teams[team])
    for region in firstSeedRankings:
        for seed in order4:
            team = region + str(seed)
            orderedTeams.append(teams[team])

    for region in firstSeedRankings[0::2]:
        team = region + str(1)
        orderedTeams.append(teams[team])
    team = region + str(1)
    orderedTeams.append(teams[team])

    return getTeamNames(orderedTeams)
