"""Generate a synthetic newspaper dataset for the example notebook.

The real corpus used in the paper is copyrighted and cannot be shared. This
script creates a small stand-in with the same columns, so that the pipeline
can be tried end to end. All articles are produced from sentence templates;
places, people and events are invented.

Each article belongs to one of nine themes. The share of each theme changes
over the years (some rise, some decline, some peak in particular years), so
that the topics-over-time analysis has something to show.

Usage:
    python generate_dummy_dataset.py
"""

import random

import pandas as pd

SEED = 42
FIRST_YEAR, LAST_YEAR = 1955, 2018
ARTICLES_PER_YEAR = 45
OUTPUT_FILE = "dummy_newspaper_dataset.csv"

PLACES = ["Northfield", "Riverton", "Eastbrook", "Marlow Valley", "Port Hadley",
          "Kingsmere", "Ashcombe", "Westmoor", "Lindholm", "Greywater"]
SURNAMES = ["Hartwell", "Mercer", "Lindqvist", "Okafor", "Brandt", "Vasquez",
            "Thornbury", "Delacroix", "Nakamura", "Petrov", "Whitlock", "Amsel"]
NEWSPAPERS = ["The Northfield Courier", "Riverton Daily Post", "Eastbrook Gazette",
              "Port Hadley Herald"]

GENERIC = [
    "Residents of {place} followed the news closely throughout the week.",
    "A spokesman said further details would be announced in the coming days.",
    "The matter was discussed at length on {weekday} evening.",
    "Local observers described the development as significant for the region.",
    "Reporters from this newspaper were present at the scene.",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

THEMES = {
    "transport": {
        "titles": ["New railway timetable for {place}", "Road works delay traffic near {place}",
                   "Bus service extended to {place}", "Station at {place} to be modernised"],
        "sentences": [
            "The railway company announced a new timetable for trains between {place} and {place2}.",
            "Passengers complained about delays on the morning train to {place}.",
            "Road works on the main highway caused long traffic queues for motorists.",
            "The transport ministry approved funds for a new bridge and a wider road.",
            "Bus drivers asked for more vehicles on the crowded commuter route.",
            "Engineers inspected the railway track and the signals after the incident.",
            "The station building will receive a new platform and a ticket office.",
            "Freight trains carried more goods through the junction than in previous years.",
            "Motorists were advised to use the bypass while the tunnel remains closed.",
            "The tram line will be extended to the eastern suburbs next year.",
            "Commuters welcomed the faster connection and the lower ticket prices.",
            "A new motorway section was opened to traffic by the transport minister.",
        ],
    },
    "agriculture": {
        "titles": ["Farmers expect a good harvest around {place}", "Milk prices worry farmers in {place}",
                   "Cattle market draws crowds to {place}", "Dry summer threatens the wheat crop"],
        "sentences": [
            "Farmers in the valley reported a good wheat harvest after a mild summer.",
            "The price of milk fell again, and dairy farmers demanded support.",
            "Cattle and sheep were sold at the weekly livestock market in {place}.",
            "The agricultural cooperative bought new tractors and harvesting machines.",
            "A dry spring damaged the potato crop on many farms in the district.",
            "Vineyard owners expect a smaller grape harvest because of the late frost.",
            "The farmers union asked the ministry for higher grain prices.",
            "Young farmers are leaving the land to look for work in the towns.",
            "The annual agricultural fair showed prize cattle, cheese and farm machinery.",
            "Orchards around {place} produced apples and pears of excellent quality.",
            "Fertiliser costs rose sharply, reducing the income of small farms.",
            "Barley and maize were sown later than usual because of the wet fields.",
        ],
    },
    "football": {
        "titles": ["{place} win the cup final", "Late goal saves a point for {place}",
                   "{place} sign a new striker", "Derby defeat angers supporters in {place}"],
        "sentences": [
            "The football club of {place} won the match by two goals to one.",
            "The striker scored a late goal in front of a large crowd of supporters.",
            "The goalkeeper saved a penalty in the second half of the game.",
            "The coach praised his players after the victory in the league.",
            "Supporters travelled to the stadium for the cup final on Sunday.",
            "The team lost the derby and dropped to fourth place in the table.",
            "The club signed a young midfielder before the start of the season.",
            "The referee sent off a defender after a hard tackle.",
            "The championship will be decided in the last match of the season.",
            "Training resumed on the pitch after the winter break.",
            "The captain lifted the trophy as the fans celebrated in the stands.",
            "A draw away from home kept the team at the top of the league.",
        ],
    },
    "elections": {
        "titles": ["Voters go to the polls in {place}", "Coalition talks begin after the election",
                   "Mayor of {place} re-elected", "Parties present their election programmes"],
        "sentences": [
            "Voters went to the polls to elect a new parliament on Sunday.",
            "The governing party lost seats, and the opposition gained votes.",
            "Candidate {surname} was elected mayor of {place} with a clear majority.",
            "The parties presented their election programmes to the public.",
            "Turnout was higher than at the previous election, officials said.",
            "Coalition talks between the two largest parties began this week.",
            "The prime minister announced his resignation after the defeat.",
            "Ballot papers were counted through the night in the town hall.",
            "The campaign focused on taxes, pensions and housing.",
            "The opposition leader called for a vote of confidence in parliament.",
            "Members of the council approved the new budget after a long debate.",
            "The election commission confirmed the final result on {weekday}.",
        ],
    },
    "floods": {
        "titles": ["River bursts its banks at {place}", "Storm damage across the region",
                   "Flood waters force families from their homes", "Heavy snow cuts off villages"],
        "sentences": [
            "Heavy rain caused the river to burst its banks near {place}.",
            "Flood water covered streets and cellars in the lower part of the town.",
            "Firemen and soldiers built sandbag barriers along the river.",
            "Hundreds of families were evacuated from their flooded homes.",
            "The storm tore roofs from houses and brought down many trees.",
            "The water level fell slowly, and the clean-up began on {weekday}.",
            "Meteorologists warned of more rain and strong winds in the coming days.",
            "The dam held, but several bridges were damaged by the flood.",
            "Heavy snowfall blocked mountain roads and cut off several villages.",
            "The government promised emergency aid for the victims of the flood.",
            "Rescue teams used boats to reach people trapped by the rising water.",
            "Damage to homes, roads and fields is estimated at several million.",
        ],
    },
    "computing": {
        "titles": ["Schools in {place} receive computers", "Local firm launches new software",
                   "Internet access reaches {place}", "Mobile phones change everyday life"],
        "sentences": [
            "The company presented a new computer with a faster processor.",
            "Schools received computers and software for their classrooms.",
            "More households now have access to the internet, a survey found.",
            "Engineers developed a program to store and search large amounts of data.",
            "Sales of mobile phones rose sharply during the past year.",
            "The bank introduced electronic payment and online accounts for customers.",
            "Experts warned users about computer viruses and weak passwords.",
            "The new website allows citizens to fill in official forms online.",
            "A local firm exports software and microchips to several countries.",
            "Fast broadband cables were laid in the town centre of {place}.",
            "Digital cameras and laptops were among the most popular gifts this year.",
            "The university opened a laboratory for computer science and networks.",
        ],
    },
    "health": {
        "titles": ["New hospital wing opens in {place}", "Doctors warn of influenza wave",
                   "Vaccination campaign begins in schools", "Shortage of nurses at {place} hospital"],
        "sentences": [
            "The hospital in {place} opened a new wing with eighty beds.",
            "Doctors reported a rising number of influenza patients this winter.",
            "A vaccination campaign for children began in schools and clinics.",
            "Nurses asked for better pay and shorter shifts on the wards.",
            "The health ministry published advice on diet, smoking and exercise.",
            "Surgeons performed a difficult heart operation at the university clinic.",
            "The epidemic spread quickly, and many patients needed treatment.",
            "A new medicine reduced the symptoms of the disease in most patients.",
            "The clinic bought modern equipment for X-ray examinations.",
            "Doctor {surname} warned that waiting times for patients are too long.",
            "Health insurance contributions will rise next year, the minister said.",
            "Ambulance crews answered more emergency calls than ever before.",
        ],
    },
    "energy": {
        "titles": ["Fuel shortage hits {place}", "Electricity prices to rise",
                   "New power station planned near {place}", "Households urged to save energy"],
        "sentences": [
            "The price of oil and petrol rose sharply, and motorists queued at filling stations.",
            "Households were urged to save electricity and heating fuel.",
            "The energy company plans a new power station near {place}.",
            "A power cut left thousands of homes without electricity for hours.",
            "The government introduced a speed limit to reduce fuel consumption.",
            "Coal deliveries were delayed, and stocks at the power station ran low.",
            "Electricity prices will rise next year, the supplier announced.",
            "Engineers repaired the damaged power line and the transformer.",
            "Wind turbines and solar panels now supply part of the electricity.",
            "The gas pipeline will be extended to supply more towns in the region.",
            "Factories reduced production because of the shortage of fuel oil.",
            "The minister defended the energy policy in a debate about supply and prices.",
        ],
    },
    "education": {
        "titles": ["New school opens in {place}", "Teachers call for smaller classes",
                   "Examination results published", "University of {place} welcomes new students"],
        "sentences": [
            "A new primary school opened in {place} with twelve classrooms.",
            "Teachers called for smaller classes and more modern textbooks.",
            "Pupils received their examination results at the end of the school year.",
            "The university welcomed a record number of new students.",
            "The education ministry presented a reform of the school curriculum.",
            "Parents criticised the long journey to the secondary school.",
            "The headmaster, Mr {surname}, thanked the staff for their work.",
            "Evening courses for adults began at the library and the college.",
            "Students protested against higher tuition fees and crowded lecture halls.",
            "The school library received a donation of several hundred books.",
            "Apprentices completed their vocational training and received certificates.",
            "Lessons in mathematics and foreign languages will be extended.",
        ],
    },
}


def theme_weights(year):
    """Relative share of each theme in a given year."""
    progress = (year - FIRST_YEAR) / (LAST_YEAR - FIRST_YEAR)
    weights = {
        "transport": 1.0,
        "agriculture": 1.8 - 1.4 * progress,                    # declines over time
        "football": 0.9 + 0.4 * progress,                       # rises slowly
        "elections": 2.6 if (year - FIRST_YEAR) % 5 == 0 else 0.35,   # election years
        "floods": 2.8 if year in (1962, 1978, 1993, 2002) else 0.3,   # flood years
        "computing": 0.0 if year < 1978 else 2.6 * (year - 1978) / (LAST_YEAR - 1978),
        "health": 2.4 if year in (1968, 1969) else 0.8,         # influenza wave
        "energy": 3.0 if year in (1973, 1974, 2008) else 0.5,   # fuel shortages
        "education": 0.8,
    }
    return weights


def make_article(theme, rng):
    spec = THEMES[theme]
    place, place2 = rng.sample(PLACES, 2)
    fill = {"place": place, "place2": place2, "surname": rng.choice(SURNAMES),
            "weekday": rng.choice(WEEKDAYS)}
    sentences = rng.sample(spec["sentences"], rng.randint(5, 8))
    sentences.insert(rng.randint(1, len(sentences)), rng.choice(GENERIC))
    title = rng.choice(spec["titles"]).format(**fill)
    content = " ".join(s.format(**fill) for s in sentences)
    return title, content


def main():
    rng = random.Random(SEED)
    rows = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        weights = theme_weights(year)
        themes = rng.choices(list(weights), weights=list(weights.values()), k=ARTICLES_PER_YEAR)
        for theme in themes:
            title, content = make_article(theme, rng)
            month, day = rng.randint(1, 12), rng.randint(1, 28)
            rows.append({
                "uid": f"DUMMY-{year}-{len(rows):05d}",
                "newspaper": rng.choice(NEWSPAPERS),
                "date": f"{year}-{month:02d}-{day:02d}",
                "year": year,
                "language": "en",
                "title": title,
                "content": content,
                "theme": theme,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, sep=";", index=False)
    print(f"Wrote {len(df)} articles ({FIRST_YEAR}-{LAST_YEAR}) to {OUTPUT_FILE}")
    print(df["theme"].value_counts().to_string())


if __name__ == "__main__":
    main()
