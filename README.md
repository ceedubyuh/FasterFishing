# FasterFishing

## FasterFishing is a Python GUI tool for “goldfishing” Magic: The Gathering Commander decks.
* It lets you import a decklist, automatically categorize cards, and run simulations to analyze opening hands and early-game performance.

### The goal is to quickly test consistency, ramp density, and land counts without needing to play full games.

### Features:
* Import decks from popular sites:
  * Moxfield 
  * Archidekt
  * MTGGoldfish

* Paste decklists directly into the app
* Automatic card categorization
  * Uses Scryfall community oracle tags
  * Detects (for now):
    * Lands
    * Ramp
    * Draw
    * Removal
    * Board wipes
    * Creatures
    * (more refinement to come)

* Opening hand simulations
  * Monte Carlo hand analysis
  * Mulligan scenarios
  * Keepable hand percentages
  * Calculates your manabase using Frank Karnsten's Mana Curve
    * Suggests what colors and sources your deck needs to improve
 
* Turn-by-turn goldfish simulation
  * Tracks resource development
  * Estimates average commander cast turn
  * Estimates average board state per turn
  * Estimates average damage presented on board and what turn you can go for a win on
  * Estimates the amount of cards you can see based on your repeatable card draw, regular card draw, tutors and graveyard recursion
  * Estimates the average turn you can execute any combos your deck includes

* Sample hand viewer
  * Displays opening hands with card images
  * Mulligan options built in

* Scryfall image integration
  * Automatically downloads and caches card art
 
* All-in-One deck builder
  * Edit your deck using Scryfall card searching and EDHREC implemented API to see suggesting cards
  * See combos listed from CommanderSpellbook and what cards you need to add to finish a combo

### Requirements
* Python 3.9+
* Required packages:
pip install requests Pillow

### Installation
1. Clone the repository:

git clone https://github.com/ceedubyuh/FasterFishing.git
cd FasterFishing

2. Install dependencies:

* ```pip install requests Pillow```
* ```pip install cloudscraper```

### Usage

Run the application:

```python FasterFishing.py```

Then:

1. Import a deck using a supported URL
**or**
2. Paste a decklist into the text field
3. View categories, run simulations, or draw sample hands

### Example Decklist Format
Commander
1 Atraxa, Praetors' Voice

Deck
1 Sol Ring
1 Cultivate
1 Swords to Plowshares
1 Island
1 Forest

You can also use:

1 Sol Ring
1 Command Tower
1 Cultivate #CMDR

### How It Works
* Decklists are parsed and card data is fetched from Scryfall.
* Cards are categorized using Scryfall’s otag: system.
* Simulations run thousands of randomized hands to produce statistical results.

  * Implementing recommended cards from EDHREC in app

### Disclaimer
This is a fan-made tool for deck analysis and is not affiliated with Wizards of the Coast.
