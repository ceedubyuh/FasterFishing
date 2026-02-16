FasterFishing

FasterFishing is a Python GUI tool for “goldfishing” Magic: The Gathering Commander decks.
It lets you import a decklist, automatically categorize cards, and run simulations to analyze opening hands and early-game performance.

The goal is to quickly test consistency, ramp density, and land counts without needing to play full games.

Features

Import decks from popular sites

Moxfield

Archidekt

MTGGoldfish

Paste decklists directly into the app

Automatic card categorization

Uses Scryfall community oracle tags

Detects:

Lands

Ramp

Draw

Removal

Board wipes

Creatures

Opening hand simulations

Monte Carlo hand analysis

Mulligan scenarios

Keepable hand percentages

Turn-by-turn goldfish simulation

Tracks resource development

Estimates average commander cast turn

Sample hand viewer

Displays opening hands with card images

Mulligan options built in

Scryfall image integration

Automatically downloads and caches card art

Requirements

Python 3.9+

Required packages:

pip install requests Pillow
Installation

Clone the repository:

git clone https://github.com/ceedubyuh/FasterFishing.git
cd FasterFishing

Install dependencies:

pip install requests Pillow
Usage

Run the application:

python FasterFishing.py

Then:

Import a deck using a supported URL
or

Paste a decklist into the text field

View categories, run simulations, or draw sample hands

Example Decklist Format
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
How It Works

Decklists are parsed and card data is fetched from Scryfall.

Cards are categorized using Scryfall’s otag: system.

Simulations run thousands of randomized hands to produce statistical results.

Roadmap / Ideas

Save and load deck profiles

Custom category definitions

Mana curve visualization

Advanced mulligan logic

Exportable reports

Disclaimer

This is a fan-made tool for deck analysis and is not affiliated with Wizards of the Coast.
