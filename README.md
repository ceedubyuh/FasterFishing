# FasterFishing

## FasterFishing is a Python GUI tool for “goldfishing” Magic: The Gathering Commander decks.
* It lets you import a decklist, automatically categorize cards, and run simulations to analyze opening hands and early-game performance.

### The goal is to quickly test consistency, ramp density, and land counts without needing to play full games or manually Goldfish your deck to oblivion. Using reliable math, you can get quick, quantifiable data about your deck without all the hassle.
### The idea isn't to make a simulator that creates an impervious deck, because the RNG in a 4-player format is nigh impossible to account for, but have a mathematical breakdown of your cards you can use to fine tune a list to perform the way you like it and statistically improve the list in ways you would have trouble doing manually.

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
    * Equipment
    * Tutors

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
  * Estimates average damage presented on board and what turn you can go for a win
      * Detects alternate wincons like empty libraries, creature count, life total etc. and the average turn you will execute those conditions
  * Estimates the amount of cards you can see based on your repeatable card draw, regular card draw and tutors
  * Estimates the average turn you can execute any combos your deck includes\
  * Comprehensive Trigger Chain system to keep track of your triggers 
  * Performs a per-card win contribution measurement to estimate how good a singular card is in your deck using a Leave One Out strategy
      *  Causal measurement: for each unique non-land card, remove it from the deck
         and re-sim to measure actual impact on kill turn. This naturally captures:
           * Combo synergy (removing a combo piece drops combo assembly rate)
           * Draw chain value (removing a draw engine means fewer cards seen → slower kills)
           * Ramp contribution (removing ramp delays commander and bombs)
         After LOO, detect synergy pairs: cards whose combined removal impact
         exceeds the sum of their individual impacts (superadditive = true synergy).

---- SMART GOLDFISH CASTING PRIORITIES ----
These model a competent Commander player's decision-making:

 CAST PROBABILITIES (simulates a goldfish with no opponents to interact with):
   * Ramp:      100% early (turns 1-4), 60% late (turns 5+, diminishing returns)
   * Draw:      100% always (card advantage is king)
   * Tutor:     85% (sometimes you hold for the right moment)
   * Board:     90% (creatures, enchantments, artifacts — your main plan)
   * Protection: 25% (counterspells/hexproof pieces — minimal value in goldfish)
   * Combo:     95% (always want these in play if you can)

 ADDITIONAL INTELLIGENCE:
   * Extra land drops detected from oracle text (Exploration, Azusa)
   * Conditional draw spells held until condition met (power-based like Return
     of the Wildspeaker wait for a big creature)
   * Ramp with summoning sickness (creatures) produces mana next turn
   * Graveyard zone tracked for recursion decks
 
 ---- INTERACTION CARD DETECTION ----
 
 Cards whose primary effect targets opponents (removal, counters, discard). In goldfish these have no target, but we simulate casting them on a turn-scaling probability curve to model realistic hand/mana usage.
 * interaction_curve: card_name -> curve_type
 * Interaction types and their per-turn cast probability:
   * "counter":  T1-3: 5%, T4-6: 15%, T7-9: 30%, T10+: 40%
   * "removal":  T1-3: 10%, T4-6: 25%, T7-9: 40%, T10+: 50%
   * "discard":  T1-3: 15%, T4-6: 20%, T7-9: 25%, T10+: 25%
   * "redirect": T1-3: 5%, T4-6: 10%, T7-9: 20%, T10+: 30%
   * "stax":     T1-3: 10%, T4-6: 15%, T7-9: 20%, T10+: 20%



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

### How It Works
* Decklists are parsed and card data is fetched from Scryfall.
* Cards are categorized using Scryfall’s otag: system.
* Simulations run thousands of randomized hands to produce statistical results.

### Disclaimer
This is a fan-made tool for deck analysis and is not affiliated with Wizards of the Coast.
