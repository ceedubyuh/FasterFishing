#!/usr/bin/env python3
"""
FasterFishing
================================
GUI app for goldfishing MTG Commander decks.
- Import decks via URL (Moxfield/Archidekt/MTGGoldfish) or paste text
- Auto-categorize using Scryfall's community oracle tags (otag:ramp, otag:removal, etc.)
- Run Monte Carlo opening hand simulations
- Draw sample hands with card images from Scryfall
- Goldfish turn-by-turn resource tracking

Requirements:  pip install requests Pillow
Usage:         python mtg_goldfish.py
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading, random, os, re, time
from io import BytesIO
import io
from collections import Counter
from dataclasses import dataclass, field

try:
    import requests
except ImportError:
    print("ERROR: pip install requests"); raise SystemExit(1)
try:
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: pip install Pillow"); raise SystemExit(1)

# Categories and the Scryfall otag used to detect them.
# Order matters: first match wins.
CATEGORY_TAGS = {
    "Board Wipe": "boardwipe",
    "Ramp":       "ramp",
    "Draw":       "card-advantage",
    "Removal":    "removal",
    "Tutor":      "tutor",
}
ALL_CATEGORIES = ["Land", "Ramp", "Draw", "Removal", "Board Wipe",
                  "Tutor", "Creature", "Other"]
CATEGORY_COLORS = {
    "Land":"#8B7355", "Ramp":"#2E8B57", "Draw":"#4169E1",
    "Removal":"#DC143C", "Board Wipe":"#FF4500", "Tutor":"#9B59B6",
    "Creature":"#DAA520", "Other":"#708090",
}
# Core MTG permanent types and land subtypes (always valid for "for each [type]" matching).
# Creature/artifact subtypes are extracted dynamically from deck type_lines at sim time.
_CORE_BATTLEFIELD_TYPES = {
    # Permanent supertypes / types
    "creature", "artifact", "enchantment", "planeswalker", "land", "permanent", "token",
    "nonland", "nontoken", "noncreature", "nonartifact",
    # Basic land subtypes
    "plains", "island", "swamp", "mountain", "forest",
}

def _build_battlefield_types(cards):
    """Build the full set of valid battlefield types from actual deck cards.
    Extracts every word from type_lines (lowercased, after splitting on spaces/dashes)
    and merges with core types. This handles any creature subtype (elf, goblin, etc.)
    without needing a hardcoded exhaustive list."""
    types = set(_CORE_BATTLEFIELD_TYPES)
    for c in cards:
        tl = (c.type_line or "").lower()
        # Remove em-dash separator (e.g. "Creature — Elf Warrior")
        tl = tl.replace("\u2014", " ").replace("—", " ").replace("-", " ").replace("//", " ")
        for word in tl.split():
            word = word.strip()
            if len(word) >= 2 and word.isalpha():
                types.add(word)
    return types
# Tutor subtypes for combo probability (not deck categories — more granular)
TUTOR_SUBTYPES = {
    "any":      "Any Card",        # Demonic Tutor, Vampiric Tutor
    "creature": "Creature Tutor",  # Fauna Shaman, Birthing Pod, Chord of Calling
    "land":     "Land Tutor",      # Fetchlands, Crop Rotation
    "instant_sorcery": "Instant/Sorcery Tutor",  # Mystical Tutor, Spellseeker
    "artifact": "Artifact Tutor",  # Fabricate, Whir of Invention
    "enchantment": "Enchantment Tutor",  # Idyllic Tutor, Enlightened Tutor
}

@dataclass
class Card:
    name: str; quantity: int = 1; category: str = "Other"
    mana_cost: str = ""; cmc: float = 0; type_line: str = ""
    image_uri: str = ""; scryfall_id: str = ""; oracle_text: str = ""
    is_commander: bool = False; layout: str = "normal"
    def __hash__(self): return hash(self.name)

@dataclass
class SimResult:
    num_sims: int = 0; hand_size: int = 7
    land_dist: dict = field(default_factory=dict)
    cat_avgs: dict = field(default_factory=dict)
    cat_dists: dict = field(default_factory=dict)
    keepable: float = 0.0
    avg_cmdr_turn: float = 0.0
    hand_quality: float = 0.0  # % of hands that can cast cmdr on curve
    per_cmdr_turns: dict = field(default_factory=dict)  # name -> avg turn
    avg_both_turn: float = 0.0  # turn when all commanders castable
    tutor_tracker: dict = field(default_factory=dict)  # card_name -> avg turn seen/tutored
    ideal_exact: float = 0.0  # % matching ideal hand exactly
    ideal_or_better: float = 0.0  # % matching ideal hand or better (>= each cat)


# ============================================================================
# SCRYFALL CLIENT
# ============================================================================
class ScryfallClient:
    BASE = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "FasterFishing/1.0", "Accept": "application/json"}
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".mtg_goldfish_cache")

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._last_req = 0
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _wait(self):
        elapsed = time.time() - self._last_req
        if elapsed < 0.08: time.sleep(0.08 - elapsed)
        self._last_req = time.time()

    def fetch_collection(self, identifiers):
        """POST /cards/collection - up to 75 per batch."""
        results = []
        for i in range(0, len(identifiers), 75):
            self._wait()
            try:
                r = self.session.post(f"{self.BASE}/cards/collection",
                                      json={"identifiers": identifiers[i:i+75]})
                if r.status_code == 200:
                    results.extend(r.json().get("data", []))
            except Exception: pass
        return results

    def fetch_by_name(self, name):
        """GET /cards/named?fuzzy= - fuzzy single card lookup. Returns card dict or None."""
        self._wait()
        try:
            r = self.session.get(f"{self.BASE}/cards/named",
                                 params={"fuzzy": name, "format": "json"})
            if r.status_code == 200:
                return r.json()
        except Exception: pass
        return None

    def autocomplete(self, query):
        """GET /cards/autocomplete - returns list of card name suggestions."""
        if len(query) < 2: return []
        self._wait()
        try:
            r = self.session.get(f"{self.BASE}/cards/autocomplete",
                                 params={"q": query})
            if r.status_code == 200:
                return r.json().get("data", [])
        except Exception: pass
        return []

    def search_names_with_tag(self, tag, card_names):
        """Query otag:<tag> against a list of card names. Returns matching names (lowercase)."""
        matched = set()
        if not card_names: return matched
        batch_size = 30  # keep URL length reasonable
        for i in range(0, len(card_names), batch_size):
            batch = card_names[i:i+batch_size]
            clauses = " or ".join(f'!"{n}"' for n in batch)
            query = f"otag:{tag} ({clauses})"
            matched.update(self._search_all_names(query))
        return matched

    def _search_all_names(self, query):
        names = set()
        url = f"{self.BASE}/cards/search"
        params = {"q": query, "unique": "cards", "format": "json"}
        while url:
            self._wait()
            try:
                r = self.session.get(url, params=params)
                if r.status_code != 200: break
                data = r.json()
                for card in data.get("data", []):
                    names.add(card.get("name", "").lower())
                if data.get("has_more"):
                    url = data.get("next_page"); params = None
                else: break
            except Exception: break
        return names

    def fetch_image(self, image_uri, card_name):
        safe = re.sub(r'[^\w\-.]', '_', card_name)
        path = os.path.join(self.CACHE_DIR, f"{safe}.jpg")
        if os.path.exists(path):
            try: return Image.open(path)
            except Exception: pass
        # If no image_uri provided, build one from Scryfall named endpoint
        # For MDFCs, the full "A // B" name works with Scryfall's exact search
        if not image_uri:
            lookup_name = card_name
            image_uri = (f"{self.BASE}/cards/named"
                         f"?exact={requests.utils.quote(lookup_name)}&format=image&version=normal")
        self._wait()
        try:
            r = self.session.get(image_uri, timeout=15, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1000:
                img = Image.open(BytesIO(r.content)); img.save(path, "JPEG"); return img
        except Exception: pass
        # Fallback: try with just the first face name if it's a // card
        if " // " in card_name:
            first_face = card_name.split(" // ")[0].strip()
            fallback_uri = (f"{self.BASE}/cards/named"
                            f"?exact={requests.utils.quote(first_face)}&format=image&version=normal")
            self._wait()
            try:
                r = self.session.get(fallback_uri, timeout=15, allow_redirects=True)
                if r.status_code == 200 and len(r.content) > 1000:
                    img = Image.open(BytesIO(r.content)); img.save(path, "JPEG"); return img
            except Exception: pass
        return None


# ============================================================================
# DECK PARSER
# ============================================================================
class DeckParser:
    @staticmethod
    def parse_text(text):
        """Parse decklist text. Returns list of (qty, name, is_commander)."""
        cards = []
        in_commander_section = False
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("//"): continue
            # Detect section headers
            line_lower = line.lower().rstrip(":")
            if line_lower in ("commander", "commanders"):
                in_commander_section = True; continue
            if line_lower in ("deck","sideboard","companion",
                              "maybeboard","mainboard","main","main deck"):
                in_commander_section = False; continue
            # Check for #CMDR marker (strip it from name)
            is_cmdr = False
            if "#cmdr" in line.lower() or "#commander" in line.lower():
                is_cmdr = True
                line = re.sub(r'\s*#(?:cmdr|commander)\b', '', line, flags=re.IGNORECASE).strip()
            elif in_commander_section:
                is_cmdr = True
            # Strip other comments
            if "#" in line:
                line = line[:line.index("#")].strip()
            if not line: continue
            m = re.match(r'^(\d+)x?\s+(.+?)(?:\s+\([A-Za-z0-9]+\)\s*\d+.*)?$', line)
            if m: qty, name = int(m.group(1)), m.group(2).strip()
            else: name = re.sub(r'\s+\([A-Za-z0-9]+\)\s*\d+.*$', '', line).strip(); qty = 1
            if name: cards.append((qty, name, is_cmdr))
        return cards

    @staticmethod
    def fetch_from_moxfield(url):
        m = re.search(r'moxfield\.com/decks/([A-Za-z0-9_-]+)', url)
        if not m: return None
        deck_id = m.group(1)
        api_url = f"https://api2.moxfield.com/v2/decks/all/{deck_id}"

        # Browser-like headers to avoid Cloudflare blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.moxfield.com/",
            "Origin": "https://www.moxfield.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }

        # Attempt 1: Direct request with browser headers
        try:
            r = requests.get(api_url, headers=headers, timeout=15)
            if r.status_code == 200:
                return DeckParser._parse_moxfield_json(r.json())
        except Exception:
            pass

        # Attempt 2: Try with cloudscraper if available (handles Cloudflare JS challenges)
        try:
            import cloudscraper
            scraper = cloudscraper.create_scraper()
            r = scraper.get(api_url, headers=headers, timeout=15)
            if r.status_code == 200:
                return DeckParser._parse_moxfield_json(r.json())
        except ImportError:
            pass  # cloudscraper not installed
        except Exception:
            pass

        # Attempt 3: Try fetching the HTML page and look for embedded JSON data
        try:
            page_url = f"https://www.moxfield.com/decks/{deck_id}"
            r = requests.get(page_url, headers=headers, timeout=15)
            if r.status_code == 200:
                # Moxfield embeds deck data in __NEXT_DATA__ script tag
                nd_match = re.search(r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>', r.text)
                if nd_match:
                    import json
                    next_data = json.loads(nd_match.group(1))
                    # Navigate the Next.js props structure to find deck data
                    props = next_data.get("props", {}).get("pageProps", {})
                    deck = props.get("deck", props)
                    if "mainboard" in deck or "commanders" in deck:
                        return DeckParser._parse_moxfield_json(deck)
        except Exception:
            pass

        return "__MOXFIELD_BLOCKED__"

    @staticmethod
    def _parse_moxfield_json(d):
        """Parse Moxfield JSON deck data into decklist text."""
        lines = []
        for n, e in d.get("commanders", {}).items():
            qty = e.get("quantity", 1)
            card_name = e.get("card", {}).get("name", n) if isinstance(e, dict) and "card" in e else n
            lines.append(f"{qty} {card_name} #CMDR")
        for n, e in d.get("mainboard", {}).items():
            qty = e.get("quantity", 1)
            card_name = e.get("card", {}).get("name", n) if isinstance(e, dict) and "card" in e else n
            lines.append(f"{qty} {card_name}")
        # Some Moxfield responses nest companions/sideboard too
        for n, e in d.get("companions", {}).items():
            qty = e.get("quantity", 1)
            card_name = e.get("card", {}).get("name", n) if isinstance(e, dict) and "card" in e else n
            lines.append(f"{qty} {card_name}")
        return "\n".join(lines) if lines else None

    @staticmethod
    def fetch_from_archidekt(url):
        m = re.search(r'archidekt\.com/(?:api/decks/|decks/)(\d+)', url)
        if not m: return None
        try:
            r = requests.get(f"https://archidekt.com/api/decks/{m.group(1)}/",
                headers={"User-Agent":"FasterFishing/1.0"}, timeout=15)
            if r.status_code == 200:
                lines = []
                for e in r.json().get("cards",[]):
                    n = e.get("card",{}).get("oracleCard",{}).get("name","")
                    if n: lines.append(f"{e.get('quantity',1)} {n}")
                return "\n".join(lines)
        except Exception: pass
        return None

    @staticmethod
    def fetch_from_mtggoldfish(url):
        m = re.search(r'mtggoldfish\.com/deck/(\d+)', url)
        if not m: return None
        try:
            r = requests.get(f"https://www.mtggoldfish.com/deck/download/{m.group(1)}",
                headers={"User-Agent":"FasterFishing/1.0"}, timeout=15)
            if r.status_code == 200: return r.text
        except Exception: pass
        return None

    @classmethod
    def fetch_from_url(cls, url):
        url = url.strip()
        if "moxfield.com" in url: return cls.fetch_from_moxfield(url)
        elif "archidekt.com" in url: return cls.fetch_from_archidekt(url)
        elif "mtggoldfish.com" in url: return cls.fetch_from_mtggoldfish(url)
        return None

# ============================================================================
# CATEGORIZER (uses Scryfall otag: queries)
# ============================================================================
class CardCategorizer:
    @staticmethod
    def categorize_all(cards, scryfall, progress_cb=None):
        # Step 1: Lands from type_line (MDFCs with land back count as lands)
        for c in cards:
            tl = c.type_line.lower()
            if " // " in tl:
                # Multi-face: check for any land face
                faces = [f.strip() for f in tl.split(" // ")]
                any_land = any("land" in f for f in faces)
                if any_land:
                    # Any card with a land face counts as a land for deck/sim purposes
                    # (MDFCs like Shatterskull Smashing // Shatterskull, the Hammer Pass
                    #  are included in decks as flexible land slots)
                    c.category = "Land"
                else:
                    c.category = ""
            elif "land" in tl and "creature" not in tl:
                c.category = "Land"
            else:
                c.category = ""

        nonland = [c for c in cards if c.category != "Land"]
        names = [c.name for c in nonland]
        if not names:
            for c in cards:
                if not c.category: c.category = "Other"
            return

        # Step 2: Query each otag
        tag_results = {}
        items = list(CATEGORY_TAGS.items())
        for i, (cat, tag) in enumerate(items):
            if progress_cb: progress_cb(i / len(items), f"Checking otag:{tag}...")
            tag_results[cat] = scryfall.search_names_with_tag(tag, names)

        # Step 3: Assign (first match wins)
        for c in nonland:
            nl = c.name.lower()
            assigned = False
            for cat in CATEGORY_TAGS:
                if nl in tag_results.get(cat, set()):
                    c.category = cat; assigned = True; break
            if not assigned:
                c.category = "Creature" if "creature" in c.type_line.lower() else "Other"

        # Step 4: Declassify gift-only cards from Draw
        # BLB "Gift a card" gives the opponent a draw, not you.
        # If a card was tagged Draw but its only draw effect is the gift, reclassify.
        for c in nonland:
            if c.category == "Draw":
                oracle = (c.oracle_text or "").lower()
                if "gift" in oracle:
                    # Remove gift reminder text, check if any self-draw remains
                    oracle_no_gift = re.sub(
                        r'gift\s+a\s+card\s*\([^)]*\)\s*', '', oracle)
                    has_self_draw = ("draw" in oracle_no_gift
                        or "into your hand" in oracle_no_gift
                        or "look at the top" in oracle_no_gift)
                    if not has_self_draw:
                        c.category = ("Creature" if "creature" in
                            (c.type_line or "").lower() else "Other")

        if progress_cb: progress_cb(1.0, "Done!")


# ============================================================================
# SIMULATION ENGINE
# ============================================================================
class SimEngine:
    @staticmethod
    def build_deck(cards):
        deck = []
        for c in cards: deck.extend([c] * c.quantity)
        return deck

    @staticmethod
    def _estimate_mana_produced(card):
        """Estimate how much mana a ramp card produces per tap/trigger.
        Parses oracle text for mana production patterns.
        Returns int (default 1 for unrecognized ramp)."""
        oracle = (card.oracle_text or "").lower()

        # Pattern: "add {C}{C}{C}" — count mana symbols in the add clause
        # Find ALL add clauses and take the one that produces the most
        add_clauses = re.findall(r'add\s+((?:\{[^}]+\}\s*)+)', oracle)
        if add_clauses:
            best = 0
            for clause in add_clauses:
                symbols = re.findall(r'\{[^}]+\}', clause)
                best = max(best, len(symbols))
            if best > 0:
                return best

        # Pattern: "add X mana" where X is a number word
        m = re.search(r'add\s+(\w+)\s+mana', oracle)
        if m:
            NUMBER_WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
                            "seven":7,"eight":8,"nine":9,"ten":10}
            w = m.group(1)
            if w in NUMBER_WORDS:
                return NUMBER_WORDS[w]
            if w.isdigit():
                return int(w)

        # Pattern: "add one mana of any color" or "add {C}"
        if re.search(r'add\s+(?:one\s+)?mana\s+of\s+any', oracle):
            return 1
        if re.search(r'adds?\s+\{[^}]+\}', oracle):
            symbols = re.findall(r'\{[^}]+\}', oracle.split("add")[1].split(".")[0]) if "add" in oracle else []
            return max(len(symbols), 1)

        # Pattern: "add an amount of {C} equal to" (Nykthos, Gaea's Cradle, etc.)
        if re.search(r'add\s+(?:an?\s+amount|that\s+much)\s+(?:of\s+)?(?:\{[^}]+\}\s*)?(?:mana\s+)?equal', oracle):
            return 3  # variable — estimate conservatively

        # Default: most ramp produces 1 mana
        return 1

    @staticmethod
    def _parse_color_requirements(mana_cost):
        """Parse mana cost string like '{2}{B}{G}' into color requirements.
        Returns dict of {color: count} for colored pips only (ignores generic).
        E.g. '{1}{B}{B}{G}' -> {'B': 2, 'G': 1}"""
        if not mana_cost:
            return {}
        pips = re.findall(r'\{([^}]+)\}', mana_cost.upper())
        reqs = {}
        for pip in pips:
            if pip in ('W', 'U', 'B', 'R', 'G'):
                reqs[pip] = reqs.get(pip, 0) + 1
            # Hybrid like {W/U} — need either, count as half requirement
            elif '/' in pip:
                parts = pip.split('/')
                colors = [p for p in parts if p in ('W', 'U', 'B', 'R', 'G')]
                if colors:
                    # For sim purposes, treat hybrid as needing any one of the colors
                    reqs.setdefault('hybrid_' + pip, colors)
        return reqs

    @staticmethod
    def _parse_color_production(card):
        """Determine what colors of mana a card can produce.
        Returns a set of color chars, e.g. {'W', 'U'} or {'C'} for colorless.
        'any' means it can produce any color."""
        oracle = (card.oracle_text or "").lower()
        name_l = card.name.lower()
        tl = (card.type_line or "").lower()
        colors = set()

        # Basic lands
        basic_map = {
            "plains": "W", "island": "U", "swamp": "B",
            "mountain": "R", "forest": "G"
        }
        for basic, color in basic_map.items():
            if basic in tl or basic in name_l:
                colors.add(color)

        # Parse "add {X}" from oracle text
        if "add" in oracle:
            add_part = oracle.split("add", 1)[1].split(".")[0]
            for sym in re.findall(r'\{([^}]+)\}', add_part):
                sym_u = sym.upper()
                if sym_u in ('W', 'U', 'B', 'R', 'G'):
                    colors.add(sym_u)
                elif sym_u == 'C':
                    colors.add('C')

        # "any color" / "any one color" / "any combination of colors"
        if re.search(r'(?:any|one)\s+(?:color|combination)', oracle):
            colors.add('any')

        # "mana of any type" / "of any color"
        if "of any type" in oracle or "of any color" in oracle:
            colors.add('any')

        # Mana dorks that add color based on what lands you control
        if "add one mana of any color" in oracle:
            colors.add('any')

        # Cards like Command Tower, City of Brass, Mana Confluence
        if "command tower" in name_l or "city of brass" in name_l or "mana confluence" in name_l:
            colors.add('any')

        # If we found nothing and it's a land, assume colorless
        if not colors and "land" in tl:
            colors.add('C')

        # Ramp artifacts — check oracle for add
        if not colors and card.category == "Ramp":
            if "any color" in oracle or "any type" in oracle:
                colors.add('any')
            elif re.search(r'add\s+\{[cwubrg]\}', oracle):
                pass  # already handled above
            else:
                colors.add('C')  # default ramp = colorless

        return colors if colors else {'C'}

    @staticmethod
    def _can_pay_colors(color_reqs, mana_pool):
        """Check if a mana pool can pay color requirements.
        color_reqs: dict from _parse_color_requirements
        mana_pool: dict of {color: count} available, 'any' = wildcard mana
        Returns True if castable."""
        any_mana = mana_pool.get('any', 0)
        remaining_any = any_mana

        for pip, count in color_reqs.items():
            if pip.startswith('hybrid_'):
                # Hybrid — can pay with any of the listed colors
                hybrid_colors = count  # this is actually the list of colors
                # Need 1 of any listed color
                paid = False
                for hc in hybrid_colors:
                    if mana_pool.get(hc, 0) > 0:
                        paid = True
                        break
                if not paid and remaining_any > 0:
                    remaining_any -= 1
                    paid = True
                if not paid:
                    return False
            else:
                available = mana_pool.get(pip, 0)
                shortfall = count - available
                if shortfall > 0:
                    if remaining_any >= shortfall:
                        remaining_any -= shortfall
                    else:
                        return False
        return True

    @staticmethod
    def sim_hands(cards, n=10000, hs=7, pcb=None, min_mull=4, land_min=2, land_max=5,
                  commander_cmc=0, commander_cmcs=None, tutor_targets=None, ideal_hand=None):
        """Run opening hand simulation with mulligan support.
        
        Args:
            cards: list of Card objects (non-commander deck)
            n: number of simulations
            hs: starting hand size (usually 7)
            min_mull: minimum hand size to mulligan to
            land_min/land_max: range for a keepable hand
            commander_cmc: CMC of commander (0 = no commander cast tracking)
            commander_cmcs: dict of {name: cmc} for per-commander tracking
            tutor_targets: list of card names to track avg turn seen/tutored
            ideal_hand: dict of {category: exact_count} for ideal hand probability
        """
        deck = SimEngine.build_deck(cards)
        results = {}  # hand_size -> SimResult

        # Pre-compute mana production for each ramp card (avoid re-parsing oracle text per sim)
        mana_cache = {}
        for c in cards:
            if c.category == "Ramp" and c.name not in mana_cache:
                mana_cache[c.name] = SimEngine._estimate_mana_produced(c)

        # Pre-identify tutors and their subtypes for tutor target tracking
        tutor_cards = {}  # card_name -> subtype
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            if re.search(r'search\s+your\s+library', oracle):
                subtype = getattr(c, '_tutor_subtype', None)
                if subtype is None:
                    # Quick classify if not already done
                    if re.search(r'search\s+your\s+library\s+for\s+a\s+card\b', oracle):
                        subtype = "any"
                    elif re.search(r'search\s+your\s+library\s+for\s+(?:a|an|up\s+to\s+\w+)\s+creature', oracle):
                        subtype = "creature"
                    elif re.search(r'search\s+your\s+library\s+for\s+(?:a|an|up\s+to\s+\w+)\s+artifact', oracle):
                        subtype = "artifact"
                    elif re.search(r'search\s+your\s+library\s+for\s+(?:a|an)\s+(?:instant|sorcery)', oracle):
                        subtype = "instant_sorcery"
                    elif re.search(r'search\s+your\s+library\s+for\s+(?:a|an|up\s+to\s+\w+)\s+enchantment', oracle):
                        subtype = "enchantment"
                    elif re.search(r'search\s+your\s+library\s+for\s+(?:a|an|up\s+to\s+\w+)\s+(?:basic\s+)?(?:land|mountain|forest|swamp|plains|island)', oracle):
                        if not re.search(r'search\s+your\s+library\s+for\s+(?:a|an)\s+(?:creature|instant|sorcery|enchantment|artifact|card)', oracle):
                            subtype = "land"
                        else:
                            subtype = "any"
                    else:
                        subtype = "any"
                tutor_cards[c.name] = subtype

        # Pre-identify draw spells for the goldfish sub-sim
        draw_cache = {}
        for c in cards:
            if c.name not in draw_cache:
                draw_cache[c.name] = SimEngine._estimate_card_draws(c)

        # Build tutor target type map for matching
        target_types = {}
        if tutor_targets:
            for tname in tutor_targets:
                for c in cards:
                    if c.name.lower() == tname.lower():
                        tl = (c.type_line or "").lower()
                        if "creature" in tl: target_types[tname] = "creature"
                        elif "instant" in tl or "sorcery" in tl: target_types[tname] = "instant_sorcery"
                        elif "artifact" in tl: target_types[tname] = "artifact"
                        elif "enchantment" in tl: target_types[tname] = "enchantment"
                        elif "land" in tl: target_types[tname] = "land"
                        else: target_types[tname] = "other"
                        break

        # Commander mulligan: always draw 7, free mull at 7, then put back
        # mull_num 0 = opening 7 (also represents free mull - same stats)
        # mull_num 1 = mull to 6 (draw 7, put 1 back)
        # mull_num 2 = mull to 5 (draw 7, put 2 back)
        # etc.
        mull_steps = [hs]  # opening 7
        for put_back in range(1, hs - min_mull + 1):
            keep_n = hs - put_back
            if keep_n >= min_mull:
                mull_steps.append(keep_n)

        for keep_count in mull_steps:
            put_back = hs - keep_count

            r = SimResult(num_sims=n, hand_size=keep_count)
            if len(deck) < hs:
                results[keep_count] = r; continue
            lc = Counter(); ct = {c: 0 for c in ALL_CATEGORIES}
            cd = {c: Counter() for c in ALL_CATEGORIES}; keep = 0
            cmdr_turns = []
            # Per-commander tracking for partner/dual commanders
            per_cmdr_turns = {name: [] for name in (commander_cmcs or {})}
            both_cmdr_turns = []  # turn when ALL commanders are castable
            on_curve = 0
            ideal_match = 0
            ideal_or_better = 0
            target_turn_sums = {}
            target_found_counts = {}
            if tutor_targets:
                for tname in tutor_targets:
                    target_turn_sums[tname] = 0
                    target_found_counts[tname] = 0

            for i in range(n):
                # Always draw 7 cards
                hand_full = random.sample(deck, hs)
                
                # Put back cards if mulliganing (keep_count < 7)
                if put_back > 0:
                    # Intelligent put-back: keep best cards for casting commander
                    # Priority: ramp (enables commander), lands (up to reasonable),
                    # draw/tutor, then cheapest spells. Put back excess/expensive.
                    scored = []
                    land_count = sum(1 for c in hand_full if c.category == "Land")
                    for c in hand_full:
                        if c.category == "Land":
                            # Keep lands up to ~CMC count, excess are lower priority
                            ideal_lands = min(commander_cmc, 4) if commander_cmc > 0 else 3
                            scored.append((c, 3 if land_count <= ideal_lands else 0))
                        elif c.category == "Ramp":
                            scored.append((c, 4))  # highest priority - enables commander
                        elif c.category == "Draw":
                            scored.append((c, 2.5))
                        elif c.category == "Tutor":
                            scored.append((c, 2.5))
                        else:
                            # Lower CMC = more useful early
                            scored.append((c, max(0, 2 - c.cmc / 5)))
                    
                    # Sort by priority descending (keep best), break ties by CMC ascending
                    scored.sort(key=lambda x: (-x[1], x[0].cmc))
                    hand = [c for c, _ in scored[:keep_count]]
                else:
                    hand = hand_full

                hc = Counter(c.category for c in hand)
                l = hc.get("Land", 0); lc[l] += 1
                
                # Smart keepable: land count + early plays + draw potential
                hand_keepable = True
                # 1. Land count in reasonable range
                if not (land_min <= l <= land_max):
                    hand_keepable = False
                # 2. Has at least 1 castable spell in first 2 turns
                if hand_keepable:
                    early_plays = sum(1 for c in hand
                                     if c.category != "Land" and c.cmc <= 2 and c.cmc > 0)
                    if early_plays == 0:
                        hand_keepable = False
                # 3. Has at least 1 non-land card (not all lands)
                if hand_keepable:
                    nonland = sum(1 for c in hand if c.category != "Land")
                    if nonland == 0:
                        hand_keepable = False
                # 4. For hands with exactly 1 land, need ramp to recover
                if hand_keepable and l == 1:
                    has_cheap_ramp = any(c.category == "Ramp" and c.cmc <= 1
                                        for c in hand)
                    if not has_cheap_ramp:
                        hand_keepable = False
                # 5. For borderline low-land hands (e.g. 2 lands for 3+ CMC cmdr),
                #    boost keepability if hand has cantrips/draw to find more lands.
                #    A 2-land hand with Preordain or Brainstorm is much more keepable
                #    than a 2-land hand with no card draw.
                if hand_keepable and l <= 2 and commander_cmc >= 3:
                    has_draw_or_cantrip = any(
                        c.category in ("Draw",) and c.cmc <= 2
                        for c in hand)
                    has_ramp = any(c.category == "Ramp" and c.cmc <= 2
                                  for c in hand)
                    if not has_draw_or_cantrip and not has_ramp:
                        hand_keepable = False

                if hand_keepable: keep += 1
                for cat in ALL_CATEGORIES:
                    x = hc.get(cat, 0); ct[cat] += x; cd[cat][x] += 1

                # Check ideal hand match
                if ideal_hand:
                    exact = True
                    at_least = True
                    for cat, target_n in ideal_hand.items():
                        actual = hc.get(cat, 0)
                        if actual != target_n:
                            exact = False
                        if actual < target_n:
                            at_least = False
                    if exact: ideal_match += 1
                    if at_least: ideal_or_better += 1

                # Full goldfish sub-sim for commander cast, hand quality, and tutor tracking
                if commander_cmc > 0 or tutor_targets:
                    # Build library: cards not drawn + put-back cards at bottom
                    not_drawn = [c for c in deck if c not in hand_full]
                    put_back_cards = [c for c in hand_full if c not in hand]
                    remaining = not_drawn
                    random.shuffle(remaining)
                    remaining.extend(put_back_cards)  # put-back goes to bottom
                    turn = 0
                    hand_sim = list(hand)
                    lib = list(remaining)
                    lands_in_play = 0
                    ramp_in_play = []  # (card, turn_cast)
                    battlefield_sim = []
                    cmdr_cast_turn = 20
                    # Per-commander tracking
                    per_cmdr_cast = {name: 20 for name in (commander_cmcs or {})}
                    found_targets = set()  # targets found this game
                    max_turns = max(20, (commander_cmc or 3) + 5)

                    # Check if targets are in opening hand
                    if tutor_targets:
                        for c in hand_sim:
                            if c.name in target_turn_sums and c.name not in found_targets:
                                target_turn_sums[c.name] += 0  # turn 0 = opening hand
                                target_found_counts[c.name] += 1
                                found_targets.add(c.name)

                    while turn < max_turns:
                        turn += 1
                        # Draw for the turn (Commander: all players draw on turn 1)
                        if lib:
                            drawn = lib.pop(0)
                            hand_sim.append(drawn)
                            # Check if drawn card is a tutor target
                            if tutor_targets and drawn.name in target_turn_sums and drawn.name not in found_targets:
                                target_turn_sums[drawn.name] += turn
                                target_found_counts[drawn.name] += 1
                                found_targets.add(drawn.name)

                        # Play a land if we have one
                        land_in_hand = [c for c in hand_sim if c.category == "Land"]
                        if land_in_hand:
                            hand_sim.remove(land_in_hand[0])
                            lands_in_play += 1

                        # Calculate available mana this turn
                        available_mana = lands_in_play
                        for rc, cast_turn in ramp_in_play:
                            mana_val = mana_cache.get(rc.name, 1)
                            tl_r = (rc.type_line or "").lower()
                            if "creature" in tl_r:
                                if turn > cast_turn:
                                    available_mana += mana_val
                            else:
                                if turn > cast_turn:
                                    available_mana += mana_val

                        # Can we cast commander?
                        if commander_cmc > 0 and available_mana >= commander_cmc and cmdr_cast_turn == 20:
                            cmdr_cast_turn = turn
                        # Per-commander tracking: each partner can be cast independently
                        if commander_cmcs:
                            for cname, ccmc in commander_cmcs.items():
                                if per_cmdr_cast[cname] == 20 and available_mana >= ccmc:
                                    per_cmdr_cast[cname] = turn

                        # Cast ramp spells with leftover mana (prioritize cheapest)
                        mana_left = available_mana
                        # Don't spend mana we need for commander this turn
                        if commander_cmc > 0 and cmdr_cast_turn == 20:
                            pass  # already checked commander above

                        cast_cmdr_this_turn = False
                        castable = sorted(
                            [c for c in hand_sim if c.category == "Ramp"
                             and 0 < c.cmc <= mana_left],
                            key=lambda x: x.cmc)
                        for rc in castable:
                            if rc.cmc <= mana_left:
                                hand_sim.remove(rc)
                                mana_left -= int(rc.cmc)
                                ramp_in_play.append((rc, turn))
                                tl_r = (rc.type_line or "").lower()
                                if "creature" not in tl_r:
                                    mana_val = mana_cache.get(rc.name, 1)
                                    mana_left += mana_val
                                    if commander_cmc > 0 and cmdr_cast_turn == 20:
                                        total_now = lands_in_play
                                        for r2, ct2 in ramp_in_play:
                                            mv2 = mana_cache.get(r2.name, 1)
                                            tl2 = (r2.type_line or "").lower()
                                            if "creature" in tl2:
                                                if turn > ct2: total_now += mv2
                                            else:
                                                total_now += mv2
                                        if total_now >= commander_cmc:
                                            cmdr_cast_turn = turn
                                            cast_cmdr_this_turn = True
                                            break
                                    # Per-commander recheck after ramp
                                    if commander_cmcs:
                                        total_now2 = lands_in_play
                                        for r2, ct2 in ramp_in_play:
                                            mv2 = mana_cache.get(r2.name, 1)
                                            tl2 = (r2.type_line or "").lower()
                                            if "creature" in tl2:
                                                if turn > ct2: total_now2 += mv2
                                            else:
                                                total_now2 += mv2
                                        for cname, ccmc in commander_cmcs.items():
                                            if per_cmdr_cast[cname] == 20 and total_now2 >= ccmc:
                                                per_cmdr_cast[cname] = turn
                        # Cast draw/tutor spells with remaining mana
                        mana_left2 = mana_left
                        draw_cast = sorted(
                            [c for c in hand_sim if c.category in ("Draw", "Tutor")
                             and 0 < c.cmc <= mana_left2],
                            key=lambda x: x.cmc)
                        for dc in draw_cast:
                            if dc.cmc <= mana_left2:
                                hand_sim.remove(dc)
                                mana_left2 -= int(dc.cmc)
                                # Draw spell: draw cards from library
                                dn, rep, _, _lc = draw_cache.get(dc.name, (0, False, "", 0))
                                if dn > 0 and not rep:
                                    for _ in range(int(dn)):
                                        if lib:
                                            drawn = lib.pop(0)
                                            hand_sim.append(drawn)
                                            if tutor_targets and drawn.name in target_turn_sums and drawn.name not in found_targets:
                                                target_turn_sums[drawn.name] += turn
                                                target_found_counts[drawn.name] += 1
                                                found_targets.add(drawn.name)
                                # Tutor: find best target in library
                                if dc.name in tutor_cards and tutor_targets:
                                    tsub = tutor_cards[dc.name]
                                    # Find unfound targets that this tutor can find
                                    for tname in tutor_targets:
                                        if tname in found_targets:
                                            continue
                                        ttype = target_types.get(tname, "other")
                                        can_find = (tsub == "any" or tsub == ttype
                                                    or (tsub == "artifact_enchantment" and ttype in ("artifact", "enchantment")))
                                        if can_find:
                                            # Check if target is in library
                                            for lc2 in lib:
                                                if lc2.name.lower() == tname.lower():
                                                    target_turn_sums[tname] += turn
                                                    target_found_counts[tname] += 1
                                                    found_targets.add(tname)
                                                    lib.remove(lc2)
                                                    hand_sim.append(lc2)
                                                    break
                                            break  # one tutor finds one card
                                battlefield_sim.append(dc)

                        # Upkeep draw engines already on battlefield
                        for perm in battlefield_sim:
                            dn, rep, _, _lc = draw_cache.get(perm.name, (0, False, "", 0))
                            if rep and dn > 0:
                                for _ in range(int(dn)):
                                    if lib:
                                        drawn = lib.pop(0)
                                        hand_sim.append(drawn)
                                        if tutor_targets and drawn.name in target_turn_sums and drawn.name not in found_targets:
                                            target_turn_sums[drawn.name] += turn
                                            target_found_counts[drawn.name] += 1
                                            found_targets.add(drawn.name)

                        # Stop early if commander cast and all targets found
                        all_cmdrs_cast = all(v < 20 for v in per_cmdr_cast.values()) if per_cmdr_cast else True
                        if cmdr_cast_turn < 20 and all_cmdrs_cast and (not tutor_targets or len(found_targets) >= len(tutor_targets)):
                            break

                    if commander_cmc > 0:
                        cmdr_turns.append(cmdr_cast_turn)
                        if cmdr_cast_turn <= commander_cmc:
                            on_curve += 1
                    # Per-commander accumulation
                    if commander_cmcs:
                        for cname in commander_cmcs:
                            per_cmdr_turns[cname].append(per_cmdr_cast[cname])
                        if len(commander_cmcs) > 1:
                            both_cmdr_turns.append(max(per_cmdr_cast.values()))

                if pcb and i % 1000 == 0: pcb(i / n)
            r.land_dist = {k: v / n * 100 for k, v in sorted(lc.items())}
            r.cat_avgs = {c: t / n for c, t in ct.items()}
            r.cat_dists = {c: {k: v / n * 100 for k, v in sorted(d.items())} for c, d in cd.items()}
            r.keepable = keep / n * 100
            r.avg_cmdr_turn = sum(cmdr_turns) / len(cmdr_turns) if cmdr_turns else 0
            r.hand_quality = on_curve / n * 100 if n > 0 else 0
            # Per-commander averages
            r.per_cmdr_turns = {}
            for cname, turns_list in per_cmdr_turns.items():
                r.per_cmdr_turns[cname] = sum(turns_list) / len(turns_list) if turns_list else 0
            r.avg_both_turn = sum(both_cmdr_turns) / len(both_cmdr_turns) if both_cmdr_turns else 0
            r.ideal_exact = ideal_match / n * 100 if ideal_hand and n > 0 else 0
            r.ideal_or_better = ideal_or_better / n * 100 if ideal_hand and n > 0 else 0
            if tutor_targets:
                r.tutor_tracker = {}
                for tname in tutor_targets:
                    cnt = target_found_counts.get(tname, 0)
                    avg = target_turn_sums[tname] / cnt if cnt > 0 else -1
                    r.tutor_tracker[tname] = {
                        "avg_turn": avg,
                        "found_pct": cnt / n * 100,
                    }
            results[keep_count] = r
        if pcb: pcb(1.0)
        return results

    @staticmethod
    def _estimate_card_draws(card):
        """Estimate how many bonus cards a card draws/selects when played.
        Returns (draw_count, is_repeating, label, life_cost) where:
          - draw_count: avg cards added to hand per trigger/activation
          - is_repeating: True if it triggers every turn
          - label: description string for the draw source list
          - life_cost: life paid per draw cycle (0 = free, >0 = costs life)
            Used to flag engines like Sylvan Library, One Ring, Necropotence
            whose practical output should be halved in analysis.
        """
        oracle = card.oracle_text.lower() if card.oracle_text else ""
        tl = card.type_line.lower() if card.type_line else ""
        name_l = card.name.lower()
        is_permanent = "instant" not in tl and "sorcery" not in tl

        # Early check — need "draw", "look at", "reveal", "mill", "exile the top",
        # "put into your hand", or equipment/sacrifice patterns
        has_draw_keyword = any(kw in oracle for kw in [
            "draw", "look at the top", "reveal the top", "mill",
            "exile the top", "put into your hand", "into your hand",
            "surveil", "equipped creature dies", "whenever a creature",
            "whenever a nontoken creature",
            "cast spells from the top", "play the top", "play lands and cast",
            "cast the top card", "play cards from the top",
        ])
        if not oracle or not has_draw_keyword:
            return (0, False, "", 0)

        # ---- GIFT MECHANIC EXCLUSION ----
        # BLB "Gift a card" — the opponent draws, not you
        # If the only "draw" in the text is inside a gift clause, skip
        if "gift" in oracle:
            # Remove gift reminder text to check if there's a real draw left
            oracle_no_gift = re.sub(r'gift\s+a\s+card[^.]*?(?:they\s+draw\s+a\s+card|opponent\s+draws?\s+a\s+card)[^.)]*[.)]', '', oracle)
            if "draw" not in oracle_no_gift and "into your hand" not in oracle_no_gift:
                return (0, False, "", 0)

        # Exclude "opponent draws" / "they draw" / "target opponent draws"
        # In goldfish sim, "target player" is always us — only exclude opponent-specific
        # Only exclude if the ONLY draw in the text is for opponents, not us
        oracle_no_opp = re.sub(r'(?:they|that player|each opponent|target\s+opponent)\s+draws?\s+[^.]*\.', '', oracle)
        # Also remove trigger conditions like "whenever an opponent draws"
        oracle_no_opp = re.sub(r'whenever\s+an?\s+opponent\s+draws?\s+[^,]*,?\s*', '', oracle_no_opp)
        if "draw" not in oracle_no_opp and "draw" in oracle:
            if "into your hand" not in oracle:
                return (0, False, "", 0)

        NUMBER_WORDS = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                        "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        def parse_n(text):
            m = re.search(r'draws?\s+(\w+)\s+cards?', text)
            if m:
                w = m.group(1)
                if w == "x": return -1  # X-dependent: resolved at cast time
                if w in NUMBER_WORDS: return NUMBER_WORDS[w]
                if w.isdigit(): return int(w)
            if "draw cards equal to" in text or "draw that many" in text: return 2
            if "draw a card" in text: return 1
            return 0

        # ---- SELECTION EFFECTS (look at top N, pick some) ----
        # "look at the top N cards... put one/any into your hand"
        look_match = re.search(r'look at the top\s+(\w+)\s+cards?', oracle)
        if look_match and "into your hand" in oracle:
            w = look_match.group(1)
            depth = NUMBER_WORDS.get(w, int(w) if w.isdigit() else 3)
            # Selection = seeing N but only keeping ~1, effective draw ≈ 0.8-1
            # (better than 1 draw since you pick the best card)
            is_rep = is_permanent and ("whenever" in oracle or "beginning" in oracle or "{t}" in oracle)
            label = f"look at top {depth}, pick 1 (selection)"
            return (1, is_rep, label, 0)

        # "reveal the top N cards... put [type] into your hand"
        reveal_match = re.search(r'reveal the top\s+(\w+)\s+cards?', oracle)
        if reveal_match and ("into your hand" in oracle or "put it into your hand" in oracle):
            w = reveal_match.group(1)
            depth = NUMBER_WORDS.get(w, int(w) if w.isdigit() else 3)
            is_rep = is_permanent and ("whenever" in oracle or "beginning" in oracle)
            # Conditional — depends on hitting the right type, ~60-70% of the time
            label = f"reveal top {depth}, conditional pick (~0.7)"
            return (0.7, is_rep, label, 0)

        # ---- MILL-AND-PICK (Ripples of Undeath style) ----
        # "mill N... you may put a card from your graveyard into your hand"
        if "mill" in oracle and ("into your hand" in oracle or "put it into your hand" in oracle):
            mill_match = re.search(r'mill\s+(\w+)', oracle)
            depth = 2
            if mill_match:
                w = mill_match.group(1)
                depth = NUMBER_WORDS.get(w, int(w) if w.isdigit() else 2)
            is_rep = is_permanent and ("beginning" in oracle or "upkeep" in oracle)
            label = f"mill {depth}, pick 1 (graveyard selection)"
            return (1, is_rep, label, 0)

        # ---- SURVEIL-AND-DRAW ----
        if "surveil" in oracle and "draw" in oracle:
            surv_match = re.search(r'surveil\s+(\w+)', oracle)
            depth = 2
            if surv_match:
                w = surv_match.group(1)
                depth = NUMBER_WORDS.get(w, int(w) if w.isdigit() else 2)
            is_rep = is_permanent and "whenever" in oracle
            n = parse_n(oracle)
            label = f"surveil {depth} + draw {max(n,1)}"
            return (max(n, 1), is_rep, label, 0)

        # ---- REPLACEMENT EFFECTS (Underrealm Lich style) ----
        # "look at the top three cards... put one into your hand and the rest into your graveyard"
        if ("instead" in oracle or "if you would draw" in oracle) and "look at" in oracle:
            look_m = re.search(r'top\s+(\w+)\s+cards?', oracle)
            depth = 3
            if look_m:
                w = look_m.group(1)
                depth = NUMBER_WORDS.get(w, int(w) if w.isdigit() else 3)
            # Replacement: doesn't add cards, but greatly improves quality
            # Model as ~0.5 effective extra draw (selection advantage)
            label = f"replacement: see {depth}, pick 1 (quality boost)"
            return (0.5, True, label, 0)

        # ---- EQUIPMENT DEATH TRIGGERS (Skullclamp) ----
        if "equipment" in tl:
            if re.search(r'(?:equipped creature dies|leaves the battlefield)[^.]*?draw', oracle):
                n = parse_n(oracle)
                # Skullclamp-style: repeating if you have creatures to sacrifice
                # Model as repeating ~1.5/turn on average (need creatures)
                label = f"equipment death trigger (draw {max(n,2)}, needs creatures)"
                return (max(n, 2), True, label, 0)

        # ---- CREATURE DEATH TRIGGERS (general) ----
        if is_permanent and re.search(r'whenever\s+(?:a|another)\s+(?:nontoken\s+)?creature[^.]*?(?:dies|is put into)[^.]*?draw', oracle):
            n = parse_n(oracle)
            label = "creature death draw (~1/turn)"
            return (max(n, 1), True, label, 0)

        # ---- EXILE TOP AND PLAY (impulse draw) ----
        if re.search(r'exile the top\s+\w+\s+cards?[^.]*?(?:you may (?:play|cast)|until end of turn)', oracle):
            ex_match = re.search(r'exile the top\s+(\w+)\s+cards?', oracle)
            depth = 1
            if ex_match:
                w = ex_match.group(1)
                depth = NUMBER_WORDS.get(w, int(w) if w.isdigit() else 1)
            is_rep = is_permanent and ("beginning" in oracle or "upkeep" in oracle or "whenever" in oracle)
            # Impulse draw: ~60-70% actually castable
            effective = max(depth * 0.6, 0.5)
            label = f"impulse exile {depth} (~{effective:.1f} effective)"
            return (effective, is_rep, label, 0)

        tl = card.type_line.lower() if card.type_line else ""
        name_l = card.name.lower()
        is_permanent = "instant" not in tl and "sorcery" not in tl

        # Detect life-cost draw engines (generalized from oracle text)
        pays_life = False
        life_per_cycle = 0

        # Pattern 1: "pay N life" anywhere in oracle (Greed, Erebos, etc.)
        m_pay = re.search(r'pay\s+(\d+)\s+life', oracle)
        if m_pay:
            pays_life = True
            life_per_cycle = int(m_pay.group(1))

        # Pattern 2: "lose N life" tied to draw/card advantage
        m_lose = re.search(r'(?:you\s+)?loses?\s+(\d+)\s+life', oracle)
        if m_lose and ("draw" in oracle or "into your hand" in oracle or "play" in oracle):
            pays_life = True
            life_per_cycle = max(life_per_cycle, int(m_lose.group(1)))

        # Pattern 3: "you lose life equal to" something variable
        if re.search(r'(?:you\s+)?lose\s+life\s+equal\s+to', oracle):
            pays_life = True
            if "mana value" in oracle or "mana cost" in oracle or "converted mana cost" in oracle:
                life_per_cycle = max(life_per_cycle, 3)  # avg CMC ~3
            elif "power" in oracle:
                life_per_cycle = max(life_per_cycle, 3)
            elif "number of" in oracle and "counter" in oracle:
                life_per_cycle = max(life_per_cycle, 3)  # cumulative counters
            else:
                life_per_cycle = max(life_per_cycle, 2)

        # Pattern 4: "pay life equal to its/their/that spell's mana cost/value"
        if re.search(r'pay\s+life\s+equal\s+to\s+(?:its|their|that\s+card|that\s+spell)', oracle):
            pays_life = True
            life_per_cycle = max(life_per_cycle, 3)  # avg CMC ~3

        # Pattern 5: "draw two additional cards... unless you pay N life for each"
        # (Sylvan Library pattern: optional life payment for extra draws)
        m_optional = re.search(r'(?:additional|extra)\s+.*?(?:pay|unless).*?(\d+)\s+life', oracle)
        if m_optional and "draw" in oracle:
            pays_life = True
            n_extra = 2  # default assume 2 extra cards
            m_extra = re.search(r'(\w+)\s+additional', oracle)
            if m_extra:
                w = m_extra.group(1)
                n_extra = {"two": 2, "three": 3, "one": 1}.get(w, int(w) if w.isdigit() else 2)
            life_per_cycle = max(life_per_cycle, int(m_optional.group(1)) * n_extra)

        # Pattern 6: "burden counter" / cumulative upkeep with life loss
        if re.search(r'(?:burden|age|fade)\s+counter', oracle) and re.search(r'(?:lose|pay)\s+.*?life', oracle):
            pays_life = True
            life_per_cycle = max(life_per_cycle, 3)

        # Pattern 7: Generic "pay life: draw" activated ability
        if not pays_life and re.search(r'pay\s+\d+\s+life[^.]*?draw', oracle):
            pays_life = True
            if m_pay:
                life_per_cycle = int(m_pay.group(1))

        # ---- REVEAL-TOP-CARD (singular): Dark Confidant, Ad Nauseam ----
        # "reveal the top card... put that card into your hand"
        if re.search(r'reveal\s+the\s+top\s+card', oracle) and "into your hand" in oracle:
            is_rep = is_permanent and ("whenever" in oracle or "beginning" in oracle)
            lc = life_per_cycle if pays_life else 0
            label = f"reveal top (costs ~{lc} life)" if pays_life else "reveal top, put into hand"
            return (1, is_rep, label, lc)

        # ---- SACRIFICE-TO-DRAW: one-shot, not repeating ----
        if re.search(r'sacrifice\s+.{0,20}:\s*[^.]*?draw', oracle):
            n = parse_n(oracle)
            return (max(n, 1), False, "sacrifice to draw", 0)

        # ---- CUMULATIVE DRAW (counter-scaling) ----
        # "draw a card for each [type] counter" or "draw cards equal to number of counters"
        # Covers: The One Ring (burden), Fathom Mage (evolve), etc.
        if re.search(r'draw\s+(?:a\s+card|cards)\s+(?:for\s+each|equal\s+to)[^.]*?counter', oracle):
            return (3, True, "cumulative (~3/turn avg, costs life)" if pays_life else "cumulative (~3/turn avg)", life_per_cycle)

        # ---- PAY-PER-CARD REPEATABLE (Necropotence-style) ----
        # "pay 1 life: exile the top card... put into your hand" / "pay 1 life, draw a card"
        # Any activated ability where you pay life per card, can activate many times
        if pays_life and life_per_cycle <= 1 and re.search(
            r'(?:pay\s+1\s+life[^.]*?(?:exile\s+the\s+top|draw|put\s+[^.]*?into\s+your\s+hand)|'
            r'pay\s+1\s+life\s*:\s*draw)', oracle):
            # Low cost per card = can draw many per turn
            return (5, True, f"pay {life_per_cycle} life each (repeatable)", life_per_cycle)

        # ---- PLAY-FROM-TOP (Bolas's Citadel, Future Sight, Mystic Forge) ----
        # "you may play/cast [spells/cards] from the top of your library"
        if re.search(r'(?:play|cast)\s+(?:cards?\s+from\s+the\s+top|the\s+top\s+card|spells?\s+from\s+the\s+top|lands\s+and\s+cast)', oracle):
            lc = life_per_cycle if pays_life else 0
            label = f"play from top (costs ~{lc} life avg)" if pays_life else "play from top (~2 cards/turn)"
            return (2, True, label, lc)

        # ---- HAND-SIZE DEPENDENT ----
        if re.search(r'draw\s+cards?\s+equal\s+to\s+(?:the\s+number\s+of\s+)?cards?\s+in\s+your\s+hand', oracle):
            return (4, False, "draws ≈ hand size (~4)", 0)

        # ---- VARIABLE X DRAW ----
        if re.search(r'draws?\s+x\s+cards', oracle):
            lc = life_per_cycle if pays_life else 0
            return (-1, False, "draws X (scales with mana)", lc)
        # "draw half X cards" (Hydroid Krasis)
        if re.search(r'draw\s+half\s+x\s+cards', oracle):
            lc = life_per_cycle if pays_life else 0
            return (-2, False, "draws half X (scales with mana)", lc)
        if re.search(r'draw\s+cards?\s+equal\s+to\s+[^.]*?(?:power|toughness|mana\s+value)', oracle):
            return (3, False, "draws ≈ power/toughness (~3)", 0)

        # ---- DRAW THAT MANY ----
        if "draw that many" in oracle:
            lc = life_per_cycle if pays_life else 0
            return (2, True, "draws that many (~2/trigger)", lc)

        # ---- Upkeep/repeating draw (Phyrexian Arena, Sylvan Library) ----
        if re.search(r'(?:at the beginning of|during) your (?:upkeep|draw step)[^.]*?draw', oracle):
            n = parse_n(oracle)
            if "two additional" in oracle: n = max(n, 2)
            lc = life_per_cycle if pays_life else 0
            label = "upkeep draw"
            if pays_life:
                label = f"upkeep draw (costs {life_per_cycle} life/turn)"
            return (max(n, 1), True, label, lc)

        # ---- Opponent-action triggers ----
        if re.search(r'whenever\s+an?\s+opponent\s+casts[^.]*?draw', oracle):
            if "unless" in oracle or "may pay" in oracle:
                return (1, True, "opponent cast (~1/turn, may not pay)", 0)
            return (2, True, "opponent cast (~2/turn)", 0)

        if re.search(r'whenever\s+an?\s+artifact[^.]*?(?:opponent|graveyard)[^.]*?draw', oracle):
            return (1, True, "opponent artifact dies (~1/turn)", 0)
        if re.search(r'(?:opponent|graveyard)[^.]*?(?:from the battlefield|is put into)[^.]*?draw', oracle):
            return (1, True, "opponent permanent dies (~1/turn)", 0)

        # ---- COMBAT DAMAGE DRAW ----
        if re.search(r'(?:deals?\s+combat\s+damage|deals?\s+damage\s+to\s+(?:a\s+)?(?:player|opponent))[^.]*?draw', oracle):
            n = parse_n(oracle)
            return (max(n, 1), True, "combat damage draw (~1/turn)", 0)

        # ---- "Whenever you cast" triggers on permanents ----
        if is_permanent and re.search(r'whenever\s+you\s+cast[^.]*?draw', oracle):
            return (1, True, "cast trigger (~1/turn)", 0)

        # ---- Activated abilities: "{T}: draw" ----
        tap_draw = re.search(r'\{t\}[^:]*:\s*[^.]*?draw', oracle)
        if tap_draw:
            pre_tap = oracle.split("{t}")[0] if "{t}" in oracle else ""
            if "sacrifice" in pre_tap:
                return (parse_n(oracle) or 1, False, "sacrifice to draw", 0)
            n = parse_n(oracle)
            lc = life_per_cycle if pays_life else 0
            label = "tap to draw"
            if pays_life:
                label = f"tap to draw (costs life)"
            return (max(n, 1), True, label, lc)

        # ---- Pay life to draw (Greed, Erebos, etc.) ----
        if pays_life and re.search(r'pay\s+\d+\s+life[^.]*?draw', oracle):
            n = parse_n(oracle)
            return (max(n, 1), True, f"pay {life_per_cycle} life to draw", life_per_cycle)

        # ---- ETB draw ----
        if re.search(r'when\s+.{0,30}\s+enters?[^.]*?draw', oracle):
            return (parse_n(oracle) or 1, False, "ETB draw", 0)

        # ---- Instant/sorcery cantrip or draw spell ----
        if not is_permanent:
            n = parse_n(oracle)
            # Handle X-dependent draw: parse_n returns -1 for X
            if n == -1:
                lc = life_per_cycle if pays_life else 0
                return (-1, False, "draws X (scales with mana)", lc)
            putback = re.search(r'put\s+(\w+)\s+(?:of them|cards?)\s+(?:back\s+)?(?:on\s+top|from your hand)', oracle)
            if putback:
                pb_word = putback.group(1)
                pb = NUMBER_WORDS.get(pb_word, int(pb_word) if pb_word.isdigit() else 0)
                n = max(0, n - pb)
            if n > 0:
                lc = life_per_cycle if pays_life else 0
                label = "spell draw" if not pays_life else f"spell draw (costs {lc} life)"
                return (n, False, label, lc)

        # ---- Generic fallback ----
        if is_permanent and "draw a card" in oracle:
            if "whenever" in oracle:
                lc = life_per_cycle if pays_life else 0
                return (1, True, "triggered draw (~1/turn)", lc)
            return (1, False, "draw", 0)

        return (0, False, "", 0)

    @staticmethod
    def sim_goldfish(cards, n=1000, turns=10, pcb=None, combo_pieces=None, commanders=None):
        """Run goldfish simulation with smart casting priorities.
        combo_pieces: optional list of sets, each set being card names in a combo.
        commanders: optional list of Card objects for commanders (always accessible from command zone)."""
        deck = SimEngine.build_deck(cards)
        if len(deck) < 7 + turns: return {}
        td = {t:{c:0 for c in ALL_CATEGORIES} for t in range(turns+1)}
        ld = {t:0 for t in range(turns+1)}
        bonus_draws = {t: 0 for t in range(turns+1)}
        total_cards = {t: 0 for t in range(turns+1)}
        # Extended stats
        avail_mana = {t: 0 for t in range(turns+1)}       # total mana available (lands + ramp)
        hand_size = {t: 0 for t in range(turns+1)}          # cards in hand
        bf_creatures = {t: 0 for t in range(turns+1)}       # creatures on battlefield
        bf_artifacts = {t: 0 for t in range(turns+1)}       # artifacts on battlefield
        bf_enchantments = {t: 0 for t in range(turns+1)}    # enchantments on battlefield
        bf_total = {t: 0 for t in range(turns+1)}           # total non-land permanents
        token_count = {t: 0 for t in range(turns+1)}        # estimated tokens created
        combat_power = {t: 0 for t in range(turns+1)}       # total creature power on bf
        cumul_damage = {t: 0 for t in range(turns+1)}       # cumulative combat damage dealt
        cards_drawn_this_turn = {t: 0 for t in range(turns+1)}  # cards drawn THIS turn
        gy_size = {t: 0 for t in range(turns+1)}             # graveyard size

        # Pre-compute draw estimates for each unique card
        draw_cache = {}
        for c in cards:
            if c.name not in draw_cache:
                draw_cache[c.name] = SimEngine._estimate_card_draws(c)

        # Pre-compute mana production for ramp
        mana_cache = {}
        for c in cards:
            if c.category == "Ramp" and c.name not in mana_cache:
                mana_cache[c.name] = SimEngine._estimate_mana_produced(c)

        # Pre-compute token generation: card_name -> (tokens_per_trigger, repeating)
        token_cache = {}
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            if c.name not in token_cache and "create" in oracle:
                n_tok = 1
                m = re.search(r'create\s+(\w+)\s+', oracle)
                if m:
                    w = m.group(1)
                    n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"x":-1}.get(w, int(w) if w.isdigit() else 1)
                repeating = "whenever" in oracle or "at the beginning" in oracle
                token_cache[c.name] = (n_tok, repeating)

        # Pre-compute creature power: card_name -> power (for combat clock)
        power_cache = {}
        for c in cards:
            tl = (c.type_line or "").lower()
            if "creature" in tl and c.name not in power_cache:
                oracle = c.oracle_text or ""
                # Try to parse power from oracle text (P/T line) or guess from CMC
                pw = 0
                m = re.search(r'(\d+)/(\d+)', oracle[-20:]) if oracle else None
                if m:
                    pw = int(m.group(1))
                else:
                    # Rough estimate: power ~= CMC - 1 for most creatures
                    pw = max(1, int(c.cmc) - 1)
                power_cache[c.name] = pw

        # Pre-compute drain/ping effects: card_name -> (damage_per_trigger, trigger_type)
        # trigger_type: "death" (Blood Artist), "etb_creature" (Purphoros), "token" (Impact Tremors)
        drain_cache = {}
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            if c.name in drain_cache:
                continue
            dmg = 0; trig = None
            # "whenever a creature dies" / "whenever a creature you control dies" -> drain
            if re.search(r'whenever\s+(?:a|another)\s+creature\s+(?:you\s+control\s+)?dies', oracle):
                if re.search(r'(?:each\s+opponent\s+)?loses?\s+(\d+)\s+life', oracle):
                    m = re.search(r'loses?\s+(\d+)\s+life', oracle)
                    dmg = int(m.group(1)) if m else 1
                elif re.search(r'deals?\s+(\d+)\s+damage\s+to\s+(?:any|each|target)', oracle):
                    m = re.search(r'deals?\s+(\d+)\s+damage', oracle)
                    dmg = int(m.group(1)) if m else 1
                elif "drain" in oracle or "lose" in oracle:
                    dmg = 1
                if dmg > 0:
                    trig = "death"
            # "whenever a creature enters" / "whenever another creature enters"
            if not trig and re.search(r'whenever\s+(?:a|another)\s+(?:creature|nontoken\s+creature)\s+(?:you\s+control\s+)?enters', oracle):
                if re.search(r'deals?\s+(\d+)\s+damage\s+to\s+each\s+opponent', oracle):
                    m = re.search(r'deals?\s+(\d+)\s+damage', oracle)
                    dmg = int(m.group(1)) if m else 1
                elif re.search(r'(?:each\s+opponent\s+)?loses?\s+(\d+)\s+life', oracle):
                    m = re.search(r'loses?\s+(\d+)\s+life', oracle)
                    dmg = int(m.group(1)) if m else 1
                if dmg > 0:
                    trig = "etb_creature"
            # "whenever a token" / "whenever you create a token"
            if not trig and re.search(r'whenever\s+(?:a|one\s+or\s+more)\s+tokens?\s+(?:enters|you\s+create)', oracle):
                if re.search(r'deals?\s+(\d+)\s+damage|loses?\s+(\d+)\s+life', oracle):
                    m = re.search(r'(?:deals?\s+|loses?\s+)(\d+)', oracle)
                    dmg = int(m.group(1)) if m else 1
                    trig = "token"
            # "at the beginning of your upkeep/end step" drain
            if not trig and re.search(r'at\s+the\s+beginning\s+of\s+(?:your|each)', oracle):
                if re.search(r'each\s+opponent\s+loses?\s+(\d+)\s+life', oracle):
                    m = re.search(r'loses?\s+(\d+)\s+life', oracle)
                    dmg = int(m.group(1)) if m else 1
                    trig = "upkeep"
            if trig:
                drain_cache[c.name] = (dmg, trig)

        # Pre-compute ETB effects (for blink/reanimate synergy)
        # etb_token_cache: card_name -> number of tokens created on ETB
        # etb_value_cache: card_name -> dict of {action: param} for all ETB effects
        etb_token_cache = {}
        etb_value_cache = {}  # card_name -> {"tokens": N, "draw": N, "damage": N, "ramp": N, "lifegain": N, "counter": N}
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            tl = (c.type_line or "").lower()
            is_spell = "instant" in tl or "sorcery" in tl

            # ---- INSTANT/SORCERY ONE-SHOT EFFECTS ----
            # These don't have "enters the battlefield" — the whole text is the effect
            if is_spell:
                effects = {}
                # Token creation: "Create X 1/1 [type] creature tokens"
                if "create" in oracle and "token" in oracle:
                    m = re.search(r'create\s+(\w+)\s+', oracle)
                    if m:
                        w = m.group(1)
                        n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                                 "six":6,"seven":7,"eight":8,"x":-1}.get(w, int(w) if w.isdigit() else 1)
                        effects["tokens"] = n_tok
                        etb_token_cache[c.name] = n_tok
                # Damage: "deals X/N damage to [target]"
                m_dmg = re.search(r'deals?\s+(\w+)\s+damage\s+to\s+(?:each\s+opponent|any\s+target|target)', oracle)
                if m_dmg:
                    w = m_dmg.group(1)
                    nv = {"x":-1}.get(w, int(w) if w.isdigit() else 0)
                    if nv != 0:
                        effects["damage"] = nv
                # Life loss: "each opponent loses X/N life"
                m_loss = re.search(r'each\s+opponent\s+loses?\s+(\w+)\s+life', oracle)
                if m_loss:
                    w = m_loss.group(1)
                    nv = {"x":-1}.get(w, int(w) if w.isdigit() else 0)
                    if nv != 0:
                        effects["damage"] = effects.get("damage", 0) + nv
                # Lifegain: "gain X/N life"
                m_gain = re.search(r'(?:you\s+)?gain\s+(\w+)\s+life', oracle)
                if m_gain and "opponents" not in oracle.split("gain")[0][-20:]:
                    w = m_gain.group(1)
                    nv = {"x":-1}.get(w, int(w) if w.isdigit() else 0)
                    if nv != 0:
                        effects["lifegain"] = nv
                # +1/+1 counters: "put X +1/+1 counters on"
                m_ctr = re.search(r'put\s+(\w+)\s+\+1/\+1\s+counter', oracle)
                if m_ctr:
                    w = m_ctr.group(1)
                    nv = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"x":-1}.get(w, int(w) if w.isdigit() else 1)
                    effects["counter"] = nv
                if effects:
                    etb_value_cache[c.name] = effects
                continue

            if "creature" not in tl and "artifact" not in tl:
                continue
            effects = {}
            # "when [name] enters the battlefield" — look for all effects
            etb_match = re.search(r'when[^.]*?enters\s+the\s+battlefield[^.]*?(?:,|\.)', oracle)
            if not etb_match:
                etb_match = re.search(r'whenever[^.]*?enters[^.]*?(?:,|\.)', oracle)
            if etb_match:
                etb_text = oracle[etb_match.start():]
                # Tokens
                if "create" in etb_text:
                    m = re.search(r'create\s+(\w+)\s+', etb_text)
                    if m:
                        w = m.group(1)
                        n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                                 "six":6,"seven":7,"eight":8,"x":-1}.get(w, int(w) if w.isdigit() else 1)
                        effects["tokens"] = n_tok
                        etb_token_cache[c.name] = n_tok
                # Draw
                m_draw = re.search(r'draw\s+(\w+)\s+cards?', etb_text)
                if m_draw:
                    w = m_draw.group(1)
                    effects["draw"] = {"a":1,"one":1,"two":2,"three":3}.get(w, int(w) if w.isdigit() else 1)
                elif "draw a card" in etb_text:
                    effects["draw"] = 1
                # Damage
                m_dmg = re.search(r'deals?\s+(\d+)\s+damage\s+to\s+(?:each\s+opponent|any\s+target|target)', etb_text)
                if m_dmg:
                    effects["damage"] = int(m_dmg.group(1))
                m_loss = re.search(r'each\s+opponent\s+loses?\s+(\d+)\s+life', etb_text)
                if m_loss:
                    effects["damage"] = effects.get("damage", 0) + int(m_loss.group(1))
                # Ramp (search for land)
                if re.search(r'search\s+your\s+library\s+for\s+(?:a|an?)\s+(?:basic\s+)?land', etb_text):
                    effects["ramp"] = 1
                # Lifegain
                m_gain = re.search(r'(?:you\s+)?gain\s+(\d+)\s+life', etb_text)
                if m_gain:
                    effects["lifegain"] = int(m_gain.group(1))
                # Counters
                m_ctr = re.search(r'put\s+(\w+)\s+\+1/\+1\s+counter', etb_text)
                if m_ctr:
                    w = m_ctr.group(1)
                    effects["counter"] = {"a":1,"one":1,"two":2,"three":3}.get(w, int(w) if w.isdigit() else 1)
                # Destroy/exile (removal on ETB for goldfish = less relevant but track it)
                if re.search(r'(?:destroy|exile)\s+(?:target|up\s+to)', etb_text):
                    effects["removal"] = 1

            if effects:
                etb_value_cache[c.name] = effects

        # ═══════════════════════════════════════════════════════════════
        # TRIGGER CHAIN SYSTEM
        # Detects card interactions that create cascading effects.
        # Events: cast, creature_etb, artifact_etb, enchantment_etb,
        #   permanent_etb, token_created, landfall, death, draw_trigger,
        #   lifegain, attack, end_step, upkeep, leave
        # Actions: create_token, draw, damage, lifegain, blink, ramp,
        #   counter, mill, sacrifice, tutor
        # ═══════════════════════════════════════════════════════════════

        # trigger_cache: card_name -> list of (event, condition, action, param)
        trigger_cache = {}
        # activated_blink_cache: card_name -> mana_cost (int)
        activated_blink_cache = {}
        blink_cards = set()
        reanimate_engine_cards = set()

        NUMBER_MAP = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                      "six":6,"seven":7,"eight":8,"nine":9,"ten":10,"x":-1}

        def _parse_trigger_count(text, after_keyword="create"):
            """Extract a number from text like 'create two 1/1' → 2."""
            m = re.search(rf'{after_keyword}\s+(\w+)\s+', text)
            if m:
                w = m.group(1)
                if w in NUMBER_MAP: return NUMBER_MAP[w]
                if w.isdigit(): return int(w)
            return 1

        def _detect_actions(oracle, triggers_list, event, condition):
            """Scan oracle text for all actions a trigger produces and append them."""
            added = False
            # --- CREATE TOKEN ---
            if re.search(r'create\s+', oracle):
                n = _parse_trigger_count(oracle, "create")
                triggers_list.append((event, condition, "create_token", n))
                added = True
            # --- DRAW ---
            if re.search(r'(?:you\s+)?(?:may\s+)?draw\s+(?:a|one|\d+|two|three)\s+cards?', oracle):
                n = 1
                m = re.search(r'draw\s+(\w+)\s+cards?', oracle)
                if m:
                    w = m.group(1)
                    n = NUMBER_MAP.get(w, int(w) if w.isdigit() else 1)
                elif "draw a card" in oracle:
                    n = 1
                triggers_list.append((event, condition, "draw", n))
                added = True
            # --- DAMAGE (to opponents/each opponent/any target) ---
            m_dmg = re.search(r'deals?\s+(\d+)\s+damage\s+to\s+(?:each\s+opponent|any\s+target|target\s+(?:player|opponent))', oracle)
            if m_dmg:
                triggers_list.append((event, condition, "damage", int(m_dmg.group(1))))
                added = True
            # --- LIFE LOSS (each opponent loses N life) ---
            m_loss = re.search(r'each\s+opponent\s+loses?\s+(\d+)\s+life', oracle)
            if m_loss:
                triggers_list.append((event, condition, "damage", int(m_loss.group(1))))
                added = True
            # --- LIFE LOSS (target player loses N life) — in goldfish, target = opponent ---
            if not m_loss:
                m_tpl = re.search(r'target\s+player\s+loses?\s+(\d+)\s+life', oracle)
                if m_tpl:
                    triggers_list.append((event, condition, "damage", int(m_tpl.group(1))))
                    added = True
            # --- LIFEGAIN ---
            m_gain = re.search(r'(?:you\s+)?gain\s+(\d+)\s+life', oracle)
            if m_gain and event not in ("lifegain",):  # avoid self-referencing loops
                triggers_list.append((event, condition, "lifegain", int(m_gain.group(1))))
                added = True
            # --- BLINK ---
            if re.search(r'exile\s+(?:target|a|another)\s+(?:nonland\s+)?(?:creature|permanent)\s+(?:you\s+control\s+)?.*?(?:return|then)', oracle):
                triggers_list.append((event, condition, "blink", 1))
                added = True
            # --- +1/+1 COUNTERS ---
            m_ctr = re.search(r'put\s+(\w+)\s+\+1/\+1\s+counter', oracle)
            if m_ctr:
                n = NUMBER_MAP.get(m_ctr.group(1), int(m_ctr.group(1)) if m_ctr.group(1).isdigit() else 1)
                triggers_list.append((event, condition, "counter", n))
                added = True
            # --- MILL ---
            m_mill = re.search(r'(?:target\s+(?:player|opponent)\s+)?mills?\s+(\w+)\s+cards?', oracle)
            if m_mill and "opponent" not in oracle:
                n = NUMBER_MAP.get(m_mill.group(1), int(m_mill.group(1)) if m_mill.group(1).isdigit() else 2)
                triggers_list.append((event, condition, "mill", n))
                added = True
            # --- RAMP (search for land / add mana) ---
            if re.search(r'search\s+your\s+library\s+for\s+(?:a|an?)\s+(?:basic\s+)?land', oracle):
                triggers_list.append((event, condition, "ramp", 1))
                added = True
            # --- SACRIFICE (a creature/permanent) ---
            if re.search(r'sacrifice\s+(?:a|another|target)\s+(?:creature|permanent|artifact)', oracle):
                triggers_list.append((event, condition, "sacrifice", 1))
                added = True
            # --- TUTOR (search your library) ---
            if re.search(r'search\s+your\s+library\s+for\s+(?:a|an?)\s+(?!.*?land)', oracle):
                triggers_list.append((event, condition, "tutor", 1))
                added = True
            return added

        all_cards_for_triggers = list(cards) + (commanders or [])
        for c in all_cards_for_triggers:
            oracle = (c.oracle_text or "").lower()
            triggers = []

            # ═══════════════════════════════════════
            # CAST TRIGGERS
            # ═══════════════════════════════════════

            # "whenever you cast a/an [type] spell"
            cast_m = re.search(r'whenever\s+you\s+cast\s+(?:a|an)\s+(\w+)\s+spell', oracle)
            if cast_m:
                cast_type = cast_m.group(1)
                # Extract the action part (text after the trigger condition)
                action_text = oracle[cast_m.end():]
                _detect_actions(action_text, triggers, "cast", cast_type)

            # "whenever you cast an instant or sorcery spell" (multi-type)
            cast_multi = re.search(r'whenever\s+you\s+cast\s+an?\s+(\w+)\s+or\s+(\w+)\s+spell', oracle)
            if cast_multi and not cast_m:
                # Treat as "noncreature" if it's instant/sorcery, else "any"
                types = {cast_multi.group(1), cast_multi.group(2)}
                if types == {"instant", "sorcery"}:
                    cond = "noncreature"
                else:
                    cond = "any"
                action_text = oracle[cast_multi.end():]
                _detect_actions(action_text, triggers, "cast", cond)

            # "whenever you cast a spell" (no type restriction)
            if not cast_m and not cast_multi and re.search(r'whenever\s+you\s+cast\s+a\s+spell', oracle):
                cast_start = re.search(r'whenever\s+you\s+cast\s+a\s+spell', oracle)
                action_text = oracle[cast_start.end():]
                _detect_actions(action_text, triggers, "cast", "any")

            # ═══════════════════════════════════════
            # CREATURE ETB TRIGGERS
            # ═══════════════════════════════════════

            # "whenever a/another creature enters the battlefield [under your control]"
            m_cetb = re.search(r'whenever\s+(?:a|another)\s+(?:nontoken\s+)?creature\s+(?:you\s+control\s+)?enters', oracle)
            if m_cetb:
                action_text = oracle[m_cetb.end():]
                _detect_actions(action_text, triggers, "creature_etb", "creature")

            # ═══════════════════════════════════════
            # ARTIFACT ETB TRIGGERS
            # ═══════════════════════════════════════

            # "whenever an artifact enters the battlefield [under your control]"
            m_aetb = re.search(r'whenever\s+(?:a|an)\s+artifact\s+(?:you\s+control\s+)?enters', oracle)
            if m_aetb:
                action_text = oracle[m_aetb.end():]
                _detect_actions(action_text, triggers, "artifact_etb", "artifact")

            # ═══════════════════════════════════════
            # ENCHANTMENT ETB (Constellation)
            # ═══════════════════════════════════════

            # "whenever [name] or another enchantment enters" / "whenever an enchantment enters"
            m_eetb = re.search(r'whenever\s+(?:a|an|another)\s+enchantment\s+(?:you\s+control\s+)?enters', oracle)
            if not m_eetb:
                m_eetb = re.search(r'whenever\s+\w+[^.]*?or\s+another\s+enchantment\s+enters', oracle)
            if m_eetb:
                action_text = oracle[m_eetb.end():]
                _detect_actions(action_text, triggers, "enchantment_etb", "enchantment")

            # ═══════════════════════════════════════
            # PERMANENT ETB (broad)
            # ═══════════════════════════════════════

            # "whenever a permanent enters" / "whenever a nonland permanent enters"
            m_petb = re.search(r'whenever\s+(?:a|another)\s+(?:nonland\s+)?permanent\s+(?:you\s+control\s+)?enters', oracle)
            if m_petb:
                action_text = oracle[m_petb.end():]
                _detect_actions(action_text, triggers, "permanent_etb", "any")

            # ═══════════════════════════════════════
            # TOKEN CREATION TRIGGERS
            # ═══════════════════════════════════════

            # "whenever you create a/one or more token(s)" / "whenever a token enters"
            m_tok = re.search(r'whenever\s+(?:you\s+create\s+(?:a|one\s+or\s+more)\s+tokens?|a\s+token\s+(?:enters|you\s+control\s+enters))', oracle)
            if m_tok:
                action_text = oracle[m_tok.end():]
                _detect_actions(action_text, triggers, "token_created", "token")

            # ═══════════════════════════════════════
            # LANDFALL TRIGGERS
            # ═══════════════════════════════════════

            # "whenever a land enters the battlefield [under your control]" / "landfall"
            m_land = re.search(r'(?:whenever\s+a\s+land\s+(?:you\s+control\s+)?enters|landfall)', oracle)
            if m_land:
                # Find the action text — may be after a dash or comma
                action_text = oracle[m_land.end():]
                _detect_actions(action_text, triggers, "landfall", "land")

            # ═══════════════════════════════════════
            # DEATH TRIGGERS
            # ═══════════════════════════════════════

            # "whenever a/another creature [you control] dies"
            m_death = re.search(r'whenever\s+(?:a|another)\s+(?:nontoken\s+)?(?:creature|artifact|permanent)\s+(?:you\s+control\s+)?dies', oracle)
            # Also: "whenever [name] or another creature dies" (Blood Artist pattern)
            if not m_death:
                m_death = re.search(r'whenever\s+\w+[^.]*?or\s+another\s+creature\s+dies', oracle)
            if m_death:
                action_text = oracle[m_death.end():]
                _detect_actions(action_text, triggers, "death", "creature")

            # ═══════════════════════════════════════
            # "WHENEVER YOU DRAW" TRIGGERS
            # ═══════════════════════════════════════

            # "whenever you draw a card" (Niv-Mizzet, Ominous Seas)
            m_draw = re.search(r'whenever\s+you\s+draw\s+a\s+card', oracle)
            if m_draw:
                action_text = oracle[m_draw.end():]
                _detect_actions(action_text, triggers, "draw_trigger", "any")

            # ═══════════════════════════════════════
            # "WHENEVER YOU GAIN LIFE" TRIGGERS
            # ═══════════════════════════════════════

            # "whenever you gain life" (Archangel of Thune, Heliod's Sun-Crowned)
            m_lifeg = re.search(r'whenever\s+you\s+gain\s+life', oracle)
            if m_lifeg:
                action_text = oracle[m_lifeg.end():]
                _detect_actions(action_text, triggers, "lifegain", "any")

            # ═══════════════════════════════════════
            # "WHENEVER ATTACKS" TRIGGERS
            # ═══════════════════════════════════════

            # "whenever [name/a creature] attacks" — non-token-creation attack triggers
            # (combat_token_cache already handles token creation on attack)
            m_atk = re.search(r'whenever\s+(?:\w+[^.]*?\s+)?attacks', oracle)
            if m_atk:
                action_text = oracle[m_atk.end():]
                # Only add if it creates tokens, draws, damages, etc.
                _detect_actions(action_text, triggers, "attack", "creature")

            # ═══════════════════════════════════════
            # "WHENEVER DEALS DAMAGE" TRIGGERS
            # ═══════════════════════════════════════

            # "whenever [this creature / a creature] deals [combat] damage [to a player]"
            m_ddmg = re.search(r'whenever\s+(?:\w+[^.]*?\s+)?deals\s+(?:combat\s+)?damage\s+(?:to\s+(?:a\s+)?(?:player|opponent))?', oracle)
            if m_ddmg:
                action_text = oracle[m_ddmg.end():]
                _detect_actions(action_text, triggers, "damage_dealt", "creature")

            # ═══════════════════════════════════════
            # LEAVE-THE-BATTLEFIELD TRIGGERS
            # ═══════════════════════════════════════

            # "whenever a creature/permanent leaves the battlefield"
            m_leave = re.search(r'whenever\s+(?:a|another)\s+(?:creature|permanent|nonland\s+permanent)\s+(?:you\s+control\s+)?leaves\s+the\s+battlefield', oracle)
            if m_leave:
                action_text = oracle[m_leave.end():]
                _detect_actions(action_text, triggers, "leave", "any")

            # ═══════════════════════════════════════
            # UPKEEP / END STEP TRIGGERS
            # ═══════════════════════════════════════

            # "at the beginning of your upkeep" (non-draw, non-drain — those are handled elsewhere)
            # Only add if it creates tokens, counters, etc. that aren't already in other caches
            m_upk = re.search(r'at\s+the\s+beginning\s+of\s+(?:your|each)\s+(?:upkeep|end\s+step)', oracle)
            if m_upk:
                action_text = oracle[m_upk.end():]
                temp_triggers = []
                _detect_actions(action_text, temp_triggers, "upkeep", "any")
                for t in temp_triggers:
                    # Avoid duplicating what drain_cache or draw_cache already handle
                    if t[2] not in ("damage",):  # damage is in drain_cache
                        triggers.append(t)

            # ═══════════════════════════════════════
            # ACTIVATED BLINK
            # ═══════════════════════════════════════

            # "{N}{C}: exile [creature/permanent]...return"
            blink_cost_m = re.search(r'\{(\d+)\}(?:\{([wubrgc])\})?[^:]*?:\s*(?:exile|you\s+may\s+exile)\s+(?:target|this|a|up\s+to|paired)\s+(?:nonland\s+)?(?:creature|permanent)', oracle)
            if not blink_cost_m:
                blink_cost_m = re.search(r'\{(\d+)\}\{([wubrgc])\}:\s*exile\s+(?:this|target|a)\s+creature.*?return', oracle)
            if blink_cost_m:
                cost = int(blink_cost_m.group(1))
                colored = 1 if blink_cost_m.group(2) else 0
                if not colored:
                    cost_region = oracle[:oracle.index(":")] if ":" in oracle else ""
                    colored = len(re.findall(r'\{[wubrgc]\}', cost_region))
                total_cost = cost + colored
                activated_blink_cache[c.name] = max(total_cost, 1)
                blink_cards.add(c.name)

            # Passive blink: "at the beginning of your end step, exile...return"
            if re.search(r'at\s+the\s+beginning\s+of\s+(?:your|each).*?(?:end\s+step|upkeep).*?exile.*?return', oracle):
                triggers.append(("end_step", "any", "blink", 1))
                blink_cards.add(c.name)
            elif re.search(r'exile\s+(?:target|a)\s+(?:creature|permanent)\s+you\s+control.*?return.*?(?:end\s+step|next\s+turn)', oracle):
                blink_cards.add(c.name)

            # Reanimate engine: repeating return from graveyard
            if re.search(r'(?:at\s+the\s+beginning|whenever).*?return\s+(?:a|target)\s+creature.*?from\s+(?:your\s+)?graveyard', oracle):
                reanimate_engine_cards.add(c.name)

            if triggers:
                trigger_cache[c.name] = triggers

        # Pre-compute combat/attack token triggers:
        # card_name -> (type, count)
        #   type: "damage_scaled" (tokens = power), "damage_fixed" (fixed N),
        #         "attack" (triggers on attack)
        combat_token_cache = {}
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            tl = (c.type_line or "").lower()
            if c.name in combat_token_cache:
                continue
            # "deals combat damage to a player, create that many" — power-scaled
            if re.search(r'deals\s+combat\s+damage\s+to\s+(?:a\s+)?(?:player|opponent)[^.]*?create\s+that\s+many', oracle):
                combat_token_cache[c.name] = ("damage_scaled", 0)
            # "deals combat damage to a player, create X tokens"
            elif re.search(r'deals\s+combat\s+damage\s+to\s+(?:a\s+)?(?:player|opponent)[^.]*?create\s+', oracle):
                m = re.search(r'create\s+(\w+)\s+', oracle)
                n_tok = 1
                if m:
                    w = m.group(1)
                    n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                             "six":6}.get(w, int(w) if w.isdigit() else 1)
                combat_token_cache[c.name] = ("damage_fixed", n_tok)
            # "whenever ~ attacks, create" — attack trigger
            elif re.search(r'whenever[^.]*?attacks[^.]*?create\s+', oracle):
                if "creature" in tl:  # only creatures attack
                    m = re.search(r'create\s+(\w+)\s+', oracle)
                    n_tok = 1
                    if m:
                        w = m.group(1)
                        n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                                 "six":6,"x":-1}.get(w, int(w) if w.isdigit() else 1)
                    combat_token_cache[c.name] = ("attack", n_tok)

        # Pre-compute anthem/buff effects: card_name -> buff_amount (to all creatures)
        # Only static buffs like "+1/+1 to creatures you control"
        anthem_cache = {}
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            if c.name in anthem_cache:
                continue
            # "+N/+N" or "+N/+0" to creatures you control
            m = re.search(r'(?:other\s+)?creatures\s+you\s+control\s+get\s+\+(\d+)/\+(\d+)', oracle)
            if not m:
                m = re.search(r'creatures\s+you\s+control\s+(?:have|get)\s+\+(\d+)/\+(\d+)', oracle)
            if m:
                anthem_cache[c.name] = int(m.group(1))  # power buff
            else:
                # Lord effects: "other [type] you control get +1/+1"
                m2 = re.search(r'other\s+\w+(?:\s+\w+)?\s+you\s+control\s+get\s+\+(\d+)/\+(\d+)', oracle)
                if m2:
                    anthem_cache[c.name] = int(m2.group(1))

        # Pre-compute +1/+1 counter sources: card_name -> (trigger, counters_per)
        # trigger: "etb_creature" (Cathars' Crusade), "upkeep" (self-grow),
        #          "cast" (Hardened Scales bonus), "entering" (Grumgully)
        counter_cache = {}
        counter_doubler_cards = set()  # Hardened Scales, Doubling Season counter half
        all_counter_cards = list(cards) + (commanders or [])
        for c in all_counter_cards:
            oracle = (c.oracle_text or "").lower()
            if c.name in counter_cache:
                continue

            # "whenever a creature enters...put a +1/+1 counter on it/each creature"
            # Cathars' Crusade, Grumgully, Good-Fortune Unicorn
            if re.search(r'whenever\s+(?:a|another)\s+(?:creature|nontoken\s+creature)[^.]*?enters[^.]*?\+1/\+1\s+counter', oracle):
                m = re.search(r'(\w+)\s+\+1/\+1\s+counter', oracle)
                n_counters = 1
                if m:
                    w = m.group(1)
                    n_counters = {"a": 1, "one": 1, "two": 2, "three": 3}.get(w, int(w) if w.isdigit() else 1)
                counter_cache[c.name] = ("etb_creature", n_counters)
                continue

            # "at the beginning of your upkeep/end step, put a +1/+1 counter on ~"
            if re.search(r'at\s+the\s+beginning\s+of\s+(?:your|each)[^.]*?put\s+(?:a|one|\d+)\s+\+1/\+1\s+counter\s+on', oracle):
                m = re.search(r'put\s+(\w+)\s+\+1/\+1\s+counter', oracle)
                n_counters = 1
                if m:
                    w = m.group(1)
                    n_counters = {"a": 1, "one": 1, "two": 2, "three": 3}.get(w, int(w) if w.isdigit() else 1)
                counter_cache[c.name] = ("upkeep", n_counters)
                continue

            # "whenever you cast a spell, put a +1/+1 counter on ~" (Managorger Hydra)
            if re.search(r'whenever\s+(?:you\s+cast|a\s+player\s+casts)\s+[^.]*?\+1/\+1\s+counter\s+on', oracle):
                counter_cache[c.name] = ("cast", 1)
                continue

            # "each creature you control enters with an additional +1/+1 counter"
            if re.search(r'(?:enters|enter)\s+(?:the\s+battlefield\s+)?with\s+(?:an?\s+)?(?:additional\s+)?(?:\w+\s+)?\+1/\+1\s+counter', oracle):
                # Could be self ("enters with") or global ("creatures enter with")
                if re.search(r'creatures?\s+(?:you\s+control\s+)?(?:enters?|enter)\s+', oracle):
                    m = re.search(r'(\w+)\s+(?:additional\s+)?\+1/\+1\s+counter', oracle)
                    n_counters = 1
                    if m:
                        w = m.group(1)
                        n_counters = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}.get(w, int(w) if w.isdigit() else 1)
                    counter_cache[c.name] = ("etb_creature", n_counters)
                    continue

            # Counter doublers: Hardened Scales, Doubling Season (counter half), Vorinclex
            # "that many plus one" / "twice that many counters" / "twice that many are put on"
            if re.search(r'(?:that\s+many\s+plus\s+one|one\s+more)\s+(?:\+1/\+1\s+)?counter', oracle):
                counter_doubler_cards.add(c.name)
            elif re.search(r'(?:twice|double)\s+(?:that\s+)?many', oracle) and "counter" in oracle:
                counter_doubler_cards.add(c.name)

        # Pre-compute token doublers: card_name -> multiplier
        # Parallel Lives, Doubling Season, Anointed Procession, Chatterfang (partial),
        # Mondrak, Primal Vigor, etc.
        token_doubler_cards = set()
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            # "twice that many" / "double the number" / "an additional" token patterns
            if re.search(r'(?:twice\s+that\s+many|double\s+the\s+number|twice\s+as\s+many)\s+(?:of\s+)?(?:those\s+)?tokens', oracle):
                token_doubler_cards.add(c.name)
            elif re.search(r'if\s+(?:one\s+or\s+more|a)\s+tokens?\s+would\s+be\s+created.*?(?:twice|double|additional)', oracle):
                token_doubler_cards.add(c.name)
            elif re.search(r'create\s+(?:one|an)\s+additional\s+(?:copy|token)', oracle):
                token_doubler_cards.add(c.name)
            # "if you would create one or more tokens, create that many plus" 
            elif re.search(r'tokens?\s+(?:are\s+)?created\s+(?:under\s+your\s+control\s+)?instead', oracle):
                token_doubler_cards.add(c.name)
        # Also check commanders for token doubling
        if commanders:
            for c in commanders:
                oracle = (c.oracle_text or "").lower()
                if re.search(r'(?:twice\s+that\s+many|double|additional)\s+(?:of\s+)?(?:those\s+)?tokens?', oracle):
                    token_doubler_cards.add(c.name)
                elif re.search(r'create\s+(?:one|an)\s+additional', oracle):
                    token_doubler_cards.add(c.name)

        # Pre-compute token type replacers: card_name -> multiplier
        # Academy Manufactor: "if you would create a Clue, Food, or Treasure token,
        #   instead create one of each" → every qualifying token becomes 3
        # Also: cards that create extra tokens of different types alongside
        token_type_replacers = {}
        all_replacer_cards = list(cards) + (commanders or [])
        for c in all_replacer_cards:
            oracle = (c.oracle_text or "").lower()
            # "instead create one of each" — triplicates Clue/Food/Treasure
            if re.search(r'instead\s+create\s+one\s+of\s+each', oracle):
                token_type_replacers[c.name] = 3
            # "whenever you create a token, create [additional type] token"
            # e.g. Chatterfang: "whenever you create one or more tokens, also create that many Squirrel tokens"
            elif re.search(r'whenever\s+you\s+create\s+(?:one\s+or\s+more|a)\s+tokens?.*?(?:also\s+)?create\s+(?:that\s+many|an?\s+)', oracle):
                token_type_replacers[c.name] = 2  # doubles output
            # "whenever you create a treasure/clue/food, create an additional [type]"
            elif re.search(r'whenever\s+you\s+create\s+(?:a|one\s+or\s+more)\s+(?:treasure|clue|food|blood)[^.]*?create\s+(?:a|an|that\s+many)', oracle):
                token_type_replacers[c.name] = 2

        # ---- BOARD MANA ENABLERS ----
        # Cards that let you tap permanents for mana (not in Ramp category)
        # Types:
        #   "tap_type" — "tap an untapped [type] you control: add mana" (Urza)
        #   "count_type" — "add {M} for each [type] you control" (Priest of Titania)
        #   "grant_tap" — "creatures you control have '{T}: Add {M}'" (Cryptolith Rite)
        battlefield_types = _build_battlefield_types(list(cards) + (commanders or []))
        board_mana_cache = {}  # card_name -> (type, permanent_type_tapped)
        all_cards_plus_cmdr = list(cards) + (commanders or [])
        for c in all_cards_plus_cmdr:
            oracle = (c.oracle_text or "").lower()
            if c.name in board_mana_cache:
                continue

            # "tap an untapped [type] you control: add {M}" (Urza, Heritage Druid)
            m = re.search(r'tap\s+(?:an?\s+)?untapped\s+(\w+)\s+you\s+control[^.]*?add\s+', oracle)
            if m:
                ptype = m.group(1)  # "artifact", "creature", etc.
                board_mana_cache[c.name] = ("tap_type", ptype)
                continue

            # "add {G} for each [type] you control" (Priest of Titania, Gaea's Cradle)
            m = re.search(r'add\s+(?:\{[^}]+\}\s*)?(?:for\s+each|equal\s+to\s+the\s+number\s+of)\s+(\w+)', oracle)
            if m:
                raw = m.group(1)
                # Singularize: handle irregular plurals then regular
                if raw.endswith("ves") and raw not in ("reserves",):
                    # elves→elf, wolves→wolf, dwarves→dwarf
                    ptype = raw[:-3] + "f"
                elif raw.endswith("s") and raw != "plains":
                    ptype = raw[:-1]
                else:
                    ptype = raw
                # Accept any battlefield-relevant type: permanent types, land subtypes, creature subtypes
                if ptype in battlefield_types:
                    board_mana_cache[c.name] = ("count_type", ptype)
                    continue

            # "creatures you control have/gain '{T}: Add {M}'" (Cryptolith Rite, Elven Chorus)
            if re.search(r'creatures?\s+you\s+control\s+(?:have|gain)\s+.*?(?:\{t\}|tap).*?add\s+', oracle):
                board_mana_cache[c.name] = ("grant_tap", "creature")
                continue

            # "you may tap [N] untapped [type]" (Heritage Druid variant)
            m = re.search(r'tap\s+(\w+)\s+untapped\s+(\w+)\s+you\s+control[^.]*?add\s+', oracle)
            if m:
                ptype = m.group(2)
                board_mana_cache[c.name] = ("tap_type", ptype)

        # ---- CLONE / COPY EFFECT DETECTION ----
        # Cards that enter as a copy of another permanent on the battlefield.
        # In goldfish, they copy the "best" creature (highest blink score or power).
        # clone_cards: set of card names that are clones
        clone_cards = set()
        all_clone_check = list(cards) + (commanders or [])
        for c in all_clone_check:
            oracle = (c.oracle_text or "").lower()
            tl = (c.type_line or "").lower()
            # "you may have ~ enter the battlefield as a copy of"
            if re.search(r'(?:you\s+may\s+have|enters?\s+(?:the\s+battlefield\s+)?as)\s+(?:a\s+)?copy\s+of', oracle):
                clone_cards.add(c.name)
            # "create a token that's a copy of" (Helm of the Host, Mirage Phalanx)
            elif re.search(r'create\s+(?:a|one)\s+token\s+(?:that.s|that\s+is)\s+a\s+copy\s+of', oracle):
                clone_cards.add(c.name)
            # "becomes a copy of" (Sakashima, Lazav)
            elif re.search(r'becomes?\s+a\s+copy\s+of', oracle):
                clone_cards.add(c.name)

        # ---- ADDITIONAL COMBAT DETECTION ----
        # Cards that grant extra combat phases, multiplying combat damage.
        # extra_combat_cache: card_name -> (type, count)
        #   type: "repeating" (each turn once on battlefield), "etb" (once when cast),
        #         "activated" (pay mana for extra combat), "attack" (on attack trigger)
        #   count: number of additional combats (usually 1)
        extra_combat_cache = {}
        all_combat_check = list(cards) + (commanders or [])
        for c in all_combat_check:
            oracle = (c.oracle_text or "").lower()
            # "additional combat phase" / "additional combat step"
            if not re.search(r'additional\s+combat\s+(?:phase|step)', oracle):
                continue
            # Activated: "{cost}: ... additional combat phase" (Aggravated Assault)
            if re.search(r'\{[^}]+\}[^:]*:\s*.*?additional\s+combat', oracle):
                # Parse mana cost for activation
                cost_m = re.search(r'\{(\d+)\}', oracle)
                act_cost = int(cost_m.group(1)) if cost_m else 5
                colored = len(re.findall(r'\{[wubrgc]\}', oracle.split(":")[0] if ":" in oracle else ""))
                extra_combat_cache[c.name] = ("activated", 1, act_cost + colored)
            # Attack trigger: "whenever ~ attacks, ... additional combat" (Aurelia, Port Razer)
            elif re.search(r'whenever\s+.*?attacks.*?additional\s+combat', oracle):
                extra_combat_cache[c.name] = ("attack", 1, 0)
            # Landfall / upkeep / beginning of phase trigger
            elif re.search(r'(?:whenever\s+a\s+land|landfall|at\s+the\s+beginning).*?additional\s+combat', oracle):
                extra_combat_cache[c.name] = ("repeating", 1, 0)
            # ETB: "when ~ enters, ... additional combat" (Combat Celebrant)
            elif re.search(r'when\s+.*?enters.*?additional\s+combat', oracle):
                extra_combat_cache[c.name] = ("etb", 1, 0)
            # Sorcery/instant: one-shot extra combat (Relentless Assault, Seize the Day)
            elif "instant" in tl or "sorcery" in tl:
                extra_combat_cache[c.name] = ("etb", 1, 0)
            # Default: treat as repeating if on a permanent
            else:
                extra_combat_cache[c.name] = ("repeating", 1, 0)

        # ---- X SPELL DETECTION ----
        # Cards with {X} in mana cost: X = remaining mana at cast time.
        # x_spell_base_cmc: card_name -> base_cmc (non-X portion of cost)
        # At cast time, X = (available_mana - base_cmc). X spells are cast LAST.
        x_spell_base_cmc = {}
        all_x_check = list(cards) + (commanders or [])
        for c in all_x_check:
            mc = (c.mana_cost or "").upper()
            if "{X}" in mc:
                # Base CMC = Scryfall's CMC (which is the non-X portion for most X spells)
                # For cards like Fireball ({X}{R}), Scryfall CMC = 1
                # For cards like Hydroid Krasis ({X}{G}{U}), Scryfall CMC = 2
                x_spell_base_cmc[c.name] = int(c.cmc)

        # ---- ALTERNATE WIN CONDITION DETECTION ----
        # Detects cards that can win the game through non-damage means.
        # alt_win_cache: card_name -> (win_type, threshold)
        #
        # COMPLETE CARD COVERAGE (33 "you win" cards + lose-the-game + infect):
        #
        # Simulatable (goldfish can track the condition per-turn):
        #   "empty_library"      — Lab Maniac, Jace Wielder, Thassa's Oracle
        #   "treasure_count"     — Revel in Riches (10+ treasures at upkeep)
        #   "life_total_high"    — Felidar Sovereign (40+), Test of Endurance (50+)
        #   "creature_count"     — Epic Struggle (20+), Halo Fountain (15 tapped)
        #   "artifact_count"     — Hellkite Tyrant (20+), Mechanized Production (8+ same name)
        #   "graveyard_creatures"— Mortal Combat (20+ creature cards in GY)
        #   "counter_self"       — Darksteel Reactor (20 charge), Helix Pinnacle (100 tower),
        #                          Azor's Elocutors (5 filibuster), Simic Ascendancy (20 growth),
        #                          Chance Encounter (10 luck), Luck Bobblehead (100 luck)
        #   "hand_size"          — Triskaidekaphile (exactly 13), Twenty-Toed Toad (20+)
        #   "power_threshold"    — Mayael's Aria (creature with power 20+)
        #   "demon_count"        — Liliana's Contract (4+ Demons different names)
        #   "second_cast"        — Approach of the Second Sun (cast twice)
        #   "poison"             — Triumph of the Hordes, Blightsteel Colossus (infect)
        #   "opponent_loses"     — Door to Nothingness (activated: target player loses)
        #
        # Partially simulatable (heuristic estimates):
        #   "type_count"         — Gallifrey Stands (13 Doctors), Dragon wins (5 colored Dragons)
        #   "gate_count"         — Maze's End (10 Gates with different names)
        #   "five_color"         — Coalition Victory (5 land types + 5 color creature),
        #                          Happily Ever After (5 colors + 6 types + life >= starting)
        #
        # Not simulatable (skip or generic heuristic):
        #   "life_total_exact"   — Near-Death Experience (exactly 1 life)
        #   "instant_win"        — Hedron Alignment (4 zones), Barren Glory (empty board+hand),
        #                          Ramses Assassin Lord, Battle of Wits (impossible in EDH),
        #                          Biovisionary (4 copies, singleton), Promising Stairs (8 Rooms)
        #
        alt_win_cache = {}
        all_altwin_check = list(cards) + (commanders or [])
        for c in all_altwin_check:
            oracle = (c.oracle_text or "").lower()
            tl = (c.type_line or "").lower()

            # ---- Infect / poison (doesn't say "win/lose the game") ----
            if "you win the game" not in oracle and "wins the game" not in oracle and "loses the game" not in oracle:
                if re.search(r'\binfect\b', oracle):
                    if "creature" in tl or re.search(r'creatures?\s+.*?(?:gain|have)\s+.*?\binfect\b', oracle):
                        alt_win_cache[c.name] = ("poison", 10)
                continue

            # ---- "Target/that player loses the game" (Door to Nothingness, Phage) ----
            if re.search(r'(?:target|that)\s+player\s+loses\s+the\s+game', oracle):
                # Check if it's an activated ability with a mana cost
                cost_m = re.search(r'\{([^}]+)\}.*?:\s*.*?(?:target|that)\s+player\s+loses', oracle)
                act_cost = 10  # default high cost
                if cost_m:
                    # Parse total mana cost from the activation
                    costs = re.findall(r'\{([^}]+)\}', oracle.split(":")[0] if ":" in oracle else "")
                    act_cost = sum(int(x) if x.isdigit() else 1 for x in costs)
                alt_win_cache[c.name] = ("opponent_loses", act_cost)
                continue

            # ======== "YOU WIN THE GAME" PATTERNS (ordered specific → generic) ========

            # ---- Library-empty wins (Lab Maniac, Jace, Thassa's Oracle) ----
            if re.search(r'library\s+has\s+no\s+cards', oracle):
                alt_win_cache[c.name] = ("empty_library", 0); continue
            if re.search(r'no\s+cards\s+in\s+(?:your\s+)?library', oracle):
                alt_win_cache[c.name] = ("empty_library", 0); continue
            if re.search(r'(?:number\s+of\s+cards\s+in\s+your\s+library|(?:equal\s+to|greater\s+than).*?cards\s+in\s+your\s+library.*?you\s+win)', oracle):
                alt_win_cache[c.name] = ("empty_library", 0); continue

            # ---- Treasure count (Revel in Riches: 10+ treasures) ----
            if re.search(r'(?:control|have)\s+(?:ten|10)\s+or\s+more\s+treasures?', oracle):
                alt_win_cache[c.name] = ("treasure_count", 10); continue

            # ---- Life total: high threshold ----
            m = re.search(r'(?:have|is)\s+(\d+)\s+or\s+more\s+life', oracle)
            if m:
                alt_win_cache[c.name] = ("life_total_high", int(m.group(1))); continue

            # ---- Life total: exactly 1 (Near-Death Experience) ----
            if re.search(r'(?:exactly\s+1\s+life|life\s+total\s+is\s+(?:exactly\s+)?1\b)', oracle):
                alt_win_cache[c.name] = ("life_total_exact", 1); continue

            # ---- Demon count (Liliana's Contract: 4+ Demons different names) ----
            if re.search(r'(?:four|4)\s+or\s+more\s+demons?\s+with\s+different\s+names', oracle):
                alt_win_cache[c.name] = ("demon_count", 4); continue

            # ---- Creature type count (Gallifrey Stands: 13 Doctors, Dragon win: 5 Dragons) ----
            m = re.search(r'(?:control|have)\s+(?:thirteen|13)\s+or\s+more\s+doctors?', oracle)
            if m:
                alt_win_cache[c.name] = ("type_count", 13); continue
            if re.search(r'(?:five|5)\s+dragons?\s+this\s+way.*?you\s+win', oracle):
                alt_win_cache[c.name] = ("type_count", 5); continue

            # ---- Creature count (Epic Struggle 20+, Halo Fountain 15 tapped) ----
            m = re.search(r'(?:control|have|untap|tap)\s+(?:four|4|fifteen|15|20|twenty)\s+(?:or\s+more\s+|(?:un)?tapped\s+)?creatures?', oracle)
            if m:
                w = re.search(r'(four|4|fifteen|15|20|twenty)', oracle)
                nv = {"four":4,"4":4,"fifteen":15,"15":15,"20":20,"twenty":20}.get(w.group(1), 20) if w else 20
                alt_win_cache[c.name] = ("creature_count", nv); continue

            # ---- Artifact count (Hellkite Tyrant 20+, Mechanized Production 8+) ----
            m = re.search(r'(?:control|have)\s+(?:eight|8|20|twenty)\s+or\s+more\s+artifacts?', oracle)
            if m:
                w = re.search(r'(eight|8|20|twenty)', oracle)
                nv = {"eight":8,"8":8,"20":20,"twenty":20}.get(w.group(1), 20) if w else 20
                alt_win_cache[c.name] = ("artifact_count", nv); continue

            # ---- Graveyard creatures (Mortal Combat: 20+ creature cards in GY) ----
            if re.search(r'(?:20|twenty)\s+or\s+more\s+creature\s+cards?\s+(?:are\s+)?in\s+your\s+graveyard', oracle):
                alt_win_cache[c.name] = ("graveyard_creatures", 20); continue

            # ---- Counter-based self wins (card accumulates its own counters) ----
            # Simic Ascendancy: "20 or more growth counters" (tracks +1/+1 placed)
            if re.search(r'(?:20|twenty)\s+or\s+more\s+growth\s+counter', oracle):
                alt_win_cache[c.name] = ("counter_self", 20); continue
            # Darksteel Reactor: "20 or more charge counters"
            if re.search(r'(?:20|twenty)\s+or\s+more\s+charge\s+counter', oracle):
                alt_win_cache[c.name] = ("counter_self", 20); continue
            # Helix Pinnacle: "100 or more tower counters"
            if re.search(r'(?:100|one\s+hundred)\s+or\s+more\s+tower\s+counter', oracle):
                alt_win_cache[c.name] = ("counter_self", 100); continue
            # Azor's Elocutors: "5 or more filibuster counters"
            if re.search(r'(?:five|5)\s+or\s+more\s+filibuster\s+counter', oracle):
                alt_win_cache[c.name] = ("counter_self", 5); continue
            # Chance Encounter: "10 or more luck counters"
            if re.search(r'(?:ten|10)\s+or\s+more\s+luck\s+counter', oracle):
                alt_win_cache[c.name] = ("counter_self", 10); continue
            # Luck Bobblehead: "100 or more luck counters"
            if re.search(r'(?:100|one\s+hundred)\s+or\s+more\s+luck\s+counter', oracle):
                alt_win_cache[c.name] = ("counter_self", 100); continue
            # Twenty-Toed Toad: "twenty or more counters on it"
            if re.search(r'(?:20|twenty)\s+or\s+more\s+counters?\s+on\s+(?:it|this)', oracle):
                alt_win_cache[c.name] = ("counter_self", 20); continue
            # Generic: "N or more [type] counters on" this permanent
            m = re.search(r'(\d+)\s+or\s+more\s+\w+\s+counters?\s+on\s+(?:it|this)', oracle)
            if m:
                alt_win_cache[c.name] = ("counter_self", int(m.group(1))); continue

            # ---- Hand size wins ----
            if re.search(r'exactly\s+thirteen\s+cards\s+in\s+your\s+hand', oracle):
                alt_win_cache[c.name] = ("hand_size", 13); continue
            if re.search(r'(?:20|twenty)\s+or\s+more\s+cards\s+in\s+(?:your\s+)?hand', oracle):
                alt_win_cache[c.name] = ("hand_size", 20); continue

            # ---- Power threshold (Mayael's Aria: creature with power 20+) ----
            if re.search(r'creature\s+with\s+power\s+(?:20|twenty)\s+or\s+(?:greater|more)', oracle):
                alt_win_cache[c.name] = ("power_threshold", 20); continue

            # ---- Approach of the Second Sun (cast twice) ----
            if re.search(r'(?:seventh\s+from\s+the\s+top|another\s+spell\s+named)', oracle):
                alt_win_cache[c.name] = ("second_cast", 2); continue

            # ---- Gate wins (Maze's End: 10 Gates with different names) ----
            if re.search(r'(?:ten|10)\s+or\s+more\s+gates?\s+(?:with\s+different|you\s+control)', oracle):
                alt_win_cache[c.name] = ("gate_count", 10); continue

            # ---- Five-color wins (Coalition Victory, Happily Ever After) ----
            if re.search(r'(?:land\s+of\s+each\s+basic\s+land\s+type|five\s+colors?\s+among)', oracle):
                alt_win_cache[c.name] = ("five_color", 5); continue

            # ---- 200+ library (Battle of Wits — impossible in 100-card Commander) ----
            if re.search(r'(?:200|two\s+hundred)\s+or\s+more\s+cards\s+in\s+your\s+library', oracle):
                alt_win_cache[c.name] = ("instant_win", 0); continue

            # ---- Generic fallback: any "you win" or "loses the game" we can't categorize ----
            alt_win_cache[c.name] = ("instant_win", 0)

        # ---- SMART GOLDFISH CASTING PRIORITIES ----
        # These model a competent Commander player's decision-making:
        #
        # CAST PROBABILITIES (simulates a goldfish with no opponents to interact with):
        #   Ramp:      100% early (turns 1-4), 60% late (turns 5+, diminishing returns)
        #   Draw:      100% always (card advantage is king)
        #   Tutor:     85% (sometimes you hold for the right moment)
        #   Board:     90% (creatures, enchantments, artifacts — your main plan)
        #   Removal:   30% (no opponents' permanents to target in goldfish; represents 
        #              incidental removal that doubles as other value, e.g. Ravenous Chupacabra)
        #   Protection: 25% (counterspells/hexproof pieces — minimal value in goldfish)
        #   Combo:     95% (always want these in play if you can)
        #
        # ADDITIONAL INTELLIGENCE:
        #   - Extra land drops detected from oracle text (Exploration, Azusa)
        #   - Conditional draw spells held until condition met (power-based like Return
        #     of the Wildspeaker wait for a big creature)
        #   - Ramp with summoning sickness (creatures) produces mana next turn
        #   - Graveyard zone tracked for recursion decks
        #
        CAST_PROBS = {
            "Ramp": (1.0, 0.6),     # (early_rate, late_rate) — split at turn 5
            "Draw": (1.0, 1.0),
            "Tutor": (0.85, 0.85),
            "Removal": (0.30, 0.30),
            "Protection": (0.25, 0.25),
            "Land": (1.0, 1.0),
            "Board": (0.90, 0.90),   # fallback for creatures/enchantments/artifacts
            "Other": (0.80, 0.80),
        }

        # ---- COMBO PIECE AWARENESS ----
        # Build a lookup: card_name_lower -> list of combo sets it belongs to
        # Each combo set is a frozenset of card names (lowered)
        combo_membership = {}  # card_name_lower -> [frozenset, ...]
        if combo_pieces:
            for combo_set in combo_pieces:
                fs = frozenset(n.lower() for n in combo_set)
                for piece in fs:
                    combo_membership.setdefault(piece, []).append(fs)
        has_combos = len(combo_membership) > 0

        # ---- COMBO ASSEMBLY TRACKING ----
        # Track per-combo: how many sims assembled by each turn, and first-assembly turn sum
        combo_labels = []  # list of " + ".join(card names) for display
        combo_sets_lower = []  # list of frozenset(name.lower()) for matching
        combo_assembled_by_turn = []  # list of {turn: count_assembled}
        combo_first_turn_sum = []  # sum of first-assembly turns (for averaging)
        combo_first_turn_count = []  # number of sims where combo was assembled at all

        if combo_pieces:
            for cs in combo_pieces:
                names = sorted(cs)
                combo_labels.append(" + ".join(n[:20] for n in names))
                combo_sets_lower.append(frozenset(n.lower() for n in cs))
                combo_assembled_by_turn.append({t: 0 for t in range(turns + 1)})
                combo_first_turn_sum.append(0)
                combo_first_turn_count.append(0)
        n_combos = len(combo_labels)

        # Alternate win condition tracking across all games
        alt_win_turn_sum = 0
        alt_win_turn_count = 0
        alt_win_types_seen = set()
        for cname, (wtype, _thresh) in alt_win_cache.items():
            alt_win_types_seen.add(wtype)

        # Detect extra land drop cards
        extra_land_cards = set()
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            if re.search(r'(?:you\s+may\s+play|play)\s+(?:an?\s+)?additional\s+land', oracle):
                extra_land_cards.add(c.name)
            elif re.search(r'you\s+may\s+play\s+(?:two|three)\s+additional\s+lands?', oracle):
                extra_land_cards.add(c.name)

        # Detect conditional draw spells (need board state to be good)
        conditional_draw_cards = {}  # card_name -> min_power needed on battlefield
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            # "draw cards equal to the greatest power" (Return of the Wildspeaker, Rishkar's Expertise)
            if re.search(r'draw\s+cards?\s+equal\s+to\s+(?:the\s+greatest|target|a)\s+(?:creature|permanent).*?power', oracle):
                conditional_draw_cards[c.name] = 4  # wait for a decent creature
            elif re.search(r'greatest\s+power\s+among\s+(?:creatures|permanents)\s+you\s+control', oracle):
                conditional_draw_cards[c.name] = 4
            # "draw X" where X depends on board
            elif re.search(r'draw\s+(?:a\s+)?card\s+for\s+each\s+(?:creature|artifact|enchantment|permanent)', oracle):
                conditional_draw_cards[c.name] = 3  # wait for 3+ permanents of type

        # ---- TUTOR RESOLUTION ----
        # Build tutor info so tutors actually search the library in goldfish
        tutor_cards_gf = {}
        for c in cards:
            oracle = (c.oracle_text or "").lower()
            if c.category == "Tutor" or re.search(r'search\s+your\s+library', oracle):
                subtype = "any"
                after_search = oracle.split("search your library")[1][:50] if "search your library" in oracle else ""
                if re.search(r'for\s+a\s+creature', after_search): subtype = "creature"
                elif re.search(r'for\s+an?\s+artifact', after_search): subtype = "artifact"
                elif re.search(r'for\s+an?\s+enchantment', after_search): subtype = "enchantment"
                elif re.search(r'for\s+(?:a\s+)?(?:basic\s+)?(?:land|forest|mountain|swamp|plains|island)', after_search): subtype = "land"
                elif re.search(r'for\s+an?\s+(?:instant|sorcery)', after_search): subtype = "instant_sorcery"
                dest = "hand"
                if "on top of your library" in oracle or "to the top" in oracle: dest = "top"
                elif "onto the battlefield" in oracle or "into play" in oracle: dest = "battlefield"
                elif "into your graveyard" in oracle: dest = "graveyard"
                tutor_cards_gf[c.name] = (subtype, dest)

        # Build type map for tutor matching
        card_type_map = {}
        for c in cards:
            tl = (c.type_line or "").lower()
            if "creature" in tl: card_type_map[c.name.lower()] = "creature"
            elif "instant" in tl or "sorcery" in tl: card_type_map[c.name.lower()] = "instant_sorcery"
            elif "artifact" in tl: card_type_map[c.name.lower()] = "artifact"
            elif "enchantment" in tl: card_type_map[c.name.lower()] = "enchantment"
            elif "land" in tl: card_type_map[c.name.lower()] = "land"
            else: card_type_map[c.name.lower()] = "other"

        # ---- COMMANDER TRACKING ----
        # Commanders are always accessible from command zone for combo checks
        # AND can be cast from command zone when mana is available
        cmdr_names_lower = set()
        cmdr_cmc = {}
        cmdr_cards = []  # Card objects for commanders
        if commanders:
            for c in commanders:
                cmdr_names_lower.add(c.name.lower())
                cmdr_cmc[c.name.lower()] = int(c.cmc)
                cmdr_cards.append(c)
                # Register commander in all caches so its abilities work on battlefield
                if c.name not in draw_cache:
                    draw_cache[c.name] = SimEngine._estimate_card_draws(c)
                if c.category == "Ramp" and c.name not in mana_cache:
                    mana_cache[c.name] = SimEngine._estimate_mana_produced(c)
                oracle = (c.oracle_text or "").lower()
                tl = (c.type_line or "").lower()
                if c.name not in token_cache and "create" in oracle:
                    n_tok = 1
                    m = re.search(r'create\s+(\w+)\s+', oracle)
                    if m:
                        w = m.group(1)
                        n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                                 "six":6,"x":-1}.get(w, int(w) if w.isdigit() else 1)
                    repeating = "whenever" in oracle or "at the beginning" in oracle
                    token_cache[c.name] = (n_tok, repeating)
                if "creature" in tl and c.name not in power_cache:
                    pw = 0
                    pm = re.search(r'(\d+)/(\d+)', oracle[-20:]) if oracle else None
                    if pm: pw = int(pm.group(1))
                    else: pw = max(1, int(c.cmc) - 1)
                    power_cache[c.name] = pw
                # ETB token cache
                if "creature" in tl and c.name not in etb_token_cache:
                    if re.search(r'when[^.]*?enters\s+the\s+battlefield[^.]*?create\s+', oracle):
                        m = re.search(r'create\s+(\w+)\s+', oracle)
                        if m:
                            w = m.group(1)
                            n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                                     "six":6,"x":-1}.get(w, int(w) if w.isdigit() else 1)
                            etb_token_cache[c.name] = n_tok
                # Combat token cache
                if c.name not in combat_token_cache:
                    if re.search(r'deals\s+combat\s+damage\s+to\s+(?:a\s+)?(?:player|opponent)[^.]*?create\s+that\s+many', oracle):
                        combat_token_cache[c.name] = ("damage_scaled", 0)
                    elif re.search(r'deals\s+combat\s+damage\s+to\s+(?:a\s+)?(?:player|opponent)[^.]*?create\s+', oracle):
                        m2 = re.search(r'create\s+(\w+)\s+', oracle)
                        n_tok = 1
                        if m2:
                            w = m2.group(1)
                            n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6}.get(w, int(w) if w.isdigit() else 1)
                        combat_token_cache[c.name] = ("damage_fixed", n_tok)
                    elif re.search(r'whenever[^.]*?attacks[^.]*?create\s+', oracle) and "creature" in tl:
                        m2 = re.search(r'create\s+(\w+)\s+', oracle)
                        n_tok = 1
                        if m2:
                            w = m2.group(1)
                            n_tok = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"x":-1}.get(w, int(w) if w.isdigit() else 1)
                        combat_token_cache[c.name] = ("attack", n_tok)
                # Drain cache
                if c.name not in drain_cache:
                    dmg = 0; dtrig = None
                    if re.search(r'whenever\s+(?:a|another)\s+creature\s+(?:you\s+control\s+)?dies', oracle):
                        m2 = re.search(r'loses?\s+(\d+)\s+life', oracle)
                        if m2: dmg = int(m2.group(1)); dtrig = "death"
                        elif re.search(r'deals?\s+(\d+)\s+damage', oracle):
                            m3 = re.search(r'deals?\s+(\d+)\s+damage', oracle)
                            if m3: dmg = int(m3.group(1)); dtrig = "death"
                    if not dtrig and re.search(r'whenever\s+(?:a|another)\s+(?:creature|nontoken)\s+(?:you\s+control\s+)?enters', oracle):
                        m2 = re.search(r'deals?\s+(\d+)\s+damage\s+to\s+each', oracle)
                        if m2: dmg = int(m2.group(1)); dtrig = "etb_creature"
                        else:
                            m3 = re.search(r'loses?\s+(\d+)\s+life', oracle)
                            if m3: dmg = int(m3.group(1)); dtrig = "etb_creature"
                    if dtrig:
                        drain_cache[c.name] = (dmg, dtrig)
                # Anthem cache
                if c.name not in anthem_cache:
                    m2 = re.search(r'(?:other\s+)?creatures\s+you\s+control\s+(?:get|have)\s+\+(\d+)/\+(\d+)', oracle)
                    if m2: anthem_cache[c.name] = int(m2.group(1))

        for i in range(n):
            sh = random.sample(deck, len(deck))
            hand = list(sh[:7]); lib = list(sh[7:])
            battlefield = []; graveyard = []
            turn_bonus = 0
            extra_land_sources = 0  # how many extra land drops from permanents in play
            sim_combo_found = [None] * n_combos  # first turn each combo assembled (None = never)
            sim_cumul_dmg = 0  # cumulative combat damage across turns
            sim_total_tokens = 0  # cumulative tokens on battlefield
            sim_total_counters = 0  # cumulative +1/+1 counters on creatures
            sim_alt_win_turn = None  # first turn an alt win condition is met (None = never)
            sim_life = 40  # player life total (for life-cost tracking and life-total wins)
            sim_cumul_lifegain = 0  # cumulative life gained across all turns
            sim_cmdr_cumul_dmg = {}  # commander_name -> cumulative combat damage dealt
            sim_approach_casts = {}  # card_name -> number of times cast (for Approach of the Second Sun)
            sim_self_counter_turns = {}  # card_name -> turns on battlefield (for Darksteel Reactor etc.)
            cmdr_on_battlefield = set()  # track which commanders are on bf
            cmdr_tax = {cn: 0 for cn in cmdr_cmc}  # commander tax (increases by 2 each cast)

            cs = Counter(c.category for c in hand)
            for cat in ALL_CATEGORIES: td[0][cat] += cs.get(cat, 0)
            ld[0] += min(cs.get("Land", 0), 1)
            total_cards[0] += len(hand)
            hand_size[0] += len(hand)

            for t in range(1, turns+1):
                draws_this_turn = 0  # track draws THIS turn
                # Draw for turn (Commander: all players draw on turn 1)
                if lib:
                    hand.append(lib.pop(0))
                    draws_this_turn += 1

                # Calculate available mana
                land_count = sum(1 for c in battlefield if c.category == "Land")
                ramp_mana = 0
                for rc in battlefield:
                    if rc.category == "Ramp":
                        tl = (rc.type_line or "").lower()
                        # Creatures have summoning sickness — need to be here since last turn
                        # We track this implicitly: only count if it survived a full turn cycle
                        ramp_mana += mana_cache.get(rc.name, 1)

                # ---- BOARD MANA FROM ENABLERS ----
                # Cards like Urza (tap artifacts for mana), Cryptolith Rite (creatures tap for mana)
                board_bonus_mana = 0
                for bc in battlefield:
                    if bc.name in board_mana_cache:
                        btype, ptype = board_mana_cache[bc.name]
                        if btype == "tap_type":
                            # Count tappable permanents of that type (excluding lands/self/ramp already counted)
                            tappable = 0
                            for pc in battlefield:
                                if pc == bc: continue  # don't count itself
                                ptl = (pc.type_line or "").lower()
                                if ptype in ptl and pc.category != "Land" and pc.category != "Ramp":
                                    tappable += 1
                            # Also count tokens of that type
                            if ptype in ("artifact", "creature"):
                                tappable += int(sim_total_tokens * 0.5)  # estimate half tokens are tappable
                            board_bonus_mana += tappable
                        elif btype == "count_type":
                            # "add mana for each [type]" — count all of that type
                            type_count = 0
                            for pc in battlefield:
                                ptl = (pc.type_line or "").lower()
                                if ptype in ptl:
                                    type_count += 1
                            if ptype in ("creature", "elf", "goblin"):
                                type_count += int(sim_total_tokens * 0.5)
                            board_bonus_mana += type_count
                        elif btype == "grant_tap":
                            # Every creature taps for 1 mana (excluding those already tapping for other reasons)
                            creature_ct = sum(1 for pc in battlefield
                                if "creature" in (pc.type_line or "").lower()
                                and pc.category != "Ramp" and pc != bc)
                            creature_ct += int(sim_total_tokens * 0.5)
                            board_bonus_mana += creature_ct

                mana = land_count + ramp_mana + board_bonus_mana
                spent = 0

                # Play land(s) — including extra land drops from Exploration/Azusa
                extra_land_sources = sum(1 for c in battlefield if c.name in extra_land_cards)
                max_lands = 1 + extra_land_sources
                lands_played_this_turn = 0
                for _ in range(max_lands):
                    land_in_hand = [c for c in hand if c.category == "Land"]
                    if land_in_hand:
                        chosen = land_in_hand[0]
                        hand.remove(chosen); battlefield.append(chosen)
                        mana += 1
                        lands_played_this_turn += 1
                    else:
                        break

                # ---- CAST COMMANDER FROM COMMAND ZONE ----
                # Commanders that aren't on battlefield can be cast if mana available
                cmdr_cast_this_turn = []
                for cmdr in cmdr_cards:
                    cn = cmdr.name.lower()
                    if cn not in cmdr_on_battlefield:
                        cmdr_cost = cmdr_cmc[cn] + cmdr_tax[cn]
                        if cmdr_cost <= mana - spent:
                            # Cast the commander
                            battlefield.append(cmdr)
                            cmdr_on_battlefield.add(cn)
                            spent += cmdr_cost
                            cmdr_cast_this_turn.append(cmdr)

                # ---- SMART CASTING ----
                # Roll cast decisions once per card this turn
                cards_to_cast = list(cmdr_cast_this_turn)  # include commanders just cast
                castable = [c for c in hand if c.category != "Land" and int(c.cmc) <= mana - spent]

                # Track what's accessible this turn (hand + battlefield + commanders) for combo checks
                accessible_names = set()
                if has_combos:
                    accessible_names = {c.name.lower() for c in hand} | {c.name.lower() for c in battlefield}
                    for cn, ccmc in cmdr_cmc.items():
                        if ccmc <= mana:
                            accessible_names.add(cn)

                # Decide what to cast with probability-based prioritization
                cast_decisions = []  # (priority, cmc, card)
                for card in castable:
                    cat = card.category
                    early = t <= 4

                    # Conditional draw: hold if condition not met
                    if card.name in conditional_draw_cards:
                        min_power = conditional_draw_cards[card.name]
                        max_pw = 0
                        bf_count = 0
                        for bc in battlefield:
                            btl = (bc.type_line or "").lower()
                            if "creature" in btl:
                                try: pw = int(bc.oracle_text.split("/")[0][-1]) if bc.oracle_text else 0
                                except: pw = 2
                                max_pw = max(max_pw, pw)
                            if bc.category != "Land":
                                bf_count += 1
                        if max_pw < min_power and bf_count < min_power:
                            continue  # hold this spell

                    # ---- COMBO PIECE LOGIC ----
                    # If this card is part of a known combo, decide based on readiness
                    card_lower = card.name.lower()
                    combo_override = None
                    if has_combos and card_lower in combo_membership:
                        best_readiness = 0.0
                        best_combo_size = 99
                        for combo_set in combo_membership[card_lower]:
                            n_total = len(combo_set)
                            # Count how many pieces of THIS combo are accessible
                            n_ready = sum(1 for p in combo_set if p in accessible_names)
                            readiness = n_ready / n_total if n_total > 0 else 0
                            if readiness > best_readiness or (readiness == best_readiness and n_total < best_combo_size):
                                best_readiness = readiness
                                best_combo_size = n_total

                        # Casting probability based on combo readiness:
                        #   100% of pieces in hand/bf → 100% (go off NOW)
                        #   >50% ready              → 70% (start deploying)
                        #   <50% ready              → check if independently useful
                        #   Only piece, nothing else → 35% (don't expose unless it does something)
                        if best_readiness >= 1.0:
                            combo_override = 1.0   # all pieces ready — slam it
                        elif best_readiness > 0.5:
                            combo_override = 0.70  # most pieces ready, start assembling
                        else:
                            # Check if this card is "independently useful"
                            # Creatures, ramp, draw are useful on their own
                            tl_check = (card.type_line or "").lower()
                            independently_useful = (cat in ("Ramp", "Draw", "Tutor") or
                                                   "creature" in tl_check)
                            if independently_useful:
                                combo_override = None  # fall through to normal category rate
                            else:
                                combo_override = 0.35  # risky to expose, low priority

                    # Determine final cast probability
                    if combo_override is not None:
                        prob = combo_override
                    else:
                        prob_pair = CAST_PROBS.get(cat, CAST_PROBS.get("Board", (0.80, 0.80)))
                        prob = prob_pair[0] if early else prob_pair[1]

                    if random.random() > prob:
                        continue  # chose not to cast this turn

                    # Priority: combo-ready pieces get highest priority
                    if combo_override is not None and combo_override >= 0.70:
                        pri = -1  # cast before everything else — going for the win
                    elif cat == "Ramp": pri = 0
                    elif cat == "Draw": pri = 1
                    elif cat == "Tutor": pri = 2
                    elif cat == "Removal": pri = 5
                    elif cat == "Protection": pri = 6
                    else: pri = 3

                    cast_decisions.append((pri, int(card.cmc), card))

                # Sort by priority then CMC (cast cheapest first within priority)
                # X spells go last (priority 10) so they can use remaining mana
                for i, (pri, cmc, card) in enumerate(cast_decisions):
                    if card.name in x_spell_base_cmc:
                        cast_decisions[i] = (10, cmc, card)  # push X spells to end
                cast_decisions.sort(key=lambda x: (x[0], x[1]))

                x_values_this_turn = {}  # card_name -> computed X value for this cast
                for _, _, card in cast_decisions:
                    if card.name in x_spell_base_cmc:
                        # X spell: use ALL remaining mana, X = remaining - base_cmc
                        base = x_spell_base_cmc[card.name]
                        remaining = mana - spent
                        if remaining < base + 1:  # need at least X=1 to be worth casting
                            continue
                        x_val = remaining - base
                        x_values_this_turn[card.name] = x_val
                        cards_to_cast.append(card)
                        spent += base + x_val  # spend full amount
                    else:
                        cost = int(card.cmc)
                        if cost > mana - spent:
                            continue
                        cards_to_cast.append(card)
                        spent += cost

                # ═══════════════════════════════════════════
                # TRIGGER CHAIN RESOLUTION
                # Process each card cast, fire triggers, resolve chains
                # Handles: cast, creature_etb, artifact_etb, enchantment_etb,
                #   permanent_etb, token_created, landfall, death, draw_trigger,
                #   lifegain, attack, end_step, upkeep, leave, damage_dealt
                # ═══════════════════════════════════════════
                tokens_created_this_turn = 0
                chain_draws = 0
                chain_damage = 0
                chain_lifegain = 0
                creatures_entered_this_turn = 0
                lands_entered_this_turn = 0
                creatures_died_this_turn = 0
                chain_counters = 0

                def get_etb_tokens(cname):
                    """Get number of tokens a card creates on ETB."""
                    v = etb_token_cache.get(cname, 0)
                    if v == -1:
                        v = x_values_this_turn.get(cname, 3)  # resolve X
                    if not v and cname in token_cache:
                        tn, tr = token_cache[cname]
                        if not tr:
                            v = tn if tn != -1 else x_values_this_turn.get(cname, 3)
                    return v

                def get_card_types(card_obj):
                    """Get set of type tags for a card."""
                    tl2 = (card_obj.type_line or "").lower()
                    types = set()
                    for w in ("creature", "artifact", "enchantment", "instant", "sorcery", "planeswalker", "land"):
                        if w in tl2: types.add(w)
                    if "creature" not in types:
                        types.add("noncreature")
                    return types

                def matches_trigger_condition(cond, card_types):
                    """Check if a trigger condition matches the card types."""
                    if cond == "any" or cond == "token": return True
                    if cond == "noncreature": return "creature" not in card_types
                    return cond in card_types

                def score_blink_target(bp):
                    """Score a permanent for blink targeting based on ETB value."""
                    ev = etb_value_cache.get(bp.name, {})
                    sc = 0
                    for k, v in ev.items():
                        if v == -1: v = 3  # estimate for scoring
                        if k == "tokens": sc += v * 3
                        elif k == "draw": sc += v * 4
                        elif k == "damage": sc += v * 2
                        elif k == "ramp": sc += v * 2
                        elif k == "lifegain": sc += v * 1
                        elif k == "counter": sc += v * 1
                    return sc

                def resolve_x(val, card_name):
                    """Resolve X-dependent values: -1 sentinel → computed X value."""
                    if val == -1:
                        return x_values_this_turn.get(card_name, 3)  # fallback 3 if not cast this turn
                    return val

                def fire_etb_effects(cname, card_types, trigger_q, source="card"):
                    """When a permanent enters the battlefield, check all battlefield
                    permanents for triggers that react to it."""
                    for bp in battlefield:
                        if bp.name not in trigger_cache:
                            continue
                        for ev, cond, act, param in trigger_cache[bp.name]:
                            # Creature ETB triggers
                            if ev == "creature_etb" and "creature" in card_types:
                                trigger_q.append((act, param, bp.name))
                            # Artifact ETB triggers
                            elif ev == "artifact_etb" and "artifact" in card_types:
                                trigger_q.append((act, param, bp.name))
                            # Enchantment ETB triggers (Constellation)
                            elif ev == "enchantment_etb" and "enchantment" in card_types:
                                trigger_q.append((act, param, bp.name))
                            # Permanent ETB triggers (broad)
                            elif ev == "permanent_etb":
                                trigger_q.append((act, param, bp.name))
                            # Token creation triggers
                            elif ev == "token_created" and source == "token":
                                trigger_q.append((act, param, bp.name))
                            # Leave-the-battlefield triggers (from blink)
                            elif ev == "leave" and source == "blink":
                                trigger_q.append((act, param, bp.name))

                def fire_draw_triggers(n_drawn, trigger_q):
                    """Fire 'whenever you draw a card' triggers for n_drawn cards."""
                    for bp in battlefield:
                        if bp.name not in trigger_cache:
                            continue
                        for ev, cond, act, param in trigger_cache[bp.name]:
                            if ev == "draw_trigger":
                                for _ in range(n_drawn):
                                    trigger_q.append((act, param, bp.name))

                def fire_lifegain_triggers(amount, trigger_q):
                    """Fire 'whenever you gain life' triggers."""
                    if amount <= 0:
                        return
                    for bp in battlefield:
                        if bp.name not in trigger_cache:
                            continue
                        for ev, cond, act, param in trigger_cache[bp.name]:
                            if ev == "lifegain":
                                trigger_q.append((act, param, bp.name))

                def fire_death_triggers(n_died, trigger_q):
                    """Fire 'whenever a creature dies' triggers."""
                    for bp in battlefield:
                        if bp.name not in trigger_cache:
                            continue
                        for ev, cond, act, param in trigger_cache[bp.name]:
                            if ev == "death":
                                for _ in range(n_died):
                                    trigger_q.append((act, param, bp.name))

                def fire_landfall_triggers(n_lands, trigger_q):
                    """Fire landfall triggers for n_lands entering."""
                    for bp in battlefield:
                        if bp.name not in trigger_cache:
                            continue
                        for ev, cond, act, param in trigger_cache[bp.name]:
                            if ev == "landfall":
                                for _ in range(n_lands):
                                    trigger_q.append((act, param, bp.name))

                def resolve_queue(trigger_q, depth=0):
                    """Resolve all triggers in the queue, handling chain reactions.
                    Uses nonlocal variables for token/draw/damage tracking."""
                    nonlocal tokens_created_this_turn, chain_draws, chain_damage
                    nonlocal chain_lifegain, creatures_entered_this_turn
                    nonlocal creatures_died_this_turn, chain_counters
                    nonlocal turn_bonus, draws_this_turn, lands_entered_this_turn

                    max_chain = 50  # safety cap for total iterations
                    iterations = 0

                    while trigger_q and iterations < max_chain:
                        iterations += 1
                        act, param, src = trigger_q.pop(0)

                        # Resolve X-dependent values: -1 sentinel → computed X
                        if param == -1:
                            param = resolve_x(param, src)

                        if act == "create_token":
                            tokens_created_this_turn += param
                            creatures_entered_this_turn += param
                            # Tokens entering trigger creature_etb, artifact_etb, token_created, permanent_etb
                            token_types = {"creature", "artifact"}  # assume creature artifact tokens (conservative)
                            for bp in battlefield:
                                if bp.name in trigger_cache:
                                    for ev, cond, act2, param2 in trigger_cache[bp.name]:
                                        if ev == "creature_etb":
                                            for _ in range(param):
                                                trigger_q.append((act2, param2, bp.name))
                                        elif ev == "artifact_etb":
                                            for _ in range(param):
                                                trigger_q.append((act2, param2, bp.name))
                                        elif ev == "token_created":
                                            for _ in range(param):
                                                trigger_q.append((act2, param2, bp.name))
                                        elif ev == "permanent_etb":
                                            for _ in range(param):
                                                trigger_q.append((act2, param2, bp.name))
                                        if len(trigger_q) > max_chain:
                                            break
                                if len(trigger_q) > max_chain:
                                    break
                            # Drain cache damage from creature/token ETB
                            for bp in battlefield:
                                if bp.name in drain_cache:
                                    ddmg, dtrig = drain_cache[bp.name]
                                    if dtrig in ("etb_creature", "token"):
                                        chain_damage += ddmg * param

                        elif act == "draw":
                            drawn_now = 0
                            for _ in range(param):
                                if lib:
                                    hand.append(lib.pop(0))
                                    chain_draws += 1
                                    turn_bonus += 1
                                    draws_this_turn += 1
                                    drawn_now += 1
                            # Fire "whenever you draw" triggers
                            if drawn_now > 0:
                                fire_draw_triggers(drawn_now, trigger_q)

                        elif act == "damage":
                            chain_damage += param
                            # "whenever you deal damage" triggers could fire here
                            # but in goldfish it's noncombat so skip damage_dealt triggers

                        elif act == "lifegain":
                            chain_lifegain += param
                            # Fire "whenever you gain life" triggers
                            fire_lifegain_triggers(param, trigger_q)

                        elif act == "blink":
                            # Find best blink target by scoring all ETB effects
                            best_target = None; best_score = 0
                            for bp in battlefield:
                                if bp.name == src:
                                    continue  # don't blink the source
                                sc = score_blink_target(bp)
                                if sc > best_score:
                                    best_score = sc; best_target = bp
                            if best_target and best_score > 0:
                                # Leave-the-battlefield triggers
                                fire_etb_effects(best_target.name, get_card_types(best_target),
                                                trigger_q, source="blink")
                                # Re-enter: fire own ETB effects
                                etb_effs = etb_value_cache.get(best_target.name, {})
                                et = etb_effs.get("tokens", 0)
                                if et != 0:
                                    trigger_q.append(("create_token", et, best_target.name))
                                elif get_etb_tokens(best_target.name) > 0:
                                    trigger_q.append(("create_token", get_etb_tokens(best_target.name), best_target.name))
                                ed = etb_effs.get("draw", 0)
                                if ed != 0:
                                    trigger_q.append(("draw", ed, best_target.name))
                                edm = etb_effs.get("damage", 0)
                                if edm != 0:
                                    trigger_q.append(("damage", edm, best_target.name))
                                elg = etb_effs.get("lifegain", 0)
                                if elg != 0:
                                    trigger_q.append(("lifegain", elg, best_target.name))
                                er = etb_effs.get("ramp", 0)
                                if er != 0:
                                    trigger_q.append(("ramp", er, best_target.name))
                                ec = etb_effs.get("counter", 0)
                                if ec != 0:
                                    trigger_q.append(("counter", ec, best_target.name))
                                # The re-entering creature triggers ETB reactions
                                creatures_entered_this_turn += 1
                                bt_types = get_card_types(best_target)
                                fire_etb_effects(best_target.name, bt_types, trigger_q, source="card")

                        elif act == "counter":
                            chain_counters += param

                        elif act == "ramp":
                            # In goldfish sim, ramp = +1 effective mana next turn
                            # We track it but actual mana calc is elsewhere
                            lands_entered_this_turn += param

                        elif act == "mill":
                            # Mill self: move cards from library to graveyard
                            for _ in range(param):
                                if lib:
                                    graveyard.append(lib.pop(0))

                        elif act == "sacrifice":
                            # In goldfish, sacrifice triggers death
                            creatures_died_this_turn += param
                            fire_death_triggers(param, trigger_q)

                        elif act == "tutor":
                            # Simplified: draw the best card from library
                            if lib:
                                hand.append(lib.pop(0))
                                turn_bonus += 1
                                draws_this_turn += 1

                # ---- Process each card cast through trigger chain ----
                for card in cards_to_cast:
                    is_cmdr = card in cmdr_cast_this_turn
                    if not is_cmdr:
                        hand.remove(card)
                    tl = (card.type_line or "").lower()
                    is_permanent = "instant" not in tl and "sorcery" not in tl
                    card_types = get_card_types(card)

                    if not is_cmdr:
                        if is_permanent:
                            battlefield.append(card)
                        else:
                            graveyard.append(card)

                    if "creature" in card_types:
                        creatures_entered_this_turn += 1

                    # ---- CLONE / COPY HANDLING ----
                    # If this card is a clone, it copies the best creature on battlefield
                    # and gets that creature's ETB effects + power + triggers
                    clone_target_name = None
                    if card.name in clone_cards and is_permanent:
                        best_clone = None; best_clone_sc = 0
                        for bp in battlefield:
                            if bp == card: continue
                            btl = (bp.type_line or "").lower()
                            if "creature" not in btl and "artifact" not in btl: continue
                            sc = score_blink_target(bp)
                            # Also consider power for combat clones
                            sc += power_cache.get(bp.name, 0) * 2
                            # Prefer creatures with triggers
                            if bp.name in trigger_cache: sc += 5
                            if sc > best_clone_sc:
                                best_clone_sc = sc; best_clone = bp
                        if best_clone:
                            clone_target_name = best_clone.name
                            # Clone gets the copied creature's power for combat
                            power_cache[card.name] = power_cache.get(best_clone.name, 1)
                            # Copy ETB effects into this card
                            if best_clone.name in etb_value_cache:
                                etb_value_cache[card.name] = etb_value_cache[best_clone.name].copy()
                            if best_clone.name in etb_token_cache:
                                etb_token_cache[card.name] = etb_token_cache[best_clone.name]
                            # Copy triggers (clone has same triggered abilities)
                            if best_clone.name in trigger_cache:
                                trigger_cache[card.name] = trigger_cache[best_clone.name]
                            # Copy drain effects
                            if best_clone.name in drain_cache:
                                drain_cache[card.name] = drain_cache[best_clone.name]

                    # Determine ETB source (clone uses copied card's effects)
                    etb_source = clone_target_name or card.name

                    # ---- FIRE CAST TRIGGERS ----
                    trigger_queue = []
                    for bp in battlefield:
                        if bp.name in trigger_cache:
                            for ev, cond, act, param in trigger_cache[bp.name]:
                                if ev == "cast" and matches_trigger_condition(cond, card_types):
                                    trigger_queue.append((act, param, bp.name))

                    # ---- FIRE ETB TRIGGERS (permanents) ----
                    if is_permanent:
                        # Card's own ETB effects (or copied creature's ETB for clones)
                        etb_effs = etb_value_cache.get(etb_source, {})
                        etb_tok = resolve_x(etb_effs.get("tokens", 0), etb_source) or get_etb_tokens(etb_source)
                        if etb_tok > 0:
                            trigger_queue.append(("create_token", etb_tok, etb_source))
                        ed = resolve_x(etb_effs.get("draw", 0), etb_source)
                        if ed > 0:
                            trigger_queue.append(("draw", ed, etb_source))
                        edm = resolve_x(etb_effs.get("damage", 0), etb_source)
                        if edm > 0:
                            trigger_queue.append(("damage", edm, etb_source))
                        elg = resolve_x(etb_effs.get("lifegain", 0), etb_source)
                        if elg > 0:
                            trigger_queue.append(("lifegain", elg, etb_source))
                        er = resolve_x(etb_effs.get("ramp", 0), etb_source)
                        if er > 0:
                            trigger_queue.append(("ramp", er, etb_source))
                        ec = resolve_x(etb_effs.get("counter", 0), etb_source)
                        if ec > 0:
                            trigger_queue.append(("counter", ec, etb_source))

                        # Other permanents react to this entering
                        fire_etb_effects(card.name, card_types, trigger_queue, source="card")

                    # ---- INSTANT/SORCERY ONE-SHOT EFFECTS ----
                    if not is_permanent:
                        etb_effs = etb_value_cache.get(card.name, {})
                        # Tokens
                        etk = resolve_x(etb_effs.get("tokens", 0), card.name)
                        if etk > 0:
                            trigger_queue.append(("create_token", etk, card.name))
                        # Damage
                        edm = resolve_x(etb_effs.get("damage", 0), card.name)
                        if edm > 0:
                            trigger_queue.append(("damage", edm, card.name))
                        # Lifegain
                        elg = resolve_x(etb_effs.get("lifegain", 0), card.name)
                        if elg > 0:
                            trigger_queue.append(("lifegain", elg, card.name))
                        # Counters
                        ec = resolve_x(etb_effs.get("counter", 0), card.name)
                        if ec > 0:
                            trigger_queue.append(("counter", ec, card.name))

                    # ---- RESOLVE ALL TRIGGERS (with chaining) ----
                    resolve_queue(trigger_queue)

                    # ---- STANDARD DRAW/TUTOR RESOLUTION ----
                    draws_n, repeating, _lbl, _lc = draw_cache.get(card.name, (0, False, "", 0))
                    # Resolve X-dependent draw values
                    if draws_n == -1:
                        draws_n = x_values_this_turn.get(card.name, 3)
                    elif draws_n == -2:
                        # Half X, rounded down (Hydroid Krasis)
                        draws_n = x_values_this_turn.get(card.name, 3) // 2
                    if draws_n > 0 and not repeating:
                        actual_draws = int(draws_n)
                        # Skip if already counted via trigger chain ETB
                        if card.name not in etb_token_cache:
                            for _ in range(actual_draws):
                                if lib:
                                    hand.append(lib.pop(0))
                                    turn_bonus += 1
                                    draws_this_turn += 1

                    # ---- TUTOR RESOLUTION ----
                    if card.name in tutor_cards_gf and lib:
                        tsub, tdest = tutor_cards_gf[card.name]
                        tutor_target = None
                        if has_combos:
                            hand_bf_names = {c.name.lower() for c in hand} | {c.name.lower() for c in battlefield}
                            for combo_set in (combo_pieces or []):
                                cs_lower = {n2.lower() for n2 in combo_set}
                                missing = cs_lower - hand_bf_names - cmdr_names_lower
                                for mp in missing:
                                    ptype = card_type_map.get(mp, "other")
                                    can_find = (tsub == "any" or tsub == ptype)
                                    if can_find:
                                        for lc in lib:
                                            if lc.name.lower() == mp:
                                                tutor_target = lc; break
                                    if tutor_target: break
                                if tutor_target: break
                        if tutor_target is None:
                            best_score = -1
                            for lc in lib:
                                if lc.category == "Land": continue
                                ptype = card_type_map.get(lc.name.lower(), "other")
                                can_find = (tsub == "any" or tsub == ptype)
                                if not can_find: continue
                                score = 0
                                dn, rep, _, _ = draw_cache.get(lc.name, (0, False, "", 0))
                                if rep and dn > 0: score = 100
                                elif dn > 0: score = 50
                                elif lc.category == "Ramp": score = 40
                                else: score = 10
                                if score > best_score:
                                    best_score = score; tutor_target = lc
                        if tutor_target:
                            lib.remove(tutor_target)
                            if tdest == "hand":
                                hand.append(tutor_target); turn_bonus += 1; draws_this_turn += 1
                            elif tdest == "top": lib.insert(0, tutor_target)
                            elif tdest == "battlefield": battlefield.append(tutor_target)
                            elif tdest == "graveyard": graveyard.append(tutor_target)

                # ---- END-OF-TURN PASSIVE BLINKS ----
                # Thassa, Conjurer's Closet etc.: blink once at end step
                for bp in battlefield:
                    if bp.name in trigger_cache:
                        for ev, cond, act, param in trigger_cache[bp.name]:
                            if ev == "end_step" and act == "blink":
                                best_target = None; best_sc = 0
                                for bt in battlefield:
                                    if bt == bp: continue
                                    sc = score_blink_target(bt)
                                    if sc > best_sc:
                                        best_sc = sc; best_target = bt
                                if best_target and best_sc > 0:
                                    # Queue all ETB effects and resolve
                                    blink_q = []
                                    etb_effs = etb_value_cache.get(best_target.name, {})
                                    for eff_key, q_act in [("tokens","create_token"),("draw","draw"),
                                        ("damage","damage"),("lifegain","lifegain"),("ramp","ramp"),("counter","counter")]:
                                        if etb_effs.get(eff_key, 0) > 0:
                                            blink_q.append((q_act, etb_effs[eff_key], best_target.name))
                                    if not etb_effs.get("tokens") and get_etb_tokens(best_target.name) > 0:
                                        blink_q.append(("create_token", get_etb_tokens(best_target.name), best_target.name))
                                    creatures_entered_this_turn += 1
                                    fire_etb_effects(best_target.name, get_card_types(best_target), blink_q, source="card")
                                    resolve_queue(blink_q)
                            # End-step non-blink triggers (create token, draw, etc.)
                            elif ev == "end_step" and act != "blink":
                                end_q = [(act, param, bp.name)]
                                resolve_queue(end_q)
                            # Upkeep triggers that create tokens/counters
                            elif ev == "upkeep" and bp not in cards_to_cast:
                                upk_q = [(act, param, bp.name)]
                                resolve_queue(upk_q)

                # ---- ACTIVATED BLINK LOOPS ----
                # Deadeye Navigator etc.: spend mana to blink repeatedly
                remaining_mana = mana - spent
                for bp in battlefield:
                    if bp.name in activated_blink_cache and bp not in cards_to_cast:
                        blink_cost = activated_blink_cache[bp.name]
                        # Find best ETB target using full scoring
                        best_target = None; best_sc = 0
                        for bt in battlefield:
                            if bt == bp: continue
                            sc = score_blink_target(bt)
                            if sc > best_sc:
                                best_sc = sc; best_target = bt
                        if best_target and best_sc > 0:
                            # Calculate mana loop
                            net_mana_per_blink = 0
                            etb_effs = etb_value_cache.get(best_target.name, {})
                            blink_tokens = etb_effs.get("tokens", 0) or get_etb_tokens(best_target.name)
                            if best_target.name in board_mana_cache:
                                net_mana_per_blink = blink_tokens
                            blinks = 0
                            available = remaining_mana
                            max_blinks = 50
                            while available >= blink_cost and blinks < max_blinks:
                                available -= blink_cost
                                available += net_mana_per_blink
                                blinks += 1
                                # Queue all ETB effects per blink
                                if blink_tokens > 0:
                                    tokens_created_this_turn += blink_tokens
                                if etb_effs.get("draw", 0) > 0:
                                    for _ in range(etb_effs["draw"]):
                                        if lib:
                                            hand.append(lib.pop(0))
                                            chain_draws += 1; turn_bonus += 1; draws_this_turn += 1
                                if etb_effs.get("damage", 0) > 0:
                                    chain_damage += etb_effs["damage"]
                                if etb_effs.get("lifegain", 0) > 0:
                                    chain_lifegain += etb_effs["lifegain"]
                                creatures_entered_this_turn += 1
                                # ETB reaction triggers from other permanents (simplified for loop perf)
                                for bp2 in battlefield:
                                    if bp2.name in drain_cache:
                                        ddmg, dtrig = drain_cache[bp2.name]
                                        if dtrig in ("etb_creature", "token"):
                                            chain_damage += ddmg
                            remaining_mana = max(0, available)

                # ---- FIRE LANDFALL TRIGGERS ----
                # Lands played this turn trigger landfall
                n_lands_played = sum(1 for c in cards_to_cast if "land" in (c.type_line or "").lower())
                if n_lands_played > 0:
                    lands_entered_this_turn += n_lands_played
                    landfall_q = []
                    fire_landfall_triggers(n_lands_played, landfall_q)
                    resolve_queue(landfall_q)

                # ---- REPEATING DRAW from permanents already on battlefield ----
                for perm in battlefield:
                    draws_n, repeating, _lbl, _lc = draw_cache.get(perm.name, (0, False, "", 0))
                    if repeating and draws_n > 0:
                        if perm not in cards_to_cast:
                            for _ in range(int(draws_n)):
                                if lib:
                                    hand.append(lib.pop(0))
                                    turn_bonus += 1
                                    draws_this_turn += 1

                # ---- REPEATING TOKEN GENERATORS (upkeep/trigger) ----
                for bc in battlefield:
                    if bc.name in token_cache:
                        tok_n, tok_rep = token_cache[bc.name]
                        if tok_rep and bc not in cards_to_cast:
                            # Resolve X sentinel: if this permanent's token count was X,
                            # use the X value from when it was cast (or estimate 3)
                            actual_tok = tok_n if tok_n != -1 else x_values_this_turn.get(bc.name, 3)
                            tokens_created_this_turn += actual_tok

                # ---- COMBAT & ATTACK TRIGGER TOKENS ----
                if t >= 3:
                    anthem_buff = sum(anthem_cache.get(bc.name, 0) for bc in battlefield)
                    for bc in battlefield:
                        if bc.name in combat_token_cache and bc not in cards_to_cast:
                            ctype, ccount = combat_token_cache[bc.name]
                            if ctype == "damage_scaled":
                                base_pw = power_cache.get(bc.name, 1)
                                # Counters distributed across creatures boost power
                                counter_bonus = sim_total_counters // max(n_creatures, 1) if n_creatures > 0 else 0
                                effective_pw = base_pw + anthem_buff + counter_bonus
                                tokens_created_this_turn += effective_pw
                            elif ctype == "damage_fixed":
                                actual_ct = ccount if ccount != -1 else x_values_this_turn.get(bc.name, 3)
                                tokens_created_this_turn += actual_ct
                            elif ctype == "attack":
                                actual_ct = ccount if ccount != -1 else x_values_this_turn.get(bc.name, 3)
                                tokens_created_this_turn += actual_ct

                # ---- TOKEN TYPE REPLACERS ----
                # Academy Manufactor: each Clue/Food/Treasure becomes all 3 (×3)
                # Chatterfang: also create Squirrel for each token (×2)
                # Applied before doublers (replacement happens first, then doubling)
                for bc in battlefield:
                    if bc.name in token_type_replacers:
                        mult = token_type_replacers[bc.name]
                        tokens_created_this_turn *= mult
                        break  # only apply strongest replacer (they don't stack well)

                # ---- TOKEN DOUBLERS ----
                n_doublers = sum(1 for bc in battlefield if bc.name in token_doubler_cards)
                if n_doublers > 0 and tokens_created_this_turn > 0:
                    tokens_created_this_turn *= (2 ** n_doublers)

                sim_total_tokens += tokens_created_this_turn
                n_tokens_this_turn = tokens_created_this_turn

                # ---- DRAIN / PING DAMAGE ----
                drain_dmg_this_turn = chain_damage  # start with damage from trigger chains
                for bc in battlefield:
                    if bc.name in drain_cache:
                        ddmg, dtrig = drain_cache[bc.name]
                        if dtrig == "death":
                            drain_dmg_this_turn += ddmg * 1 if t >= 3 else 0
                        elif dtrig == "etb_creature":
                            pass  # already counted in trigger chain
                        elif dtrig == "token":
                            pass  # already counted in trigger chain
                        elif dtrig == "upkeep":
                            if bc not in cards_to_cast:
                                drain_dmg_this_turn += ddmg

                # ---- RECORD STATS ----
                all_seen = hand + battlefield
                cs = Counter(c.category for c in all_seen)
                for cat in ALL_CATEGORIES: td[t][cat] += cs.get(cat, 0)
                land_ct = sum(1 for c in battlefield if c.category == "Land")
                ld[t] += min(land_ct, t + 1)
                bonus_draws[t] += turn_bonus
                total_cards[t] += len(all_seen)

                # Available mana (lands + ramp + board enablers)
                ramp_mana_total = sum(mana_cache.get(rc.name, 1) for rc in battlefield if rc.category == "Ramp")
                avail_mana[t] += land_ct + ramp_mana_total + board_bonus_mana
                hand_size[t] += len(hand)

                # Battlefield composition
                n_creatures = 0; n_artifacts = 0; n_enchantments = 0; n_nonland = 0
                total_power = 0
                anthem_buff = sum(anthem_cache.get(bc.name, 0) for bc in battlefield) if t >= 3 else 0

                # ---- +1/+1 COUNTER ACCUMULATION ----
                counters_added_this_turn = chain_counters  # from trigger chains
                n_counter_doublers = sum(1 for bc in battlefield if bc.name in counter_doubler_cards)
                for bc in battlefield:
                    if bc.name in counter_cache and bc not in cards_to_cast:
                        ctrig, cn_per = counter_cache[bc.name]
                        if ctrig == "etb_creature":
                            # Triggers for each creature that entered this turn
                            counters_added_this_turn += cn_per * creatures_entered_this_turn
                        elif ctrig == "upkeep":
                            # Self-growing: adds counters to itself each turn
                            counters_added_this_turn += cn_per
                        elif ctrig == "cast":
                            # Triggers per spell cast (~cards_to_cast count)
                            counters_added_this_turn += cn_per * max(1, len(cards_to_cast))
                # Counter doublers (Hardened Scales = +1 per counter event, Doubling Season = x2)
                if n_counter_doublers > 0 and counters_added_this_turn > 0:
                    # Conservative: each doubler adds ~50% more counters
                    counters_added_this_turn = int(counters_added_this_turn * (1.5 ** n_counter_doublers))
                sim_total_counters += counters_added_this_turn

                for bc in battlefield:
                    btl = (bc.type_line or "").lower()
                    if bc.category != "Land": n_nonland += 1
                    if "creature" in btl:
                        n_creatures += 1
                        total_power += power_cache.get(bc.name, 1)
                    if "artifact" in btl: n_artifacts += 1
                    if "enchantment" in btl: n_enchantments += 1
                if t >= 3:
                    total_power += anthem_buff * n_creatures
                    total_power += sim_total_tokens * (1 + anthem_buff)
                    # +1/+1 counters spread across creatures add to total power
                    total_power += sim_total_counters

                bf_creatures[t] += n_creatures
                bf_artifacts[t] += n_artifacts
                bf_enchantments[t] += n_enchantments
                bf_total[t] += n_nonland
                token_count[t] += sim_total_tokens  # cumulative tokens on board
                combat_power[t] += total_power

                # Cumulative damage: combat (from T3) + drain/ping
                combat_dmg = total_power if t >= 3 else 0

                # ---- ADDITIONAL COMBAT PHASES ----
                # Extra combats multiply combat damage (not drain/ping)
                if t >= 3 and combat_dmg > 0:
                    extra_combats = 0
                    for bc in battlefield:
                        if bc.name in extra_combat_cache:
                            ec_type, ec_count, ec_cost = extra_combat_cache[bc.name]
                            if ec_type == "repeating" and bc not in cards_to_cast:
                                extra_combats += ec_count
                            elif ec_type == "attack" and bc not in cards_to_cast:
                                # Attack triggers: get 1 extra combat per turn
                                # (real play: only 1 extra from Aurelia per turn cycle)
                                extra_combats += ec_count
                            elif ec_type == "activated" and bc not in cards_to_cast:
                                # Activated: spend remaining mana on extra combats
                                avail = mana - spent
                                while avail >= ec_cost and extra_combats < 5:
                                    avail -= ec_cost
                                    extra_combats += ec_count
                            elif ec_type == "etb":
                                if bc in cards_to_cast:
                                    extra_combats += ec_count
                    # Cap at 5 extra combats (safety)
                    extra_combats = min(extra_combats, 5)
                    if extra_combats > 0:
                        combat_dmg *= (1 + extra_combats)

                total_dmg_this_turn = combat_dmg + drain_dmg_this_turn
                sim_cumul_dmg += total_dmg_this_turn
                cumul_damage[t] += sim_cumul_dmg
                cards_drawn_this_turn[t] += draws_this_turn
                gy_size[t] += len(graveyard)

                # ---- PER-TURN COMBO CHECK ----
                # Tracks whether all pieces have been FOUND (hand + battlefield + command zone)
                # regardless of whether you have mana to cast them all — a player holding
                # all combo pieces knows they're about to win
                if n_combos > 0:
                    accessible = {c.name.lower() for c in hand} | {c.name.lower() for c in battlefield}
                    # Commanders are always accessible from command zone (no mana gate)
                    for cn in cmdr_names_lower:
                        accessible.add(cn)
                    for ci in range(n_combos):
                        if sim_combo_found[ci] is None and combo_sets_lower[ci].issubset(accessible):
                            sim_combo_found[ci] = t

                # ---- PER-TURN ALT-WIN TRACKER UPDATES ----
                bf_names = {bc.name for bc in battlefield}
                # Accumulate lifegain across turns
                sim_cumul_lifegain += chain_lifegain
                sim_life = 40 + sim_cumul_lifegain  # simplified: start at 40, add all gains (ignore life costs for now)

                # Track commander combat damage (21 from one commander = win)
                if t >= 3:
                    for cm in (commanders or []):
                        if cm.name in bf_names:
                            cpw = power_cache.get(cm.name, 0)
                            if cpw > 0:
                                cmdr_dmg = cpw + anthem_buff + (sim_total_counters // max(n_creatures, 1) if n_creatures > 0 else 0)
                                sim_cmdr_cumul_dmg[cm.name] = sim_cmdr_cumul_dmg.get(cm.name, 0) + cmdr_dmg

                # Track self-counter cards (Darksteel Reactor, Helix Pinnacle, Azor's Elocutors)
                for bc in battlefield:
                    if bc.name in alt_win_cache:
                        awtype, _ = alt_win_cache[bc.name]
                        if awtype == "counter_self" and bc not in cards_to_cast:
                            sim_self_counter_turns[bc.name] = sim_self_counter_turns.get(bc.name, 0) + 1

                # Track Approach of the Second Sun casts
                for card in cards_to_cast:
                    if card.name in alt_win_cache:
                        awtype, _ = alt_win_cache[card.name]
                        if awtype == "second_cast":
                            sim_approach_casts[card.name] = sim_approach_casts.get(card.name, 0) + 1

                # Calculate infect damage this turn (only from infect creatures)
                infect_dmg_this_turn = 0
                if t >= 3:
                    for bc in battlefield:
                        if bc.name in alt_win_cache:
                            awtype, _ = alt_win_cache[bc.name]
                            if awtype == "poison" and bc not in cards_to_cast:
                                ipw = power_cache.get(bc.name, 1)
                                ipw += anthem_buff + (sim_total_counters // max(n_creatures, 1) if n_creatures > 0 else 0)
                                infect_dmg_this_turn += ipw
                    # If "all creatures gain infect" sorcery was cast this turn, all power is infect
                    for card in cards_to_cast:
                        if card.name in alt_win_cache:
                            awtype, _ = alt_win_cache[card.name]
                            if awtype == "poison" and ("sorcery" in (card.type_line or "").lower() or "instant" in (card.type_line or "").lower()):
                                infect_dmg_this_turn = total_power  # all creatures swing with infect
                # Accumulate infect damage unconditionally
                if infect_dmg_this_turn > 0:
                    sim_cmdr_cumul_dmg["__infect__"] = sim_cmdr_cumul_dmg.get("__infect__", 0) + infect_dmg_this_turn

                # ---- PER-TURN ALTERNATE WIN CONDITION CHECK ----
                if sim_alt_win_turn is None and alt_win_cache:
                    hand_names = {hc.name for hc in hand}
                    all_accessible = bf_names | hand_names
                    for aw_name, (aw_type, aw_thresh) in alt_win_cache.items():
                        if sim_alt_win_turn is not None:
                            break
                        on_bf = aw_name in bf_names
                        in_hand = aw_name in hand_names

                        if aw_type == "empty_library":
                            # Lab Maniac / Jace / Thassa's Oracle: library empty + enabler accessible
                            if (on_bf or in_hand) and len(lib) == 0:
                                sim_alt_win_turn = t

                        elif aw_type == "treasure_count":
                            # Revel in Riches: 10+ treasures at upkeep
                            # Estimate treasure fraction: if deck has treasure-making cards,
                            # a portion of tokens are treasures. Conservative: all tokens count
                            # since in a Revel deck, most tokens ARE treasures.
                            if on_bf and sim_total_tokens >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "life_total_high":
                            # Felidar Sovereign (40+), Test of Endurance (50+): cumulative life
                            if on_bf and sim_life >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "life_total_exact":
                            # Near-Death Experience: exactly 1 life at upkeep
                            # Can't meaningfully goldfish this — skip
                            pass

                        elif aw_type == "creature_count":
                            # Epic Struggle (20+), Halo Fountain (15 tapped)
                            # Token creatures ARE creatures. n_creatures = real cards,
                            # sim_total_tokens ≈ creature tokens on board
                            total_cr = n_creatures + sim_total_tokens
                            if on_bf and total_cr >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "artifact_count":
                            # Hellkite Tyrant (20+), Mechanized Production (8+)
                            # n_artifacts = real artifact cards on bf
                            # In artifact-heavy decks, most tokens are artifact tokens (thopters, treasures)
                            total_art = n_artifacts + sim_total_tokens
                            if on_bf and total_art >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "graveyard_creatures":
                            # Mortal Combat: 20+ creature cards in graveyard
                            gy_creatures = sum(1 for gc in graveyard
                                             if "creature" in (gc.type_line or "").lower())
                            if on_bf and gy_creatures >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "counter_self":
                            # Self-counter cards accumulate their own counters each turn.
                            # Darksteel Reactor: 1 charge/turn → 20 turns
                            # Helix Pinnacle: spend mana → {X}: add X tower counters
                            # Azor's Elocutors: 1 filibuster/upkeep → 5 turns
                            # Simic Ascendancy: tracks +1/+1 counters placed on creatures
                            card_counters = 0
                            turns_on_bf = sim_self_counter_turns.get(aw_name, 0)
                            oracle_l = ""
                            for cc in list(cards) + (commanders or []):
                                if cc.name == aw_name:
                                    oracle_l = (cc.oracle_text or "").lower()
                                    break
                            if "growth" in oracle_l or "ascendancy" in oracle_l.replace(" ", ""):
                                # Simic Ascendancy: tracks +1/+1 counters placed
                                card_counters = sim_total_counters
                            elif aw_thresh >= 100:
                                # Helix Pinnacle: spend remaining mana each turn
                                card_counters = turns_on_bf * max(3, (mana - spent))
                            elif aw_thresh <= 5:
                                # Azor's Elocutors: 1 per upkeep
                                card_counters = turns_on_bf
                            else:
                                # Darksteel Reactor: 1 charge per upkeep
                                card_counters = turns_on_bf
                            if on_bf and card_counters >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "hand_size":
                            # Triskaidekaphile: exactly 13 cards in hand at upkeep
                            # Twenty-Toed Toad: 20+ cards in hand
                            if aw_thresh == 13:
                                if on_bf and len(hand) == 13:
                                    sim_alt_win_turn = t
                            else:
                                if on_bf and len(hand) >= aw_thresh:
                                    sim_alt_win_turn = t

                        elif aw_type == "power_threshold":
                            # Mayael's Aria: creature with power 20+
                            max_pw = 0
                            counter_per = sim_total_counters // max(n_creatures, 1) if n_creatures > 0 else 0
                            for bp in battlefield:
                                if "creature" in (bp.type_line or "").lower():
                                    bpw = power_cache.get(bp.name, 0) + anthem_buff + counter_per
                                    max_pw = max(max_pw, bpw)
                            # Token power: base 1 + anthem + counters
                            if sim_total_tokens > 0:
                                tok_pw = 1 + anthem_buff + counter_per
                                max_pw = max(max_pw, tok_pw)
                            if on_bf and max_pw >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "demon_count":
                            # Liliana's Contract: 4+ Demons with different names
                            demon_names = set()
                            for bp in battlefield:
                                if "demon" in (bp.type_line or "").lower():
                                    demon_names.add(bp.name)
                            if on_bf and len(demon_names) >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "poison":
                            # Infect: 10 poison on one opponent = lethal
                            # Only count cumulative damage dealt by infect creatures
                            sim_cumul_infect = sim_cmdr_cumul_dmg.get("__infect__", 0)
                            if sim_cumul_infect >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "second_cast":
                            # Approach of the Second Sun: cast from hand after previously casting it
                            # First cast: goes 7th from top of library (not graveyard)
                            # Second cast from hand: you win
                            cast_count = sim_approach_casts.get(aw_name, 0)
                            if cast_count >= 2:
                                sim_alt_win_turn = t

                        elif aw_type == "gate_count":
                            # Maze's End: 10 Gates with different names on battlefield
                            # Count lands with "Gate" in type or name
                            gate_names = set()
                            for bp in battlefield:
                                btl = (bp.type_line or "").lower()
                                bn = bp.name.lower()
                                if "gate" in btl or "gate" in bn:
                                    gate_names.add(bp.name)
                            # Also count lands dropped that are gates
                            if on_bf and len(gate_names) >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "five_color":
                            # Coalition Victory: land of each basic type + creature of each color
                            # Happily Ever After: 5 colors among permanents, 6+ card types, life >= 40
                            # Heuristic: if on bf and we have 5+ different-color permanents
                            # and late enough, likely achievable in a 5c deck
                            if on_bf and t >= 6:
                                # Count distinct colors among permanents' mana costs
                                colors_seen = set()
                                for bp in battlefield:
                                    mc = (bp.mana_cost or "").upper()
                                    for color in "WUBRG":
                                        if "{" + color + "}" in mc:
                                            colors_seen.add(color)
                                if len(colors_seen) >= 4:  # 4+ colors = likely 5c deck succeeds
                                    sim_alt_win_turn = t

                        elif aw_type == "type_count":
                            # Gallifrey Stands (13 Doctors), Dragon Shields (5 colored Dragons)
                            # In goldfish, these are very hard to achieve.
                            # Heuristic: if on bf with many creatures + tokens, check threshold
                            total_cr = n_creatures + sim_total_tokens
                            if on_bf and total_cr >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "opponent_loses":
                            # Door to Nothingness, etc.: activated ability to make opponent lose
                            # Check if we have enough mana to activate
                            if on_bf and (mana - spent) >= aw_thresh:
                                sim_alt_win_turn = t

                        elif aw_type == "instant_win":
                            # Generic "you win the game" — complex conditions we can't fully simulate.
                            # Heuristic: if on battlefield for 2+ turns in late game
                            if on_bf and t >= 7:
                                sim_alt_win_turn = t

                    # ---- COMMANDER DAMAGE WIN (21 from one commander) ----
                    if sim_alt_win_turn is None:
                        for cm_name, cm_dmg in sim_cmdr_cumul_dmg.items():
                            if cm_name.startswith("__"): continue  # skip infect tracker
                            if cm_dmg >= 21:
                                sim_alt_win_turn = t
                                break

                # ---- COMMANDER DAMAGE WIN (always active, even without alt-win cards) ----
                if sim_alt_win_turn is None and not alt_win_cache:
                    for cm_name, cm_dmg in sim_cmdr_cumul_dmg.items():
                        if cm_name.startswith("__"): continue
                        if cm_dmg >= 21:
                            sim_alt_win_turn = t
                            break

            # End of game: accumulate combo results
            if n_combos > 0:
                for ci in range(n_combos):
                    ft = sim_combo_found[ci]
                    if ft is not None:
                        combo_first_turn_sum[ci] += ft
                        combo_first_turn_count[ci] += 1
                        # Mark assembled for all turns from first-assembly onward
                        for tt in range(ft, turns + 1):
                            combo_assembled_by_turn[ci][tt] += 1

            # Accumulate alt-win results
            if sim_alt_win_turn is not None:
                alt_win_turn_sum += sim_alt_win_turn
                alt_win_turn_count += 1

            if pcb and i % 100 == 0: pcb(i / n)

        for t in td:
            for cat in td[t]: td[t][cat] /= n
            ld[t] /= n
            bonus_draws[t] /= n
            total_cards[t] /= n
            avail_mana[t] /= n
            hand_size[t] /= n
            bf_creatures[t] /= n
            bf_artifacts[t] /= n
            bf_enchantments[t] /= n
            bf_total[t] /= n
            token_count[t] /= n
            combat_power[t] /= n
            cumul_damage[t] /= n
            cards_drawn_this_turn[t] /= n
            gy_size[t] /= n

        # Sanitize: prevent -0.0 from appearing in results (replace with +0.0)
        for d in (ld, bonus_draws, total_cards, avail_mana, hand_size, bf_creatures,
                  bf_artifacts, bf_enchantments, bf_total, token_count, combat_power,
                  cumul_damage, cards_drawn_this_turn, gy_size):
            for t in d:
                if d[t] == 0: d[t] = 0.0  # convert int 0 or -0.0 to +0.0
        if pcb: pcb(1.0)

        # Estimate "win turn": first turn avg cumulative damage >= 40 (one opponent)
        win_turn_1opp = None
        win_turn_3opp = None
        for t in range(1, turns + 1):
            if win_turn_1opp is None and cumul_damage[t] >= 40:
                win_turn_1opp = t
            if win_turn_3opp is None and cumul_damage[t] >= 120:
                win_turn_3opp = t

        # Count draw sources in deck
        draw_sources = []
        for c in cards:
            dn, rep, lbl, _lc = draw_cache.get(c.name, (0, False, "", 0))
            if dn > 0:
                draw_sources.append({"name": c.name, "qty": c.quantity, "draws": dn,
                                      "repeating": rep, "label": lbl})

        # Build combo stats for output
        combo_stats = []
        if n_combos > 0:
            for ci in range(n_combos):
                found_count = combo_first_turn_count[ci]
                avg_turn = combo_first_turn_sum[ci] / found_count if found_count > 0 else None
                pct_by_turn = {t: combo_assembled_by_turn[ci][t] / n * 100 for t in range(turns + 1)}
                combo_stats.append({
                    "label": combo_labels[ci],
                    "pieces": len(combo_sets_lower[ci]),
                    "found_count": found_count,
                    "found_pct": found_count / n * 100,
                    "avg_turn": avg_turn,
                    "pct_by_turn": pct_by_turn,
                })

        # Alternate win condition stats
        alt_win_stats = None
        if alt_win_cache:
            avg_alt_turn = alt_win_turn_sum / alt_win_turn_count if alt_win_turn_count > 0 else None
            alt_win_stats = {
                "cards": list(alt_win_cache.keys()),
                "types": list(alt_win_types_seen),
                "win_count": alt_win_turn_count,
                "win_pct": alt_win_turn_count / n * 100,
                "avg_turn": avg_alt_turn,
            }

        return {"turn_avgs": td, "land_drops": ld, "bonus_draws": bonus_draws,
                "total_cards": total_cards, "draw_sources": draw_sources,
                "combo_stats": combo_stats,
                "avail_mana": avail_mana, "hand_size": hand_size,
                "bf_creatures": bf_creatures, "bf_artifacts": bf_artifacts,
                "bf_enchantments": bf_enchantments, "bf_total": bf_total,
                "token_count": token_count, "combat_power": combat_power,
                "cumul_damage": cumul_damage, "cards_drawn_this_turn": cards_drawn_this_turn,
                "gy_size": gy_size,
                "win_turn_1opp": win_turn_1opp, "win_turn_3opp": win_turn_3opp,
                "alt_win_stats": alt_win_stats}

# ============================================================================
# SAMPLE HAND POP-UP
# ============================================================================
class SampleHandWindow:
    def __init__(self, parent, cards, scryfall, pil_cache):
        self.cards = cards; self.scry = scryfall; self.pc = pil_cache; self.refs = []
        self.library = []; self.hand = []
        self.win = tk.Toplevel(parent); self.win.title("Sample Opening Hand")
        self.win.geometry("1200x550"); self.win.configure(bg="#1a1a2e")
        bf = tk.Frame(self.win, bg="#1a1a2e"); bf.pack(fill=tk.X, padx=10, pady=5)
        for txt, sz in [("New Hand (7)",7),("Mulligan (6)",6),("Mulligan (5)",5)]:
            tk.Button(bf, text=txt, font=("Segoe UI",10,"bold"),
                command=lambda s=sz: self._new_hand(s), bg="#0f3460", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(bf, text="Draw 1", font=("Segoe UI",10,"bold"),
            command=lambda: self._draw_cards(1), bg="#2E8B57", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(bf, text="Draw 2", font=("Segoe UI",10,"bold"),
            command=lambda: self._draw_cards(2), bg="#2E8B57", fg="white").pack(side=tk.LEFT, padx=5)
        self.info = tk.StringVar(value="")
        tk.Label(bf, textvariable=self.info, bg="#1a1a2e", fg="#e0e0e0",
                 font=("Segoe UI",10)).pack(side=tk.RIGHT, padx=10)
        self.lib_info = tk.StringVar(value="")
        tk.Label(bf, textvariable=self.lib_info, bg="#1a1a2e", fg="#999",
                 font=("Segoe UI",9)).pack(side=tk.RIGHT, padx=10)
        self.hf = tk.Frame(self.win, bg="#1a1a2e")
        self.hf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._new_hand()

    def _new_hand(self, sz=7):
        """Shuffle deck and draw a new opening hand."""
        deck = SimEngine.build_deck(self.cards)
        random.shuffle(deck)
        self.hand = deck[:sz]; self.library = deck[sz:]
        self._render()

    def _draw_cards(self, n=1):
        """Draw n cards from the top of the library."""
        for _ in range(n):
            if self.library:
                self.hand.append(self.library.pop(0))
        self._render()

    def _render(self):
        for w in self.hf.winfo_children(): w.destroy()
        self.refs.clear()
        cats = Counter(c.category for c in self.hand)
        l, r, d = cats.get("Land",0), cats.get("Ramp",0), cats.get("Draw",0)
        self.info.set(f"{len(self.hand)} cards  |  Lands: {l}  Ramp: {r}  Draw: {d}")
        self.lib_info.set(f"Library: {len(self.library)} cards")
        for card in self.hand:
            cf = tk.Frame(self.hf, bg="#1a1a2e"); cf.pack(side=tk.LEFT, padx=3, fill=tk.Y)
            if card.name in self.pc:
                hand_img = self.pc[card.name].resize((130, 181), Image.LANCZOS)
                photo = ImageTk.PhotoImage(hand_img)
                tk.Label(cf, image=photo, bg="#1a1a2e").pack()
                self.refs.append(photo)
            else:
                col = CATEGORY_COLORS.get(card.category, "#708090")
                ph = tk.Frame(cf, bg=col, width=130, height=181); ph.pack_propagate(False); ph.pack()
                tk.Label(ph, text=card.name, wraplength=120, bg=col, fg="white",
                         font=("Segoe UI",8,"bold")).pack(expand=True)
                threading.Thread(target=self._load, args=(card, cf, ph), daemon=True).start()
            cc = CATEGORY_COLORS.get(card.category, "#708090")
            tk.Label(cf, text=card.category, bg=cc, fg="white",
                     font=("Segoe UI",8,"bold"), padx=4, pady=1).pack(fill=tk.X)

    def _load(self, card, frame, placeholder):
        img = self.scry.fetch_image(card.image_uri, card.name)
        if img:
            self.pc[card.name] = img
            hand_img = img.resize((130, 181), Image.LANCZOS)
            photo = ImageTk.PhotoImage(hand_img)
            def update_ui():
                try:
                    if placeholder.winfo_exists():
                        placeholder.destroy()
                        lbl = tk.Label(frame, image=photo, bg="#1a1a2e")
                        children = frame.winfo_children()
                        if children: lbl.pack(before=children[-1])
                        else: lbl.pack()
                        self.refs.append(photo)
                except tk.TclError: pass
            self.win.after(0, update_ui)


# ============================================================================
# GOLDFISH GAME WINDOW
# ============================================================================
class GoldfishGameWindow:
    """Interactive goldfish (solitaire) Commander game with manual play and AI auto-play."""

    def __init__(self, parent, deck_cards, commanders, scryfall, pil_cache):
        self.scry = scryfall; self.pc = pil_cache; self.refs = []
        self.commanders = commanders
        self.deck_source = deck_cards
        self.cmdr_cmc = max((int(c.cmc) for c in commanders), default=0)

        self.win = tk.Toplevel(parent); self.win.title("Goldfish Game")
        self.win.geometry("1500x900"); self.win.configure(bg="#0d1117")
        self.win.minsize(1200, 700)

        # Game state
        self.library = []; self.hand = []; self.battlefield = []
        self.graveyard = []; self.exile = []; self.command_zone = list(commanders)
        self.turn = 0; self.lands_played = 0; self.max_lands_per_turn = 1
        self.mana_available = 0; self.mana_spent = 0
        self.log_lines = []; self.commander_cast_count = 0
        self.game_over = False

        self._build_ui()
        self._new_game()

    def _build_ui(self):
        bg = "#0d1117"
        # Top toolbar
        tb = tk.Frame(self.win, bg="#161b22"); tb.pack(fill=tk.X)
        tk.Button(tb, text="New Game", font=("Segoe UI",10,"bold"), bg="#238636", fg="white",
                  command=self._new_game).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(tb, text="Next Turn", font=("Segoe UI",10,"bold"), bg="#1f6feb", fg="white",
                  command=self._next_turn).pack(side=tk.LEFT, padx=5)
        tk.Button(tb, text="Draw", font=("Segoe UI",10,"bold"), bg="#2E8B57", fg="white",
                  command=lambda: self._do_draw(1)).pack(side=tk.LEFT, padx=5)

        tk.Frame(tb, bg="#30363d", width=2).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=3)
        tk.Button(tb, text="AI Auto-Play Turn", font=("Segoe UI",10,"bold"), bg="#8957e5", fg="white",
                  command=self._ai_play_turn).pack(side=tk.LEFT, padx=5)
        self.ai_turns_var = tk.StringVar(value="5")
        tk.Label(tb, text="Auto-play turns:", bg="#161b22", fg="#c9d1d9",
                 font=("Segoe UI",9)).pack(side=tk.LEFT, padx=(10,3))
        tk.Entry(tb, textvariable=self.ai_turns_var, width=4, font=("Segoe UI",10),
                 bg="#0d1117", fg="white", insertbackground="white").pack(side=tk.LEFT)
        tk.Button(tb, text="AI Auto-Play N Turns", font=("Segoe UI",10,"bold"), bg="#8957e5", fg="white",
                  command=self._ai_play_n_turns).pack(side=tk.LEFT, padx=5)

        self.turn_info = tk.StringVar(value="Turn 0")
        tk.Label(tb, textvariable=self.turn_info, bg="#161b22", fg="#58a6ff",
                 font=("Segoe UI",12,"bold")).pack(side=tk.RIGHT, padx=15)
        self.mana_info = tk.StringVar(value="Mana: 0/0")
        tk.Label(tb, textvariable=self.mana_info, bg="#161b22", fg="#7ee787",
                 font=("Segoe UI",11,"bold")).pack(side=tk.RIGHT, padx=10)

        # Main content: left = game zones, right = log
        main = tk.PanedWindow(self.win, orient=tk.HORIZONTAL, bg=bg, sashwidth=4)
        main.pack(fill=tk.BOTH, expand=True)

        game_frame = tk.Frame(main, bg=bg); main.add(game_frame, width=1100)

        # Command Zone
        cz_frame = tk.LabelFrame(game_frame, text="Command Zone", bg="#161b22", fg="#c9d1d9",
                                  font=("Segoe UI",10,"bold"), padx=5, pady=5)
        cz_frame.pack(fill=tk.X, padx=10, pady=(5,2))
        self.cz_display = tk.Frame(cz_frame, bg="#161b22")
        self.cz_display.pack(fill=tk.X)

        # Battlefield
        bf_frame = tk.LabelFrame(game_frame, text="Battlefield", bg="#161b22", fg="#c9d1d9",
                                  font=("Segoe UI",10,"bold"), padx=5, pady=5)
        bf_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)
        bf_canvas = tk.Canvas(bf_frame, bg="#0d1117", highlightthickness=0)
        bf_sb = tk.Scrollbar(bf_frame, orient=tk.HORIZONTAL, command=bf_canvas.xview)
        bf_canvas.configure(xscrollcommand=bf_sb.set)
        bf_sb.pack(side=tk.BOTTOM, fill=tk.X)
        bf_canvas.pack(fill=tk.BOTH, expand=True)
        self.bf_inner = tk.Frame(bf_canvas, bg="#0d1117")
        bf_canvas.create_window((0,0), window=self.bf_inner, anchor=tk.NW)
        self.bf_inner.bind("<Configure>", lambda e: bf_canvas.configure(scrollregion=bf_canvas.bbox("all")))
        self.bf_canvas = bf_canvas

        # Hand
        hd_frame = tk.LabelFrame(game_frame, text="Hand", bg="#161b22", fg="#c9d1d9",
                                  font=("Segoe UI",10,"bold"), padx=5, pady=5)
        hd_frame.pack(fill=tk.X, padx=10, pady=2)
        hd_canvas = tk.Canvas(hd_frame, bg="#0d1117", highlightthickness=0, height=200)
        hd_sb = tk.Scrollbar(hd_frame, orient=tk.HORIZONTAL, command=hd_canvas.xview)
        hd_canvas.configure(xscrollcommand=hd_sb.set)
        hd_sb.pack(side=tk.BOTTOM, fill=tk.X)
        hd_canvas.pack(fill=tk.BOTH, expand=True)
        self.hd_inner = tk.Frame(hd_canvas, bg="#0d1117")
        hd_canvas.create_window((0,0), window=self.hd_inner, anchor=tk.NW)
        self.hd_inner.bind("<Configure>", lambda e: hd_canvas.configure(scrollregion=hd_canvas.bbox("all")))
        self.hd_canvas = hd_canvas

        # Zones bar (graveyard/exile/library counts)
        zones_bar = tk.Frame(game_frame, bg="#161b22")
        zones_bar.pack(fill=tk.X, padx=10, pady=(2,5))
        self.zones_info = tk.StringVar(value="")
        tk.Label(zones_bar, textvariable=self.zones_info, bg="#161b22", fg="#8b949e",
                 font=("Segoe UI",10)).pack(side=tk.LEFT)

        # Right panel: log
        log_frame = tk.Frame(main, bg=bg); main.add(log_frame, width=350)
        tk.Label(log_frame, text="Game Log", bg=bg, fg="#e94560",
                 font=("Segoe UI",11,"bold")).pack(anchor=tk.W, padx=5, pady=5)
        self.log_text = tk.Text(log_frame, font=("Consolas",9), bg="#161b22", fg="#c9d1d9",
                                state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0,5))
        self.log_text.tag_configure("turn", foreground="#58a6ff", font=("Consolas",10,"bold"))
        self.log_text.tag_configure("play", foreground="#7ee787")
        self.log_text.tag_configure("draw", foreground="#d2a8ff")
        self.log_text.tag_configure("info", foreground="#8b949e")
        self.log_text.tag_configure("cmdr", foreground="#FFD700", font=("Consolas",10,"bold"))

    def _log(self, msg, tag="info"):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _new_game(self):
        deck = SimEngine.build_deck(self.deck_source)
        random.shuffle(deck)
        self.hand = deck[:7]; self.library = deck[7:]
        self.battlefield = []; self.graveyard = []; self.exile = []
        self.command_zone = list(self.commanders)
        self.turn = 0; self.lands_played = 0; self.mana_available = 0; self.mana_spent = 0
        self.commander_cast_count = 0; self.game_over = False
        self.log_text.configure(state=tk.NORMAL); self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self._log("=== NEW GAME ===", "turn")
        if self.commanders:
            names = ", ".join(c.name for c in self.commanders)
            self._log(f"Commander: {names}", "cmdr")
        self._log(f"Opening hand: {len(self.hand)} cards", "draw")
        self._log(f"Library: {len(self.library)} cards", "info")
        self._render_all()

    def _next_turn(self):
        self.turn += 1; self.lands_played = 0
        self.mana_spent = 0
        # Recalculate max land drops from permanents on battlefield
        self.max_lands_per_turn = 1
        for perm in self.battlefield:
            oracle = perm.oracle_text.lower() if perm.oracle_text else ""
            if "additional land" in oracle:
                m = re.search(r'play\s+(\w+)\s+additional\s+lands?', oracle)
                if m:
                    n = self._parse_number(m.group(1))
                    self.max_lands_per_turn += n
                else:
                    self.max_lands_per_turn = max(self.max_lands_per_turn, 2)
        # Untap: recalculate mana from lands on battlefield
        self.mana_available = sum(1 for c in self.battlefield if c.category == "Land")
        # Ramp permanents add mana
        self.mana_available += sum(1 for c in self.battlefield if c.category == "Ramp")
        self._log(f"\n--- TURN {self.turn} ---", "turn")
        self._log(f"Untap, upkeep. Mana available: {self.mana_available}", "info")
        # Upkeep triggers
        self._check_upkeep_triggers()
        # Draw
        if self.turn > 1 or self.turn == 1:  # always draw in goldfish
            self._do_draw(1)
        self._render_all()

    def _do_draw(self, n=1):
        for _ in range(n):
            if self.library:
                card = self.library.pop(0)
                self.hand.append(card)
                self._log(f"Draw: {card.name} ({card.category})", "draw")
            else:
                self._log("Library empty!", "info")
        self._render_all()

    def _play_card(self, card):
        """Play a card from hand to battlefield."""
        if card.category == "Land":
            if self.lands_played >= self.max_lands_per_turn:
                self._log(f"Can't play {card.name} — already played a land this turn.", "info")
                return
            self.hand.remove(card); self.battlefield.append(card)
            self.lands_played += 1; self.mana_available += 1
            self._log(f"Play land: {card.name} (lands: {self.lands_played})", "play")
            self._resolve_etb_effects(card)
        else:
            cost = int(card.cmc)
            avail = self.mana_available - self.mana_spent
            if cost > avail:
                self._log(f"Can't cast {card.name} (CMC {cost}) — only {avail} mana available.", "info")
                return
            self.hand.remove(card); self.battlefield.append(card)
            self.mana_spent += cost
            self._log(f"Cast: {card.name} (CMC {cost}, mana left: {self.mana_available - self.mana_spent})", "play")
            self._resolve_etb_effects(card)
            self._check_cast_triggers(card)
        self._render_all()

    def _play_commander(self, cmdr):
        """Cast commander from command zone."""
        cost = int(cmdr.cmc) + (2 * self.commander_cast_count)
        avail = self.mana_available - self.mana_spent
        if cost > avail:
            self._log(f"Can't cast {cmdr.name} (cost {cost} with tax) — only {avail} mana.", "info")
            return
        self.command_zone.remove(cmdr); self.battlefield.append(cmdr)
        self.mana_spent += cost; self.commander_cast_count += 1
        self._log(f"Cast commander: {cmdr.name} (cost {cost} with {(self.commander_cast_count-1)*2} tax)", "cmdr")
        self._resolve_etb_effects(cmdr)
        self._check_cast_triggers(cmdr)
        self._render_all()

    # ---- EFFECT RESOLUTION ----
    NUMBER_WORDS = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                    "six":6,"seven":7,"eight":8,"nine":9,"ten":10}

    @staticmethod
    def _parse_draw_count(text):
        """Extract how many cards a draw effect draws from oracle text snippet."""
        text = text.lower()
        m = re.search(r'draws?\s+(\w+)\s+cards?', text)
        if m:
            word = m.group(1)
            if word in GoldfishGameWindow.NUMBER_WORDS: return GoldfishGameWindow.NUMBER_WORDS[word]
            if word.isdigit(): return int(word)
        if "draw cards equal to" in text: return 2  # approximate
        if "draw a card" in text: return 1
        return 0

    @staticmethod
    def _parse_number(word):
        """Parse a number word or digit string into int."""
        if word in GoldfishGameWindow.NUMBER_WORDS: return GoldfishGameWindow.NUMBER_WORDS[word]
        if word.isdigit(): return int(word)
        return 0

    def _resolve_etb_effects(self, card):
        """Check if a card has enter-the-battlefield triggers (draw, extra lands, etc.)."""
        oracle = card.oracle_text.lower() if card.oracle_text else ""
        if not oracle: return

        # -- Extra land drops --
        self._check_extra_land_drops(card, is_etb=True)

        # -- Draw effects --
        # ETB triggers: "when ~ enters the battlefield, draw"
        etb_patterns = [
            r'when\s+.{0,30}\s+enters?\s+(?:the\s+battlefield)?[^.]*?draw[^.]*',
            r'when\s+you\s+unlock\s+this\s+door[^.]*?draw[^.]*',
            r'when\s+.{0,30}\s+enters[^.]*?draw[^.]*',
        ]
        for pattern in etb_patterns:
            m = re.search(pattern, oracle)
            if m:
                self._resolve_draw_effect(card, m.group(0), "ETB")
                return

        # Cantrips: instants/sorceries with direct draw
        if card.type_line and ("instant" in card.type_line.lower() or "sorcery" in card.type_line.lower()):
            self._resolve_spell_draw(card, oracle)
            return

        # Generic fallback: "when" + "enters" + "draw"
        if "when" in oracle and "draw" in oracle:
            sentences = oracle.split(".")
            for sentence in sentences:
                if "when" in sentence and "draw" in sentence and "enters" in sentence:
                    n = self._parse_draw_count(sentence)
                    if n > 0:
                        self._log(f"  → {card.name} trigger: draw {n}", "draw")
                        self._do_draw_silent(n)
                        return

    def _resolve_spell_draw(self, card, oracle):
        """Handle draw effects on instants/sorceries, including draw-then-put-back."""
        sentences = oracle.split(".")
        for sentence in sentences:
            s = sentence.strip()
            if any(skip in s for skip in ["deals combat", "you control dies"]):
                continue
            n = self._parse_draw_count(s)
            if n > 0:
                # Check for "then put N back on top" (Brainstorm, Telling Time, etc.)
                putback = re.search(r'(?:then\s+)?put\s+(\w+)\s+(?:of them|cards?)\s+(?:back\s+)?(?:on\s+top|from your hand)', oracle)
                if putback:
                    pb_count = self._parse_number(putback.group(1))
                    self._resolve_draw_putback(card, n, pb_count)
                else:
                    self._log(f"  → {card.name}: draw {n}", "draw")
                    self._do_draw_silent(n)
                # Check for scry after draw
                scry_m = re.search(r'scry\s+(\w+)', oracle)
                if scry_m:
                    scry_n = self._parse_number(scry_m.group(1))
                    if scry_n > 0:
                        self._resolve_scry(card, scry_n)
                return

        # Standalone scry (Preordain: scry 2 then draw)
        if "scry" in oracle and "draw" in oracle:
            scry_m = re.search(r'scry\s+(\w+)', oracle)
            draw_n = self._parse_draw_count(oracle)
            if scry_m and draw_n > 0:
                scry_n = self._parse_number(scry_m.group(1))
                self._resolve_scry(card, scry_n)
                self._log(f"  → {card.name}: draw {draw_n}", "draw")
                self._do_draw_silent(draw_n)

    def _resolve_draw_putback(self, card, draw_n, putback_n):
        """Draw N cards, then put M cards from hand back on top of library (Brainstorm)."""
        self._log(f"  → {card.name}: draw {draw_n}, put {putback_n} back on top", "draw")
        self._do_draw_silent(draw_n)
        # AI picks worst cards to put back: highest CMC non-land cards, or excess lands
        if putback_n > 0 and len(self.hand) >= putback_n:
            # Score cards: lower = more desirable to keep
            def keep_score(c):
                # Lands we need: keep if we have few
                land_count = sum(1 for x in self.battlefield if x.category == "Land")
                if c.category == "Land":
                    return -5 if land_count < self.turn + 1 else 10  # put back excess lands
                if c.category == "Ramp": return -3  # keep ramp
                if c.category == "Draw": return -2  # keep draw
                avail = self.mana_available - self.mana_spent
                if int(c.cmc) > avail + 3: return int(c.cmc)  # put back uncastable expensive cards
                return 0
            # Sort hand by "worst" (highest score = put back first)
            candidates = sorted(self.hand, key=keep_score, reverse=True)
            for _ in range(min(putback_n, len(self.hand))):
                if candidates:
                    worst = candidates.pop(0)
                    self.hand.remove(worst)
                    self.library.insert(0, worst)  # put on top
                    self._log(f"    Put back: {worst.name}", "info")

    def _resolve_scry(self, card, n):
        """Scry N: look at top N, AI puts non-useful cards to bottom."""
        if not self.library or n <= 0: return
        look = min(n, len(self.library))
        top_cards = self.library[:look]
        # Keep lands if we need them, keep low-CMC playable cards on top
        land_count = sum(1 for c in self.battlefield if c.category == "Land")
        avail = self.mana_available - self.mana_spent
        keep_top = []
        send_bottom = []
        for c in top_cards:
            if c.category == "Land" and land_count < self.turn + 2:
                keep_top.append(c)
            elif c.category in ("Ramp", "Draw") and int(c.cmc) <= avail + 2:
                keep_top.append(c)
            elif int(c.cmc) <= avail + 1:
                keep_top.append(c)
            else:
                send_bottom.append(c)
        # Reconstruct library
        remaining = self.library[look:]
        self.library = keep_top + remaining + send_bottom
        if send_bottom:
            names = ", ".join(c.name for c in send_bottom)
            self._log(f"  → {card.name} scry {n}: bottom {len(send_bottom)} ({names})", "info")
        if keep_top:
            self._log(f"  → {card.name} scry {n}: kept {len(keep_top)} on top", "info")

    def _check_cast_triggers(self, cast_card):
        """Check battlefield for 'whenever you cast' triggers that draw."""
        for perm in self.battlefield:
            if perm is cast_card: continue
            oracle = perm.oracle_text.lower() if perm.oracle_text else ""
            if not oracle: continue

            cast_triggers = [
                r'whenever\s+you\s+cast\s+[^.]*?spell[^.]*?draw[^.]*',
                r'whenever\s+you\s+cast\s+[^.]*?draw[^.]*',
            ]
            for pattern in cast_triggers:
                m = re.search(pattern, oracle)
                if m:
                    trigger_text = m.group(0)
                    type_match = True
                    cast_type = cast_card.type_line.lower() if cast_card.type_line else ""
                    for ttype in ["creature", "instant", "sorcery", "enchantment", "artifact",
                                  "noncreature", "nonland"]:
                        if ttype in trigger_text:
                            if ttype == "noncreature":
                                type_match = "creature" not in cast_type
                            elif ttype == "nonland":
                                type_match = "land" not in cast_type
                            else:
                                type_match = ttype in cast_type
                            break

                    if type_match:
                        n = self._parse_draw_count(trigger_text)
                        if n > 0:
                            self._log(f"  → {perm.name} triggers: draw {n}", "draw")
                            self._do_draw_silent(n)
                            break

    def _check_upkeep_triggers(self):
        """Check battlefield for upkeep draw triggers and opponent-cast approximations."""
        for perm in self.battlefield:
            oracle = perm.oracle_text.lower() if perm.oracle_text else ""
            if not oracle: continue

            # Standard upkeep draw: "at the beginning of your upkeep, draw"
            m = re.search(r'(?:at the beginning of|during) your upkeep[^.]*?draw[^.]*', oracle)
            if m:
                n = self._parse_draw_count(m.group(0))
                if n > 0:
                    self._log(f"  → {perm.name} upkeep: draw {n}", "draw")
                    self._do_draw_silent(n)

            # Sylvan Library: "draw two additional cards" during draw step
            if re.search(r'draw\s+two\s+additional\s+cards', oracle):
                self._log(f"  → {perm.name}: draw 2 additional (keeping all, goldfish = no life pressure)", "draw")
                self._do_draw_silent(2)

        # Opponent-cast triggers (Rhystic Study, Mystic Remora, etc.)
        # In a 4-player pod, opponents cast ~2-3 spells per turn cycle
        self._check_opponent_cast_approximation()

    def _check_opponent_cast_approximation(self):
        """Approximate opponent spell casts for triggers like Rhystic Study."""
        opponent_casts_per_turn = 2  # conservative estimate for 4-player pod
        for perm in self.battlefield:
            oracle = perm.oracle_text.lower() if perm.oracle_text else ""
            if not oracle: continue

            # "Whenever an opponent casts a spell" + draw effect
            m = re.search(r'whenever\s+an?\s+opponent\s+casts\s+[^.]*?draw[^.]*', oracle)
            if m:
                n = self._parse_draw_count(m.group(0))
                if n > 0:
                    # Check for "unless that player pays" (Rhystic Study) — assume ~60% draw rate
                    if "unless" in m.group(0) or "may pay" in oracle:
                        effective = max(1, int(opponent_casts_per_turn * 0.6 * n))
                        self._log(f"  → {perm.name}: ~{effective} draws (opponents cast ~{opponent_casts_per_turn}/turn, ~60% don't pay)", "draw")
                    else:
                        effective = opponent_casts_per_turn * n
                        self._log(f"  → {perm.name}: ~{effective} draws (opponents cast ~{opponent_casts_per_turn}/turn)", "draw")
                    self._do_draw_silent(effective)

            # Mystic Remora style: "whenever an opponent casts a noncreature spell"
            m2 = re.search(r'whenever\s+an?\s+opponent\s+casts\s+a\s+noncreature[^.]*?draw[^.]*', oracle)
            if m2 and not m:  # don't double-trigger
                n = self._parse_draw_count(m2.group(0))
                if n > 0:
                    effective = max(1, int(opponent_casts_per_turn * 0.5 * n))
                    self._log(f"  → {perm.name}: ~{effective} draws (noncreature opponent casts)", "draw")
                    self._do_draw_silent(effective)

    def _check_activated_draw_abilities(self):
        """Check battlefield for activated abilities that draw cards (tap/mana cost)."""
        for perm in list(self.battlefield):  # copy list since we may remove (sacrifice)
            oracle = perm.oracle_text.lower() if perm.oracle_text else ""
            if not oracle or "draw" not in oracle: continue

            # Skip sacrifice-to-draw (Mind Stone): "{N}, {T}, Sacrifice ~: Draw"
            if re.search(r'sacrifice[^:]*:\s*[^.]*?draw', oracle):
                # Only activate if it's worth it (late game, need cards)
                avail = self.mana_available - self.mana_spent
                sac_cost_m = re.search(r'\{(\d+)\}[^:]*sacrifice', oracle)
                sac_mana = int(sac_cost_m.group(1)) if sac_cost_m else 0
                if avail >= sac_mana and len(self.hand) <= 2:
                    # Sacrifice it: remove from battlefield, draw
                    n = self._parse_draw_count(oracle)
                    self.battlefield.remove(perm)
                    self.mana_spent += sac_mana
                    self._log(f"  → Sacrifice {perm.name}: draw {n}", "draw")
                    self._do_draw_silent(max(n, 1))
                continue

            # The One Ring: cumulative burden counter draw
            if "burden counter" in oracle:
                # Track burden counters on this specific perm
                if not hasattr(perm, '_burden_counters'):
                    perm._burden_counters = 0
                perm._burden_counters += 1
                n = perm._burden_counters
                self._log(f"  → Activate {perm.name}: burden {n}, draw {n}", "draw")
                self._do_draw_silent(n)
                continue

            # Standard tap-to-draw: "{T}: draw a card" or "{N}, {T}: draw a card"
            tap_draw = re.search(r'\{t\}[^:]*:\s*[^.]*?draw[^.]*', oracle)
            if tap_draw:
                trigger_text = tap_draw.group(0)
                n = self._parse_draw_count(trigger_text)
                if n <= 0: continue

                # Parse mana cost: look for {N} before {T}
                mana_cost_match = re.search(r'\{(\d+)\}.*?\{t\}', oracle)
                mana_cost = 0
                if mana_cost_match:
                    mana_cost = int(mana_cost_match.group(1))

                # Also check for colored mana costs like {U}, {B} etc. (count as 1 each)
                colored_costs = re.findall(r'\{[wubrgWUBRG]\}', oracle.split("{t}")[0]) if "{t}" in oracle else []
                mana_cost += len(colored_costs)

                avail = self.mana_available - self.mana_spent
                if mana_cost <= avail:
                    self.mana_spent += mana_cost
                    self._log(f"  → Activate {perm.name}: pay {mana_cost}, draw {n}", "draw")
                    self._do_draw_silent(n)
                continue

            # Pattern: "{N}: draw a card" (no tap, repeatable — like Thrasios)
            notap_draw = re.search(r'\{(\d+)\}[^{]*?:\s*[^.]*?draw[^.]*', oracle)
            if notap_draw:
                trigger_text = notap_draw.group(0)
                n = self._parse_draw_count(trigger_text)
                mana_cost = int(notap_draw.group(1))
                if n > 0:
                    avail = self.mana_available - self.mana_spent
                    activations = avail // max(mana_cost, 1) if mana_cost > 0 else 1
                    activations = min(activations, 3)
                    if activations > 0:
                        total_cost = mana_cost * activations
                        self.mana_spent += total_cost
                        total_draw = n * activations
                        self._log(f"  → Activate {perm.name} x{activations}: pay {total_cost}, draw {total_draw}", "draw")
                        self._do_draw_silent(total_draw)

    def _check_extra_land_drops(self, card, is_etb=False):
        """Check if a card grants extra land drops (Exploration, Oracle of Mul Daya, etc.)."""
        oracle = card.oracle_text.lower() if card.oracle_text else ""
        if not oracle: return
        # "You may play an additional land" or "play two additional lands"
        m = re.search(r'(?:you may )?play\s+(\w+)\s+additional\s+lands?', oracle)
        if m:
            n = self._parse_number(m.group(1))
            if n > 0:
                self.max_lands_per_turn = 1 + n
                self._log(f"  → {card.name}: can now play {self.max_lands_per_turn} lands/turn", "play")
        # "You may play an additional land on each of your turns"
        elif "additional land" in oracle:
            self.max_lands_per_turn = max(self.max_lands_per_turn, 2)
            self._log(f"  → {card.name}: can now play 2 lands/turn", "play")

    def _do_draw_silent(self, n):
        """Draw n cards without re-rendering (caller renders)."""
        for _ in range(n):
            if self.library:
                card = self.library.pop(0)
                self.hand.append(card)
                self._log(f"    Drew: {card.name} ({card.category})", "draw")
            else:
                self._log("    Library empty!", "info")

    def _render_all(self):
        self.refs.clear()
        self.turn_info.set(f"Turn {self.turn}")
        avail = self.mana_available - self.mana_spent
        self.mana_info.set(f"Mana: {avail}/{self.mana_available}")
        land_count = sum(1 for c in self.battlefield if c.category == "Land")
        self.zones_info.set(
            f"Library: {len(self.library)}  |  Hand: {len(self.hand)}  |  "
            f"Battlefield: {len(self.battlefield)} ({land_count} lands)  |  "
            f"Graveyard: {len(self.graveyard)}  |  Exile: {len(self.exile)}")
        self._render_zone(self.cz_display, self.command_zone, click_fn=self._play_commander, sz=(100,140))
        self._render_zone(self.bf_inner, self.battlefield, sz=(100,140))
        self._render_zone(self.hd_inner, self.hand, click_fn=self._play_card, sz=(120,168))

    def _render_zone(self, frame, cards, click_fn=None, sz=(100,140)):
        for w in frame.winfo_children(): w.destroy()
        if not cards:
            tk.Label(frame, text="(empty)", bg=frame["bg"], fg="#484f58",
                     font=("Segoe UI",9,"italic")).pack(side=tk.LEFT, padx=10)
            return
        for card in cards:
            cf = tk.Frame(frame, bg=frame["bg"]); cf.pack(side=tk.LEFT, padx=2, pady=2)
            if card.name in self.pc:
                img = self.pc[card.name].resize(sz, Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl = tk.Label(cf, image=photo, bg=frame["bg"], cursor="hand2" if click_fn else "")
                lbl.pack()
                self.refs.append(photo)
                if click_fn:
                    lbl.bind("<Button-1>", lambda e, c=card: click_fn(c))
            else:
                col = CATEGORY_COLORS.get(card.category, "#708090")
                ph = tk.Frame(cf, bg=col, width=sz[0], height=sz[1])
                ph.pack_propagate(False); ph.pack()
                lbl = tk.Label(ph, text=f"{card.name}\n(CMC {int(card.cmc)})", wraplength=sz[0]-10,
                               bg=col, fg="white", font=("Segoe UI",7,"bold"),
                               cursor="hand2" if click_fn else "")
                lbl.pack(expand=True)
                if click_fn:
                    lbl.bind("<Button-1>", lambda e, c=card: click_fn(c))
                    ph.bind("<Button-1>", lambda e, c=card: click_fn(c))

    # ---- AI AUTO-PLAY ----
    def _ai_play_turn(self):
        """AI plays a single turn using heuristics."""
        if self.turn == 0:
            self._next_turn()
            return

        self._next_turn()
        self._ai_do_plays()

    def _ai_play_n_turns(self):
        """AI plays N turns."""
        try: n = int(self.ai_turns_var.get())
        except ValueError: n = 5
        if self.turn == 0:
            # Start the game first
            self._next_turn()
            self._ai_do_plays()
            n -= 1
        for _ in range(n):
            self._next_turn()
            self._ai_do_plays()

    def _ai_do_plays(self):
        """AI heuristic: play land, cast ramp, cast draw, cast commander, cast spells, activate abilities."""
        # 1. Play a land (prefer non-MDFC basic lands first)
        self._ai_play_land()

        # 2. Cast ramp spells (cheapest first)
        ramp = sorted([c for c in self.hand if c.category == "Ramp"], key=lambda c: c.cmc)
        for card in ramp:
            avail = self.mana_available - self.mana_spent
            if int(card.cmc) <= avail:
                self._play_card(card)

        # 3. Cast draw spells (cheapest first — they may draw into more lands/ramp)
        draw_spells = sorted([c for c in self.hand if c.category == "Draw"], key=lambda c: c.cmc)
        for card in draw_spells:
            avail = self.mana_available - self.mana_spent
            if int(card.cmc) <= avail:
                self._play_card(card)

        # 3b. After draw spells resolve, try to play another land if we drew one
        self._ai_play_land()

        # 4. Cast commander if able
        for cmdr in list(self.command_zone):
            cost = int(cmdr.cmc) + (2 * self.commander_cast_count)
            avail = self.mana_available - self.mana_spent
            if cost <= avail:
                self._play_commander(cmdr)

        # 5. Cast other spells (highest CMC first to use mana efficiently)
        castable = sorted([c for c in self.hand if c.category not in ("Land", "Ramp", "Draw")],
                          key=lambda c: c.cmc, reverse=True)
        for card in castable:
            avail = self.mana_available - self.mana_spent
            cost = int(card.cmc)
            if cost <= avail and cost > 0:
                self._play_card(card)

        # 6. Activate draw abilities with leftover mana
        self._check_activated_draw_abilities()

        # 7. Final land play attempt (from activated draw)
        self._ai_play_land()

        self._render_all()

    def _ai_play_land(self):
        """AI attempts to play a land from hand."""
        lands = [c for c in self.hand if c.category == "Land"]
        if lands and self.lands_played < self.max_lands_per_turn:
            basics = [c for c in lands if " // " not in c.name]
            chosen = basics[0] if basics else lands[0]
            self._play_card(chosen)


# ============================================================================
# MAIN GUI
# ============================================================================

# ============================================================================
# DECK EDITOR WINDOW
# ============================================================================
class DeckEditorWindow:
    """Full deck editor with Scryfall search + EDHREC recommendations, card images."""

    THUMB_SIZE = (170, 238)  # card thumbnail size
    GRID_COLS = 5

    def __init__(self, app):
        self.app = app
        self.refs = []  # image references to prevent GC
        self.edhrec_data = None
        self.edhrec_loading = False

        self.win = tk.Toplevel(app.root)
        self.win.title("Deck Editor")
        self.win.geometry("1200x800")
        self.win.configure(bg="#1a1a2e")
        self.win.transient(app.root)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Notebook with tabs
        self.nb = ttk.Notebook(self.win)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Scryfall Search
        self.scry_frame = ttk.Frame(self.nb)
        self.nb.add(self.scry_frame, text="  Add Cards (Scryfall)  ")
        self._build_scryfall_tab()

        # Tab 2: EDHREC Recommendations
        self.edhrec_frame = ttk.Frame(self.nb)
        self.nb.add(self.edhrec_frame, text="  EDHREC Recommendations  ")
        self._build_edhrec_tab()

        # Tab 3: Current deck overview with images
        self.deck_frame = ttk.Frame(self.nb)
        self.nb.add(self.deck_frame, text="  Deck Overview  ")
        self._build_deck_overview_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.win, textvariable=self.status_var, bg="#1a1a2e", fg="#e0e0e0",
                 anchor=tk.W, font=("Segoe UI", 9)).pack(fill=tk.X, padx=5, pady=2)

        # [BUGFIX] Mousewheel: route scroll to whichever canvas is in the active tab
        self._tab_canvases = [self.scry_canvas, self.edhrec_canvas, self.deck_canvas]
        for widget in [self.win, self.scry_canvas, self.edhrec_canvas, self.deck_canvas,
                       self.scry_inner, self.edhrec_inner, self.deck_inner]:
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

        # [BUGFIX] Auto-load EDHREC when switching to that tab; refresh deck overview
        self._edhrec_auto_loaded = False
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_mousewheel(self, event):
        """Route mousewheel to the active tab's canvas."""
        try:
            tab_idx = self.nb.index(self.nb.select())
        except Exception:
            return
        if tab_idx < len(self._tab_canvases):
            canvas = self._tab_canvases[tab_idx]
            if event.num == 4:
                canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                canvas.yview_scroll(3, "units")
            elif event.delta:
                canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_tab_changed(self, event=None):
        """Auto-load EDHREC data when switching to the EDHREC tab."""
        try:
            tab_idx = self.nb.index(self.nb.select())
        except Exception:
            return
        if tab_idx == 1 and not self._edhrec_auto_loaded and not self.edhrec_loading:
            self._edhrec_auto_loaded = True
            self._edhrec_load()
        if tab_idx == 2:
            self._render_deck_overview()

    def _on_close(self):
        self.app._refresh_tree()
        self.app._refresh_summary()
        self.win.destroy()

    # ======== SCRYFALL SEARCH TAB ========
    def _build_scryfall_tab(self):
        f = self.scry_frame
        top = ttk.Frame(f); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(top, text="Search Scryfall:", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.scry_search_var = tk.StringVar()
        self.scry_entry = ttk.Entry(top, textvariable=self.scry_search_var, font=("Segoe UI", 12), width=40)
        self.scry_entry.pack(side=tk.LEFT, padx=10)
        self.scry_entry.focus_set()
        ttk.Button(top, text="Search", command=self._scry_do_search).pack(side=tk.LEFT, padx=5)

        # Qty selector
        ttk.Label(top, text="Qty:").pack(side=tk.LEFT, padx=(15, 0))
        self.scry_qty_var = tk.StringVar(value="1")
        ttk.Spinbox(top, from_=1, to=99, width=4, textvariable=self.scry_qty_var,
                     font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=3)

        # Scrollable results grid
        container = ttk.Frame(f)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.scry_canvas = tk.Canvas(container, bg="#16213e", highlightthickness=0)
        self.scry_sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.scry_canvas.yview)
        self.scry_canvas.configure(yscrollcommand=self.scry_sb.set)
        self.scry_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.scry_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scry_inner = tk.Frame(self.scry_canvas, bg="#16213e")
        self.scry_canvas.create_window((0, 0), window=self.scry_inner, anchor=tk.NW)
        self.scry_inner.bind("<Configure>", lambda e: self.scry_canvas.configure(
            scrollregion=self.scry_canvas.bbox("all")))

        # Autocomplete on keyrelease with debounce
        self._scry_after = None
        self.scry_entry.bind("<KeyRelease>", self._scry_on_key)
        self.scry_entry.bind("<Return>", lambda e: self._scry_do_search())

    def _scry_on_key(self, event):
        if self._scry_after:
            self.win.after_cancel(self._scry_after)
        self._scry_after = self.win.after(400, self._scry_autocomplete)

    def _scry_autocomplete(self):
        q = self.scry_search_var.get().strip()
        if len(q) < 2: return
        def go():
            names = self.app.scry.autocomplete(q)
            self.win.after(0, lambda: self._scry_show_autocomplete(names))
        threading.Thread(target=go, daemon=True).start()

    def _scry_show_autocomplete(self, names):
        """Show autocomplete as small card images grid."""
        if not names: return
        self.status_var.set(f"Found {len(names)} suggestions, loading images...")
        # Fetch full data for top results (with images)
        def go():
            results = []
            for name in names[:12]:
                sc = self.app.scry.fetch_by_name(name)
                if sc:
                    results.append(sc)
            self.win.after(0, lambda: self._scry_render_results(results))
        threading.Thread(target=go, daemon=True).start()

    def _scry_do_search(self):
        q = self.scry_search_var.get().strip()
        if not q: return
        self.status_var.set(f"Searching '{q}'...")
        def go():
            try:
                url = f"https://api.scryfall.com/cards/search?q={requests.utils.quote(q)}&order=edhrec"
                r = requests.get(url, headers={"User-Agent": "FasterFishing/1.0"}, timeout=10)
                if r.status_code == 200:
                    cards = r.json().get("data", [])[:20]
                    self.win.after(0, lambda: self._scry_render_results(cards))
                else:
                    self.win.after(0, lambda: self.status_var.set("No results found."))
            except Exception as e:
                self.win.after(0, lambda: self.status_var.set(f"Error: {e}"))
        threading.Thread(target=go, daemon=True).start()

    def _scry_render_results(self, cards_data):
        """Render Scryfall results as a grid of card images with Add buttons."""
        for w in self.scry_inner.winfo_children():
            w.destroy()
        self.refs.clear()
        if not cards_data:
            tk.Label(self.scry_inner, text="No results", bg="#16213e", fg="#e0e0e0").pack()
            return
        my_cards = {c.name.lower() for c in self.app.cards}
        cols = self.GRID_COLS

        for idx, sc in enumerate(cards_data):
            row, col = divmod(idx, cols)
            cell = tk.Frame(self.scry_inner, bg="#16213e", padx=4, pady=4)
            cell.grid(row=row, column=col, sticky=tk.N)
            name = sc.get("name", "?")
            in_deck = name.lower() in my_cards

            # Image
            img_url = ""
            if "image_uris" in sc:
                img_url = sc["image_uris"].get("normal", sc["image_uris"].get("small", ""))
            elif "card_faces" in sc and sc["card_faces"]:
                face = sc["card_faces"][0]
                if "image_uris" in face:
                    img_url = face["image_uris"].get("normal", face["image_uris"].get("small", ""))

            self._load_card_image_async(cell, img_url, name)

            # Name + CMC label
            cmc = sc.get("cmc", 0)
            lbl_text = f"{name} (CMC {int(cmc)})"
            fg = "#FFD700" if in_deck else "#e0e0e0"
            tk.Label(cell, text=lbl_text, bg="#16213e", fg=fg, font=("Segoe UI", 8),
                     wraplength=self.THUMB_SIZE[0]).pack()

            # Button: Add or In Deck
            if in_deck:
                tk.Label(cell, text="✓ In Deck", bg="#16213e", fg="#2ECC71",
                         font=("Segoe UI", 8, "bold")).pack()
            else:
                tk.Button(cell, text="+ Add to Deck", font=("Segoe UI", 8, "bold"),
                          bg="#0f3460", fg="white", cursor="hand2",
                          command=lambda s=sc: self._add_from_scryfall(s)).pack()

        self.status_var.set(f"Showing {len(cards_data)} cards")
        self.scry_inner.update_idletasks()
        self.scry_canvas.configure(scrollregion=self.scry_canvas.bbox("all"))

    def _load_card_image_async(self, parent, url, name, size=None):
        """Load a card image asynchronously and display in parent frame."""
        if not size:
            size = self.THUMB_SIZE
        # Check PIL cache first
        if name in self.app.pil_cache:
            pil = self.app.pil_cache[name].resize(size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            lbl = tk.Label(parent, image=photo, bg="#16213e")
            lbl.pack()
            self.refs.append(photo)
            return

        # Placeholder
        ph = tk.Frame(parent, bg="#2a2a4e", width=size[0], height=size[1])
        ph.pack_propagate(False)
        ph.pack()
        tk.Label(ph, text="Loading...", bg="#2a2a4e", fg="#708090",
                 font=("Segoe UI", 8)).pack(expand=True)

        if not url:
            return

        def go():
            pil_img = self.app.scry.fetch_image(url, name)
            if pil_img:
                self.app.pil_cache[name] = pil_img
                self.win.after(0, lambda: self._replace_placeholder(ph, pil_img, size))

        threading.Thread(target=go, daemon=True).start()

    def _replace_placeholder(self, placeholder, pil_img, size):
        try:
            resized = pil_img.resize(size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            for w in placeholder.winfo_children():
                w.destroy()
            placeholder.configure(width=size[0], height=size[1])
            lbl = tk.Label(placeholder, image=photo, bg="#16213e")
            lbl.pack()
            self.refs.append(photo)
        except tk.TclError:
            pass  # widget destroyed

    def _add_from_scryfall(self, sc):
        """Add a card from Scryfall data to the deck."""
        name = sc.get("name", "")
        if not name: return
        try:
            qty = int(self.scry_qty_var.get())
        except ValueError:
            qty = 1

        # Check duplicate
        for c in self.app.cards:
            if c.name.lower() == name.lower():
                c.quantity += qty
                self.app._refresh_tree()
                self.app._refresh_summary()
                self.status_var.set(f"+{qty} {c.name} (now x{c.quantity})")
                return

        # Build card
        layout = sc.get("layout", "normal")
        has_faces = "card_faces" in sc and sc["card_faces"]
        img = ""
        if "image_uris" in sc:
            img = sc["image_uris"].get("normal", sc["image_uris"].get("small", ""))
        elif has_faces:
            face = sc["card_faces"][0]
            if "image_uris" in face:
                img = face["image_uris"].get("normal", face["image_uris"].get("small", ""))

        cmc = sc.get("cmc", 0)
        oracle = sc.get("oracle_text", "")
        if not oracle and has_faces:
            oracle = " // ".join(f.get("oracle_text", "") for f in sc["card_faces"])

        card = Card(name=name, quantity=qty, mana_cost=sc.get("mana_cost", ""),
                    cmc=cmc, type_line=sc.get("type_line", ""), image_uri=img,
                    scryfall_id=sc.get("id", ""), oracle_text=oracle, layout=layout)

        # Auto-categorize
        tl = card.type_line.lower()
        if " // " in tl:
            faces = [f.strip() for f in tl.split(" // ")]
            if any("land" in f for f in faces):
                card.category = "Land"
        elif "land" in tl and "creature" not in tl:
            card.category = "Land"
        if card.category == "Other":
            for cat, tag in CATEGORY_TAGS.items():
                matched = self.app.scry.search_names_with_tag(tag, [name])
                if name.lower() in matched:
                    card.category = cat
                    break
            else:
                if "creature" in tl:
                    card.category = "Creature"

        self.app.cards.append(card)
        self.app._refresh_tree()
        self.app._refresh_summary()
        self.status_var.set(f"Added: {name} x{qty}")
        self.app._sts(f"Added {name} x{qty} to deck")

        # Preload image
        if name not in self.app.pil_cache and img:
            def load():
                pil = self.app.scry.fetch_image(img, name)
                if pil:
                    self.app.pil_cache[name] = pil
            threading.Thread(target=load, daemon=True).start()

    # ======== EDHREC TAB ========
    def _build_edhrec_tab(self):
        f = self.edhrec_frame
        top = ttk.Frame(f); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(top, text="EDHREC Recommendations", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.edhrec_load_btn = ttk.Button(top, text="Load Recommendations", command=self._edhrec_load)
        self.edhrec_load_btn.pack(side=tk.LEFT, padx=15)
        self.edhrec_status = tk.StringVar(value="Set a commander, then click Load.")
        tk.Label(top, textvariable=self.edhrec_status, bg="#1a1a2e", fg="#00d2ff",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)

        # Filter row
        filt = ttk.Frame(f); filt.pack(fill=tk.X, padx=10)
        ttk.Label(filt, text="Filter:").pack(side=tk.LEFT)
        self.edhrec_filter_var = tk.StringVar(value="All")
        self.edhrec_filter_combo = ttk.Combobox(filt, textvariable=self.edhrec_filter_var,
                                                  state="readonly", width=25)
        self.edhrec_filter_combo["values"] = ["All"]
        self.edhrec_filter_combo.pack(side=tk.LEFT, padx=5)
        self.edhrec_filter_combo.bind("<<ComboboxSelected>>", lambda e: self._edhrec_render())

        self.edhrec_hide_in_deck = tk.BooleanVar(value=True)
        tk.Checkbutton(filt, text="Hide cards already in deck", variable=self.edhrec_hide_in_deck,
                       bg="#1a1a2e", fg="#e0e0e0", selectcolor="#1a1a2e",
                       activebackground="#1a1a2e", activeforeground="#e0e0e0",
                       command=self._edhrec_render).pack(side=tk.LEFT, padx=15)

        # Scrollable grid
        container = ttk.Frame(f)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        self.edhrec_canvas = tk.Canvas(container, bg="#16213e", highlightthickness=0)
        self.edhrec_sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.edhrec_canvas.yview)
        self.edhrec_canvas.configure(yscrollcommand=self.edhrec_sb.set)
        self.edhrec_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.edhrec_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.edhrec_inner = tk.Frame(self.edhrec_canvas, bg="#16213e")
        self.edhrec_canvas.create_window((0, 0), window=self.edhrec_inner, anchor=tk.NW)
        self.edhrec_inner.bind("<Configure>", lambda e: self.edhrec_canvas.configure(
            scrollregion=self.edhrec_canvas.bbox("all")))

        self.edhrec_all_recs = {}  # section -> list of card dicts

    def _edhrec_load(self):
        cmdrs = [c for c in self.app.cards if c.is_commander]
        if not cmdrs:
            self.edhrec_status.set("No commander set! Set one on the Deck tab first.")
            return
        if self.edhrec_loading:
            return
        self.edhrec_loading = True
        cmdr = cmdrs[0]
        self.edhrec_load_btn.configure(state=tk.DISABLED)
        self.edhrec_status.set(f"Loading EDHREC data for {cmdr.name}...")

        def go():
            data = self.app._fetch_edhrec(cmdr.name)
            self.win.after(0, lambda: self._edhrec_process(data, cmdr.name))
        threading.Thread(target=go, daemon=True).start()

    def _edhrec_process(self, data, cmdr_name):
        self.edhrec_load_btn.configure(state=tk.NORMAL)
        self.edhrec_loading = False
        if "error" in data:
            self.edhrec_status.set(f"Error: {data['error']}")
            return

        self.edhrec_data = data
        container = data.get("container", data)
        card_lists = container.get("json_dict", container).get("cardlists", [])
        total_decks = container.get("json_dict", {}).get("num_decks", "?")

        self.edhrec_all_recs = {}
        sections = ["All"]
        for cl in card_lists:
            header = cl.get("header", "Cards")
            cards = cl.get("cardviews", [])
            if not cards: continue
            sections.append(header)
            self.edhrec_all_recs[header] = cards

        self.edhrec_filter_combo["values"] = sections
        self.edhrec_status.set(f"Loaded {sum(len(v) for v in self.edhrec_all_recs.values())} recommendations from {total_decks} decks")
        self._edhrec_render()

    def _edhrec_render(self):
        """Render EDHREC recommendations as card image grid."""
        for w in self.edhrec_inner.winfo_children():
            w.destroy()
        self.refs.clear()

        if not self.edhrec_all_recs:
            tk.Label(self.edhrec_inner, text="Click 'Load Recommendations' to fetch data",
                     bg="#16213e", fg="#708090").pack(padx=20, pady=20)
            return

        filt = self.edhrec_filter_var.get()
        hide = self.edhrec_hide_in_deck.get()
        my_cards = {c.name.lower() for c in self.app.cards}

        all_cards = []
        if filt == "All":
            for section, cards in self.edhrec_all_recs.items():
                all_cards.extend(cards)
        else:
            all_cards = self.edhrec_all_recs.get(filt, [])

        # Deduplicate by name
        seen = set()
        unique_cards = []
        for cv in all_cards:
            name = cv.get("name", "")
            if not name or name.lower() in seen: continue
            if hide and name.lower() in my_cards: continue
            seen.add(name.lower())
            unique_cards.append(cv)

        cols = self.GRID_COLS
        for idx, cv in enumerate(unique_cards[:40]):
            row, col = divmod(idx, cols)
            cell = tk.Frame(self.edhrec_inner, bg="#16213e", padx=4, pady=4)
            cell.grid(row=row, column=col, sticky=tk.N)

            name = cv.get("name", "?")
            in_deck = name.lower() in my_cards

            # Build Scryfall image URL from card name
            safe_name = requests.utils.quote(name)
            img_url = f"https://api.scryfall.com/cards/named?format=image&exact={safe_name}&version=normal"
            self._load_card_image_async(cell, img_url, name)

            # Synergy + inclusion info
            synergy = cv.get("synergy", 0)
            syn_str = f"+{synergy:.0%}" if isinstance(synergy, float) and synergy > 0 else (
                f"{synergy:.0%}" if isinstance(synergy, float) else str(synergy))
            inclusion = cv.get("inclusion", cv.get("num_decks", 0))
            inc_str = f"{inclusion:.0%}" if isinstance(inclusion, float) and inclusion <= 1 else str(inclusion)

            info_text = f"{name}\nSyn: {syn_str}  In: {inc_str}"
            fg = "#FFD700" if in_deck else "#e0e0e0"
            tk.Label(cell, text=info_text, bg="#16213e", fg=fg, font=("Segoe UI", 7),
                     wraplength=self.THUMB_SIZE[0], justify=tk.CENTER).pack()

            if in_deck:
                tk.Label(cell, text="✓ In Deck", bg="#16213e", fg="#2ECC71",
                         font=("Segoe UI", 8, "bold")).pack()
            else:
                tk.Button(cell, text="+ Add to Deck", font=("Segoe UI", 8, "bold"),
                          bg="#0f3460", fg="white", cursor="hand2",
                          command=lambda n=name: self._add_edhrec_card(n)).pack()

        self.status_var.set(f"Showing {min(len(unique_cards), 40)} of {len(unique_cards)} recommendations")
        self.edhrec_inner.update_idletasks()
        self.edhrec_canvas.configure(scrollregion=self.edhrec_canvas.bbox("all"))

    def _add_edhrec_card(self, name):
        """Add a card by name from EDHREC — fetch from Scryfall and add."""
        self.status_var.set(f"Adding {name}...")
        def go():
            sc = self.app.scry.fetch_by_name(name)
            if sc:
                self.win.after(0, lambda: self._add_from_scryfall(sc))
                self.win.after(100, self._edhrec_render)
            else:
                self.win.after(0, lambda: self.status_var.set(f"Not found: {name}"))
        threading.Thread(target=go, daemon=True).start()

    # ======== DECK OVERVIEW TAB ========
    def _build_deck_overview_tab(self):
        f = self.deck_frame
        top = ttk.Frame(f); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(top, text="Deck Overview (Card Images)", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(top, text="Refresh", command=self._render_deck_overview).pack(side=tk.LEFT, padx=15)
        self.deck_overview_filter = tk.StringVar(value="All")
        cats = ["All"] + ALL_CATEGORIES
        ttk.Combobox(top, textvariable=self.deck_overview_filter, state="readonly",
                     values=cats, width=15).pack(side=tk.LEFT, padx=5)
        self.deck_overview_filter.trace_add("write", lambda *a: self._render_deck_overview())

        container = ttk.Frame(f)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.deck_canvas = tk.Canvas(container, bg="#16213e", highlightthickness=0)
        self.deck_sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.deck_canvas.yview)
        self.deck_canvas.configure(yscrollcommand=self.deck_sb.set)
        self.deck_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.deck_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.deck_inner = tk.Frame(self.deck_canvas, bg="#16213e")
        self.deck_canvas.create_window((0, 0), window=self.deck_inner, anchor=tk.NW)
        self.deck_inner.bind("<Configure>", lambda e: self.deck_canvas.configure(
            scrollregion=self.deck_canvas.bbox("all")))

        self.win.after(100, self._render_deck_overview)

    def _render_deck_overview(self):
        for w in self.deck_inner.winfo_children():
            w.destroy()
        self.refs.clear()

        filt = self.deck_overview_filter.get()
        cards = self.app.cards if filt == "All" else [c for c in self.app.cards if c.category == filt]
        cols = self.GRID_COLS

        for idx, card in enumerate(sorted(cards, key=lambda c: (c.category, c.cmc, c.name))):
            row, col = divmod(idx, cols)
            cell = tk.Frame(self.deck_inner, bg="#16213e", padx=4, pady=4)
            cell.grid(row=row, column=col, sticky=tk.N)

            # Image
            if card.image_uri:
                self._load_card_image_async(cell, card.image_uri, card.name)
            elif card.name in self.app.pil_cache:
                self._load_card_image_async(cell, "", card.name)
            else:
                # Color placeholder
                cat_col = CATEGORY_COLORS.get(card.category, "#708090")
                ph = tk.Frame(cell, bg=cat_col, width=self.THUMB_SIZE[0], height=self.THUMB_SIZE[1])
                ph.pack_propagate(False); ph.pack()
                tk.Label(ph, text=f"{card.name}\n(CMC {int(card.cmc)})", wraplength=self.THUMB_SIZE[0]-10,
                         bg=cat_col, fg="white", font=("Segoe UI", 8, "bold")).pack(expand=True)

            # Info
            qty_str = f"x{card.quantity}" if card.quantity > 1 else ""
            cmdr_str = " ★" if card.is_commander else ""
            lbl = f"{card.name}{cmdr_str} {qty_str}\n[{card.category}]"
            tk.Label(cell, text=lbl, bg="#16213e", fg="#e0e0e0", font=("Segoe UI", 7),
                     wraplength=self.THUMB_SIZE[0], justify=tk.CENTER).pack()

            # Remove button
            tk.Button(cell, text="✕ Remove", font=("Segoe UI", 7), bg="#8B0000", fg="white",
                      cursor="hand2",
                      command=lambda c=card: self._remove_from_deck(c)).pack()

        self.status_var.set(f"Deck: {sum(c.quantity for c in self.app.cards)} cards ({len(cards)} shown)")
        self.deck_inner.update_idletasks()
        self.deck_canvas.configure(scrollregion=self.deck_canvas.bbox("all"))

    def _remove_from_deck(self, card):
        if card.quantity > 1:
            card.quantity -= 1
        else:
            self.app.cards.remove(card)
        self.app._refresh_tree()
        self.app._refresh_summary()
        self._render_deck_overview()
        self.status_var.set(f"Removed: {card.name}")



# ============================================================================
# MAIN APPLICATION
# ============================================================================
class FasterFishing:
    def __init__(self, root):
        self.root = root; self.root.title("MTG Commander Goldfish Simulator")
        self.root.geometry("1500x1000"); self.root.minsize(1100, 700)
        self.cards = []; self.scry = ScryfallClient()
        self.imgs = {}       # card_name -> ImageTk.PhotoImage (for preview display)
        self.pil_cache = {}  # card_name -> PIL.Image (raw, resizable)
        self.sel_idx = None
        self.style = ttk.Style(); self.style.theme_use("clam"); self._theme()
        self._build_ui()

    def _theme(self):
        bg, fg = "#1a1a2e", "#e0e0e0"; self.bg = bg; self.fg = fg
        self.root.configure(bg=bg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI",10))
        self.style.configure("TButton", font=("Segoe UI",10,"bold"))
        self.style.configure("H.TLabel", font=("Segoe UI",14,"bold"), foreground="#e94560", background=bg)
        self.style.configure("S.TLabel", font=("Segoe UI",11,"bold"), foreground="#e94560", background=bg)
        self.style.configure("TNotebook", background=bg)
        self.style.configure("TNotebook.Tab", font=("Segoe UI",10,"bold"), padding=[12,4])

    def _build_ui(self):
        self.nb = ttk.Notebook(self.root); self.nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for title, builder in [("  Import Deck  ", self._ui_import),
            ("  Deck & Categories  ", self._ui_deck), ("  Analysis  ", self._ui_analysis),
            ("  Simulations  ", self._ui_sim), ("  Goldfish Turns  ", self._ui_gf)]:
            f = ttk.Frame(self.nb); self.nb.add(f, text=title); builder(f)
        self.status = tk.StringVar(value="Ready - Import a deck to begin!")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    # ---- IMPORT TAB ----
    def _ui_import(self, f):
        uf = ttk.LabelFrame(f, text="Import from URL", padding=10); uf.pack(fill=tk.X, padx=10, pady=(10,5))
        ttk.Label(uf, text="Supported: Moxfield, Archidekt, MTGGoldfish").pack(anchor=tk.W)
        row = ttk.Frame(uf); row.pack(fill=tk.X, pady=5)
        self.url_e = ttk.Entry(row, font=("Segoe UI",11)); self.url_e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.url_b = ttk.Button(row, text="Import URL", command=self._imp_url); self.url_b.pack(side=tk.RIGHT)
        tf = ttk.LabelFrame(f, text="Import from Text", padding=10); tf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        ttk.Label(tf, text='Paste decklist (e.g. "1 Sol Ring" or "1x Sol Ring (CMR) 319"):').pack(anchor=tk.W)
        self.dtxt = scrolledtext.ScrolledText(tf, font=("Consolas",10), bg="#16213e", fg="#e0e0e0", insertbackground="#e0e0e0", height=20)
        self.dtxt.pack(fill=tk.BOTH, expand=True, pady=5)
        br = ttk.Frame(tf); br.pack(fill=tk.X)
        ttk.Button(br, text="Import Text", command=self._imp_txt).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(br, text="Clear", command=self._clr).pack(side=tk.LEFT)
        self.ip = ttk.Progressbar(br, mode="determinate", length=300); self.ip.pack(side=tk.RIGHT, padx=5)
        self.ip_lbl = ttk.Label(br, text=""); self.ip_lbl.pack(side=tk.RIGHT, padx=5)

    # ---- DECK TAB ----
    def _ui_deck(self, f):
        p = ttk.PanedWindow(f, orient=tk.HORIZONTAL); p.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left = ttk.Frame(p); p.add(left, weight=3)

        # Toolbar row
        tb = ttk.Frame(left); tb.pack(fill=tk.X, pady=(0,5))
        ttk.Label(tb, text="Deck", style="H.TLabel").pack(side=tk.LEFT)
        self.dcnt = tk.StringVar(value="0 cards"); ttk.Label(tb, textvariable=self.dcnt).pack(side=tk.LEFT, padx=10)
        ttk.Button(tb, text="Remove Selected", command=self._remove_card).pack(side=tk.RIGHT, padx=5)
        ttk.Button(tb, text="Edit Deck", command=self._open_deck_editor).pack(side=tk.RIGHT, padx=5)

        # Deck storage buttons
        ttk.Button(tb, text="Save Deck", command=self._save_deck).pack(side=tk.RIGHT, padx=2)
        ttk.Button(tb, text="Load Deck", command=self._load_deck).pack(side=tk.RIGHT, padx=2)

        # View toggle: Text List vs Image Grid
        self._deck_view_mode = tk.StringVar(value="list")
        view_frame = ttk.Frame(left); view_frame.pack(fill=tk.X, pady=(0,3))
        ttk.Radiobutton(view_frame, text="📋 List View", variable=self._deck_view_mode,
                        value="list", command=self._toggle_deck_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="🖼 Image Grid", variable=self._deck_view_mode,
                        value="grid", command=self._toggle_deck_view).pack(side=tk.LEFT, padx=5)

        # Commander display at top of left panel
        self.cmdr_display = ttk.Frame(left)
        self.cmdr_display.pack(fill=tk.X, pady=(0,3))
        self.cmdr_display_label = tk.Label(self.cmdr_display, text="No commander set",
            bg="#1a1a2e", fg="#FFD700", font=("Segoe UI", 10, "bold"), anchor=tk.W, padx=8, pady=4)
        self.cmdr_display_label.pack(fill=tk.X)
        self.cmdr_img_labels = []  # commander image labels in the display bar

        # Container that holds either the tree or the image grid
        self.deck_view_container = ttk.Frame(left)
        self.deck_view_container.pack(fill=tk.BOTH, expand=True)

        # Debounce tracking for add card dialog
        self._add_search_after_id = None

        # --- LIST VIEW (Treeview) ---
        self.tree_frame = ttk.Frame(self.deck_view_container)
        cols = ("cmdr","qty","name","category","cmc","type")
        self.tree = ttk.Treeview(self.tree_frame, columns=cols, show="headings", height=25)
        self._sort_col = "name"; self._sort_rev = False
        for c, w, a, display in [("cmdr",30,tk.CENTER,"*"),("qty",40,tk.CENTER,"Qty"),
                      ("name",200,tk.W,"Name"),("category",100,tk.CENTER,"Category"),
                      ("cmc",50,tk.CENTER,"CMC"),("type",180,tk.W,"Type")]:
            self.tree.heading(c, text=display, command=lambda col=c: self._sort_tree(col))
            self.tree.column(c, width=w, anchor=a)
        sb = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # --- IMAGE GRID VIEW ---
        self.grid_frame = ttk.Frame(self.deck_view_container)
        self.grid_canvas = tk.Canvas(self.grid_frame, bg="#16213e", highlightthickness=0)
        self.grid_scrollbar = ttk.Scrollbar(self.grid_frame, orient=tk.VERTICAL, command=self.grid_canvas.yview)
        self.grid_canvas.configure(yscrollcommand=self.grid_scrollbar.set)
        self.grid_inner = tk.Frame(self.grid_canvas, bg="#16213e")
        self.grid_canvas.create_window((0,0), window=self.grid_inner, anchor=tk.NW)
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.grid_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.grid_inner.bind("<Configure>", lambda e: self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all")))
        self.grid_canvas.bind("<MouseWheel>", lambda e: self.grid_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.grid_inner.bind("<MouseWheel>", lambda e: self.grid_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._grid_photos = []  # prevent GC

        # Show list view by default
        self.tree_frame.pack(fill=tk.BOTH, expand=True)

        # --- RIGHT PANEL ---
        right = ttk.Frame(p); p.add(right, weight=2)
        self.img_label = ttk.Label(right, text="Select a card to preview"); self.img_label.pack(pady=10)
        # Right-click to save image
        self.img_label.bind("<Button-3>", self._save_card_image_menu)
        # Drag hint
        self._save_img_hint = ttk.Label(right, text="Right-click image to save", foreground="#666")
        self._save_img_hint.pack()

        # Commander controls
        cmdr_frame = ttk.LabelFrame(right, text="Commander", padding=10)
        cmdr_frame.pack(fill=tk.X, padx=5, pady=(0,5))
        self.cmdr_info = tk.StringVar(value="No commander set")
        ttk.Label(cmdr_frame, textvariable=self.cmdr_info, foreground="#FFD700").pack(anchor=tk.W)
        cmdr_btns = ttk.Frame(cmdr_frame)
        cmdr_btns.pack(fill=tk.X, pady=(5,0))
        ttk.Button(cmdr_btns, text="Set Selected as Commander",
                   command=self._toggle_commander).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(cmdr_btns, text="Clear Commander(s)",
                   command=self._clear_commanders).pack(side=tk.LEFT)

        cf = ttk.LabelFrame(right, text="Override Category", padding=10); cf.pack(fill=tk.X, padx=5, pady=5)
        self.cat_var = tk.StringVar()
        self.cat_frame = cf
        for i, cat in enumerate(ALL_CATEGORIES):
            col = CATEGORY_COLORS.get(cat, "#708090")
            rb = tk.Radiobutton(cf, text=cat, variable=self.cat_var, value=cat, command=self._set_cat,
                bg=self.bg, fg=col, selectcolor=self.bg, activebackground=self.bg, activeforeground=col,
                font=("Segoe UI",10,"bold"), indicatoron=True)
            rb.grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)
        new_row = len(ALL_CATEGORIES) // 3 + 1
        tk.Button(cf, text="+ New Category", font=("Segoe UI",9,"bold"), bg="#0f3460", fg="white",
                  command=self._add_category).grid(row=new_row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        sf = ttk.LabelFrame(right, text="Deck Composition", padding=10); sf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.summary = tk.Text(sf, font=("Consolas",10), bg="#16213e", fg="#e0e0e0", height=12, state=tk.DISABLED)
        self.summary.pack(fill=tk.BOTH, expand=True)

    def _toggle_deck_view(self):
        """Switch between list and image grid view."""
        mode = self._deck_view_mode.get()
        if mode == "list":
            self.grid_frame.pack_forget()
            self.tree_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self.tree_frame.pack_forget()
            self.grid_frame.pack(fill=tk.BOTH, expand=True)
            self._render_deck_grid()

    def _render_deck_grid(self):
        """Render deck as image grid in the grid view."""
        for w in self.grid_inner.winfo_children():
            w.destroy()
        self._grid_photos = []
        if not self.cards:
            return
        # Commanders first, then sorted by category + name
        cmdrs = [c for c in self.cards if c.is_commander]
        others = sorted([c for c in self.cards if not c.is_commander],
                       key=lambda c: (c.category, c.name.lower()))
        all_sorted = cmdrs + others
        cols = 5
        thumb_w, thumb_h = 130, 182
        for i, card in enumerate(all_sorted):
            row, col = divmod(i, cols)
            cell = tk.Frame(self.grid_inner, bg="#16213e", padx=2, pady=2)
            cell.grid(row=row, column=col, padx=2, pady=2)
            # Load image
            if card.name in self.pil_cache:
                img = self.pil_cache[card.name].resize((thumb_w, thumb_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._grid_photos.append(photo)
                lbl = tk.Label(cell, image=photo, bg="#16213e", cursor="hand2")
                lbl.pack()
                lbl.bind("<Button-1>", lambda e, c=card: self._grid_card_click(c))
            else:
                lbl = tk.Label(cell, text=card.name[:20], bg="#2a2a4a", fg="#e0e0e0",
                              width=18, height=11, font=("Segoe UI", 7))
                lbl.pack()
                lbl.bind("<Button-1>", lambda e, c=card: self._grid_card_click(c))
                # Async load
                threading.Thread(target=lambda c=card: self._load_grid_img(c), daemon=True).start()
            # Commander badge
            if card.is_commander:
                tk.Label(cell, text="⭐ CMDR", bg="#FFD700", fg="#000", font=("Segoe UI", 7, "bold")).pack()
        self.grid_inner.update_idletasks()
        self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all"))

    def _load_grid_img(self, card):
        """Load a card image for grid view async."""
        img = self.scry.fetch_image(card.image_uri, card.name)
        if img:
            self.pil_cache[card.name] = img
            self.root.after(0, self._render_deck_grid)

    def _grid_card_click(self, card):
        """Handle clicking a card in grid view - show preview."""
        idx = self.cards.index(card)
        self.sel_idx = idx
        self.cat_var.set(card.category)
        if card.name in self.pil_cache:
            preview = self.pil_cache[card.name].resize((250, 349), Image.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
            self.imgs[card.name] = photo
            self.img_label.configure(image=photo, text="")
        else:
            self.img_label.configure(image="", text="Loading...")
            threading.Thread(target=self._load_img, args=(card,), daemon=True).start()

    def _save_card_image_menu(self, event):
        """Right-click on card preview to save image."""
        if self.sel_idx is None or self.sel_idx >= len(self.cards):
            return
        card = self.cards[self.sel_idx]
        if card.name not in self.pil_cache:
            return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label=f"Save '{card.name}' image...", command=lambda: self._save_card_image(card))
        menu.post(event.x_root, event.y_root)

    def _save_card_image(self, card):
        """Save card image to file."""
        if card.name not in self.pil_cache:
            return
        from tkinter import filedialog
        safe_name = re.sub(r'[^\w\-. ]', '_', card.name)
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
            initialfile=f"{safe_name}.png",
            title=f"Save {card.name} Image")
        if path:
            self.pil_cache[card.name].save(path)
            self._sts(f"Saved: {path}")

    def _update_cmdr_display(self):
        """Update the commander display bar at top of deck list."""
        for w in self.cmdr_img_labels:
            w.destroy()
        self.cmdr_img_labels = []
        cmdrs = [c for c in self.cards if c.is_commander]
        if not cmdrs:
            self.cmdr_display_label.configure(text="  No commander set")
            return
        names = " & ".join(c.name for c in cmdrs)
        cmc = max((int(c.cmc) for c in cmdrs), default=0)
        self.cmdr_display_label.configure(text=f"  ⭐ {names}  (CMC {cmc})")
        # Show small commander thumbnails
        for c in cmdrs:
            if c.name in self.pil_cache:
                thumb = self.pil_cache[c.name].resize((50, 70), Image.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                lbl = tk.Label(self.cmdr_display, image=photo, bg="#1a1a2e")
                lbl.image = photo  # prevent GC
                lbl.pack(side=tk.LEFT, padx=2)
                self.cmdr_img_labels.append(lbl)

    def _save_deck(self):
        """Save current deck to JSON file."""
        if not self.cards:
            messagebox.showwarning("Empty Deck", "No cards to save."); return
        from tkinter import filedialog
        cmdrs = [c.name for c in self.cards if c.is_commander]
        default_name = cmdrs[0] if cmdrs else "deck"
        safe_name = re.sub(r'[^\w\-. ]', '_', default_name)
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Deck", "*.json"), ("All", "*.*")],
            initialfile=f"{safe_name}.json",
            title="Save Deck")
        if not path:
            return
        import json
        deck_data = {
            "format": "faster_fishing_v1",
            "cards": []
        }
        for c in self.cards:
            deck_data["cards"].append({
                "name": c.name, "quantity": c.quantity, "category": c.category,
                "mana_cost": c.mana_cost, "cmc": c.cmc, "type_line": c.type_line,
                "image_uri": c.image_uri, "scryfall_id": c.scryfall_id,
                "oracle_text": c.oracle_text, "is_commander": c.is_commander,
                "layout": c.layout
            })
        with open(path, "w") as f:
            json.dump(deck_data, f, indent=2)
        self._sts(f"Deck saved: {path}")

    def _load_deck(self):
        """Load deck from JSON file."""
        from tkinter import filedialog
        import json
        path = filedialog.askopenfilename(
            filetypes=[("JSON Deck", "*.json"), ("All", "*.*")],
            title="Load Deck")
        if not path:
            return
        try:
            with open(path, "r") as f:
                deck_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Load Error", str(e)); return

        if deck_data.get("format") != "faster_fishing_v1":
            messagebox.showwarning("Unknown Format", "This file doesn't appear to be a Faster Fishing deck.")
            return

        self.cards.clear()
        for cd in deck_data.get("cards", []):
            c = Card(
                name=cd["name"], quantity=cd.get("quantity", 1),
                category=cd.get("category", "Other"),
                mana_cost=cd.get("mana_cost", ""), cmc=cd.get("cmc", 0),
                type_line=cd.get("type_line", ""),
                image_uri=cd.get("image_uri", ""),
                scryfall_id=cd.get("scryfall_id", ""),
                oracle_text=cd.get("oracle_text", ""),
                is_commander=cd.get("is_commander", False),
                layout=cd.get("layout", "normal")
            )
            self.cards.append(c)
        self._refresh_tree(); self._refresh_summary()
        self._sts(f"Loaded deck: {path} ({sum(c.quantity for c in self.cards)} cards)")

    def _open_deck_editor(self):
        """Open the full deck editor window with Scryfall search and EDHREC tabs."""
        DeckEditorWindow(self)


    # ---- SIMULATION TAB ----
    def _ui_sim(self, f):
        ctrl = ttk.LabelFrame(f, text="Simulation Settings", padding=10); ctrl.pack(fill=tk.X, padx=10, pady=10)
        # Row 1: Simulations, Hand size, buttons
        row1 = ttk.Frame(ctrl); row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Simulations:").pack(side=tk.LEFT)
        self.sim_n = tk.StringVar(value="10000"); ttk.Entry(row1, textvariable=self.sim_n, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(row1, text="Hand size:").pack(side=tk.LEFT, padx=(20,0))
        self.hand_sz = tk.StringVar(value="7"); ttk.Entry(row1, textvariable=self.hand_sz, width=5).pack(side=tk.LEFT, padx=5)
        self.sim_btn = ttk.Button(row1, text="Run Simulation", command=self._run_sim); self.sim_btn.pack(side=tk.RIGHT, padx=5)
        ttk.Button(row1, text="Draw Sample Hand", command=self._show_hand).pack(side=tk.RIGHT, padx=5)

        # Row 2: Mulligan
        row2 = ttk.Frame(ctrl); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Mulligan down to:").pack(side=tk.LEFT)
        self.min_mull = tk.StringVar(value="4"); ttk.Entry(row2, textvariable=self.min_mull, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="(Commander free mull at 7)").pack(side=tk.LEFT, padx=(0,20))

        # Row 3: Card Priority List
        row3 = ttk.Frame(ctrl); row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Card Priority List:").pack(side=tk.LEFT)
        self.tutor_targets_var = tk.StringVar(value="")
        self.tutor_targets_entry = ttk.Entry(row3, textvariable=self.tutor_targets_var, width=50)
        self.tutor_targets_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(row3, text="Pick...", command=self._pick_tutor_targets).pack(side=tk.LEFT, padx=2)

        # Row 4: Ideal Hand Configuration
        ideal_frame = ttk.LabelFrame(ctrl, text="Ideal Hand Configuration (probability tracked in results)", padding=5)
        ideal_frame.pack(fill=tk.X, pady=(5, 0))
        ideal_row = ttk.Frame(ideal_frame); ideal_row.pack(fill=tk.X)

        self.ideal_hand = {}  # category -> StringVar
        # Show spinners for common categories
        ideal_cats = ["Land", "Ramp", "Draw", "Removal", "Board Wipe", "Tutor", "Creature"]
        for cat in ideal_cats:
            cf = ttk.Frame(ideal_row)
            cf.pack(side=tk.LEFT, padx=4)
            ttk.Label(cf, text=f"{cat}:", font=("Segoe UI", 8)).pack(side=tk.LEFT)
            var = tk.StringVar(value="")
            self.ideal_hand[cat] = var
            sb = ttk.Spinbox(cf, from_=0, to=7, textvariable=var, width=3,
                            font=("Segoe UI", 9))
            sb.pack(side=tk.LEFT, padx=2)

        ttk.Label(ideal_row, text="(blank = any)", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(10, 0))

        self.sim_prog = ttk.Progressbar(ctrl, mode="determinate", length=500); self.sim_prog.pack(fill=tk.X, pady=5)
        rp = ttk.PanedWindow(f, orient=tk.HORIZONTAL); rp.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        lf = ttk.Frame(rp); rp.add(lf, weight=1)
        ttk.Label(lf, text="Results", style="S.TLabel").pack(anchor=tk.W)
        self.sim_text = tk.Text(lf, font=("Consolas",10), bg="#16213e", fg="#e0e0e0", state=tk.DISABLED, wrap=tk.WORD)
        self.sim_text.pack(fill=tk.BOTH, expand=True)
        for tag, color in [("header","#e94560"),("good","#2ECC71"),("warn","#F39C12"),("bad","#E74C3C"),("info","#00d2ff"),("mull","#BB86FC")]:
            kw = {"foreground": color}
            if tag == "header": kw["font"] = ("Consolas",11,"bold")
            self.sim_text.tag_configure(tag, **kw)
        rf = ttk.Frame(rp); rp.add(rf, weight=1)
        ttk.Label(rf, text="Land Distribution Chart", style="S.TLabel").pack(anchor=tk.W)
        self.chart = tk.Canvas(rf, bg="#16213e", highlightthickness=0); self.chart.pack(fill=tk.BOTH, expand=True)

    # ---- GOLDFISH TAB ----
    def _ui_gf(self, f):
        ctrl = ttk.LabelFrame(f, text="Goldfish Settings", padding=10); ctrl.pack(fill=tk.X, padx=10, pady=10)
        row = ttk.Frame(ctrl); row.pack(fill=tk.X)
        ttk.Label(row, text="Simulations:").pack(side=tk.LEFT)
        self.gf_n = tk.StringVar(value="1000"); ttk.Entry(row, textvariable=self.gf_n, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Turns:").pack(side=tk.LEFT, padx=(15,0))
        self.gf_t = tk.StringVar(value="10"); ttk.Entry(row, textvariable=self.gf_t, width=5).pack(side=tk.LEFT, padx=5)
        self.gf_btn = ttk.Button(row, text="Run Goldfish", command=self._run_gf); self.gf_btn.pack(side=tk.RIGHT)
        ttk.Button(row, text="Play Goldfish", command=self._show_goldfish_game).pack(side=tk.RIGHT, padx=5)
        self.gf_prog = ttk.Progressbar(ctrl, mode="determinate", length=500); self.gf_prog.pack(fill=tk.X, pady=5)
        self.gf_text = tk.Text(f, font=("Consolas",10), bg="#16213e", fg="#e0e0e0", state=tk.DISABLED, wrap=tk.WORD)
        self.gf_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.gf_text.tag_configure("header", foreground="#e94560", font=("Consolas",11,"bold"))
        self.gf_text.tag_configure("info", foreground="#00d2ff")
        self.gf_text.tag_configure("good", foreground="#2ECC71")
        self.gf_text.tag_configure("sub", foreground="#FFD700", font=("Consolas",10,"bold"))
        self.gf_text.tag_configure("warn", foreground="#FF6B6B")

    # ================================================================
    # ACTION METHODS
    # ================================================================
    def _sts(self, msg):
        self.status.set(msg); self.root.update_idletasks()

    def _imp_url(self):
        url = self.url_e.get().strip()
        if not url: messagebox.showwarning("No URL", "Enter a deck URL."); return
        self._sts("Fetching deck from URL..."); self.url_b.configure(state=tk.DISABLED)
        def go():
            txt = DeckParser.fetch_from_url(url)
            self.root.after(0, lambda: self._url_done(txt))
        threading.Thread(target=go, daemon=True).start()

    def _url_done(self, txt):
        self.url_b.configure(state=tk.NORMAL)
        if txt == "__MOXFIELD_BLOCKED__":
            self._sts("Moxfield blocked the request (Cloudflare).")
            messagebox.showinfo(
                "Moxfield Export Help",
                "Moxfield blocked all automated request attempts.\n\n"
                "For better Moxfield support, install cloudscraper:\n"
                "  pip install cloudscraper\n\n"
                "Or import your deck manually:\n"
                "1. Open your deck on Moxfield\n"
                "2. Click the '...' menu (top right)\n"
                "3. Select 'Export' > 'Export for MTGO'\n"
                "4. Copy the text and paste it in the Import Text box\n\n"
                "Tip: Put your commander on the first line\n"
                "with #CMDR at the end, e.g.:\n"
                "1 Atraxa, Praetors' Voice #CMDR")
        elif txt:
            self.dtxt.delete("1.0", tk.END); self.dtxt.insert("1.0", txt)
            self._sts("Deck fetched! Auto-importing...")
            # Auto-import the fetched text directly
            self._imp_txt()
        else:
            self._sts("Failed to fetch deck.")
            messagebox.showerror("Import Failed",
                "Could not fetch that URL.\n\n"
                "For Moxfield: use Export > Export for MTGO,\n"
                "then paste the text here.\n\n"
                "Tip: Mark your commander with #CMDR:\n"
                "1 Atraxa, Praetors' Voice #CMDR")

    def _imp_txt(self):
        text = self.dtxt.get("1.0", tk.END).strip()
        if not text: messagebox.showwarning("Empty", "Paste a decklist first."); return
        parsed = DeckParser.parse_text(text)
        if not parsed: messagebox.showwarning("Error", "Could not parse any cards."); return
        self._sts(f"Loading {len(parsed)} cards from Scryfall..."); self.ip["value"] = 0
        def go():
            # Build identifiers — for split/room cards, users might enter just one face
            ids = [{"name": n} for _, n, _ in parsed]
            qty_map = {}; cmdr_set = set()
            for q, n, is_cmdr in parsed:
                qty_map[n.lower()] = qty_map.get(n.lower(), 0) + q
                if is_cmdr: cmdr_set.add(n.lower())
            self.root.after(0, lambda: self.ip_lbl.configure(text="Fetching card data..."))
            sc_cards = self.scry.fetch_collection(ids)

            # Build a lookup: map each face name (lowercase) -> full Scryfall name
            # This prevents MDFCs from appearing as two separate cards
            found_full = set()   # full scryfall names already processed
            found_faces = set()  # individual face names already processed
            cards = []

            for i, sc in enumerate(sc_cards):
                full_name = sc.get("name", "")  # e.g. "Shatterskull Smashing // Shatterskull, the Hammer Pass"
                fn_lower = full_name.lower()
                if fn_lower in found_full:
                    continue
                found_full.add(fn_lower)

                # Collect all face names for this card (for matching against user input)
                face_names = [fn_lower]
                if " // " in full_name:
                    for part in full_name.split(" // "):
                        face_names.append(part.strip().lower())
                found_faces.update(face_names)

                layout = sc.get("layout", "normal")
                has_faces = "card_faces" in sc and sc["card_faces"]

                # --- IMAGE: prefer root image_uris, fallback to first face ---
                img = ""
                if "image_uris" in sc:
                    img = sc["image_uris"].get("normal", sc["image_uris"].get("small", ""))
                elif has_faces:
                    face = sc["card_faces"][0]
                    if "image_uris" in face:
                        img = face["image_uris"].get("normal", face["image_uris"].get("small", ""))

                # --- CMC: handle rooms, split cards, and other multi-face layouts ---
                cmc = sc.get("cmc", 0)
                if has_faces and cmc == 0:
                    # Root cmc missing — calculate from face mana costs
                    cmc = self._calc_face_cmc_total(sc["card_faces"])

                # --- ORACLE TEXT: combine from faces if not at root ---
                oracle = sc.get("oracle_text", "")
                if not oracle and has_faces:
                    oracle = " // ".join(f.get("oracle_text", "") for f in sc["card_faces"])

                # --- TYPE LINE: use root (already combined by Scryfall) ---
                type_line = sc.get("type_line", "")

                # --- QUANTITY: match against any face name from user input ---
                qty = 0
                is_cmdr = False
                for fn in face_names:
                    if fn in qty_map:
                        qty = max(qty, qty_map[fn])
                    if fn in cmdr_set:
                        is_cmdr = True
                if qty == 0:
                    qty = 1  # fallback

                c = Card(name=full_name, quantity=qty, mana_cost=sc.get("mana_cost",""),
                    cmc=cmc, type_line=type_line, image_uri=img,
                    scryfall_id=sc.get("id",""), oracle_text=oracle,
                    is_commander=is_cmdr, layout=layout)
                cards.append(c)
                p = (i+1)/max(len(sc_cards),1)*50
                self.root.after(0, lambda v=p: self.ip.configure(value=v))

            # Add cards not found by Scryfall collection endpoint
            # Try fuzzy name search for each missing card (handles partial room/split names)
            for q, n, is_cmdr in parsed:
                if n.lower() not in found_faces:
                    self.root.after(0, lambda nm=n: self.ip_lbl.configure(
                        text=f"Looking up: {nm}..."))
                    sc = self.scry.fetch_by_name(n)
                    if sc:
                        full_name = sc.get("name", n)
                        fn_lower = full_name.lower()
                        if fn_lower in found_full:
                            continue  # already have this card
                        found_full.add(fn_lower)
                        layout = sc.get("layout", "normal")
                        has_faces = "card_faces" in sc and sc["card_faces"]
                        img = ""
                        if "image_uris" in sc:
                            img = sc["image_uris"].get("normal", sc["image_uris"].get("small", ""))
                        elif has_faces:
                            face = sc["card_faces"][0]
                            if "image_uris" in face:
                                img = face["image_uris"].get("normal", face["image_uris"].get("small", ""))
                        cmc = sc.get("cmc", 0)
                        if has_faces and cmc == 0:
                            cmc = self._calc_face_cmc_total(sc["card_faces"])
                        oracle = sc.get("oracle_text", "")
                        if not oracle and has_faces:
                            oracle = " // ".join(f.get("oracle_text", "") for f in sc["card_faces"])
                        type_line = sc.get("type_line", "")
                        cards.append(Card(name=full_name, quantity=q,
                            mana_cost=sc.get("mana_cost",""), cmc=cmc,
                            type_line=type_line, image_uri=img,
                            scryfall_id=sc.get("id",""), oracle_text=oracle,
                            is_commander=is_cmdr, layout=layout))
                    else:
                        cards.append(Card(name=n, quantity=q, is_commander=is_cmdr))

            def cat_progress(pct, msg=""):
                v = 50 + pct*50
                self.root.after(0, lambda: self.ip.configure(value=v))
                if msg: self.root.after(0, lambda: self.ip_lbl.configure(text=msg))
            CardCategorizer.categorize_all(cards, self.scry, cat_progress)
            self.root.after(0, lambda: self._import_done(cards))
        threading.Thread(target=go, daemon=True).start()

    @staticmethod
    def _calc_face_cmc_total(card_faces):
        """Sum CMC from card faces by parsing mana cost strings. For Rooms."""
        total = 0
        for face in card_faces:
            mc = face.get("mana_cost", "")
            if not mc:
                continue
            # Count mana symbols: {1}, {2}, {W}, {U}, {B}, {R}, {G}, etc.
            symbols = re.findall(r'\{([^}]+)\}', mc)
            for sym in symbols:
                if sym.isdigit():
                    total += int(sym)
                elif sym in ("X", "Y", "Z"):
                    pass  # variable costs = 0
                elif "/" in sym:
                    total += 1  # hybrid = 1
                else:
                    total += 1  # colored pip = 1
        return total

    def _import_done(self, cards):
        self.cards = cards; self.ip_lbl.configure(text="")
        self._refresh_tree(); self._refresh_summary()
        total = sum(c.quantity for c in self.cards)
        self._sts(f"Imported {len(self.cards)} unique cards ({total} total) - preloading images...")
        self.nb.select(1)
        # Preload all card images in background
        threading.Thread(target=self._preload_images, daemon=True).start()

    def _preload_images(self):
        """Download all card images in the background so they're ready for sample hands."""
        total = len(self.cards)
        for i, card in enumerate(self.cards):
            if card.name in self.pil_cache:
                continue  # already loaded
            img = self.scry.fetch_image(card.image_uri, card.name)
            if img:
                self.pil_cache[card.name] = img
            if (i + 1) % 5 == 0 or i == total - 1:
                loaded = len(self.pil_cache)
                self.root.after(0, lambda l=loaded: self._sts(
                    f"Preloading images... {l}/{total}"))
        self.root.after(0, lambda: self._sts(
            f"Ready! {len(self.pil_cache)}/{total} images loaded."))

    def _clr(self):
        self.cards.clear(); self.imgs.clear(); self.pil_cache.clear()
        self._refresh_tree(); self._refresh_summary()
        self.dtxt.delete("1.0", tk.END); self._sts("Deck cleared.")

    def _recat(self):
        if not self.cards: messagebox.showwarning("No Deck", "Import first!"); return
        self._sts("Re-categorizing via Scryfall tags...")
        def go():
            CardCategorizer.categorize_all(self.cards, self.scry)
            self.root.after(0, self._refresh_tree); self.root.after(0, self._refresh_summary)
            self.root.after(0, lambda: self._sts("Re-categorization complete!"))
        threading.Thread(target=go, daemon=True).start()

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        # Split commanders from deck
        commanders = [(i, c) for i, c in enumerate(self.cards) if c.is_commander]
        deck_cards = [(i, c) for i, c in enumerate(self.cards) if not c.is_commander]

        # Sort deck cards by current sort column
        col = self._sort_col; rev = self._sort_rev
        def sort_key(item):
            _, c = item
            if col == "cmdr": return (0 if c.is_commander else 1,)
            elif col == "qty": return (c.quantity,)
            elif col == "name": return (c.name.lower(),)
            elif col == "category": return (c.category.lower(),)
            elif col == "cmc": return (c.cmc,)
            elif col == "type": return (c.type_line.lower(),)
            return (c.name.lower(),)
        deck_cards.sort(key=sort_key, reverse=rev)

        # Commanders always at top
        for orig_idx, c in commanders:
            self.tree.insert("", tk.END, iid=str(orig_idx),
                values=("⭐", c.quantity, c.name, c.category, int(c.cmc), c.type_line),
                tags=("commander",))
        # Separator if there are commanders
        if commanders:
            self.tree.insert("", tk.END, iid="__sep__",
                values=("", "", "─── Deck (" + str(sum(c.quantity for _, c in deck_cards)) + " cards) ───", "", "", ""),
                tags=("separator",))
        # Non-commander cards
        for orig_idx, c in deck_cards:
            self.tree.insert("", tk.END, iid=str(orig_idx),
                values=("", c.quantity, c.name, c.category, int(c.cmc), c.type_line))

        # Style the commander rows and separator
        self.tree.tag_configure("commander", foreground="#FFD700")
        self.tree.tag_configure("separator", foreground="#555555")

        n_cmdrs = sum(1 for c in self.cards if c.is_commander)
        deck_size = sum(c.quantity for c in self.cards if not c.is_commander)
        self.dcnt.set(f"{deck_size} in deck + {n_cmdrs} commander(s) ({len(self.cards)} unique)")
        self._update_cmdr_info()
        self._update_cmdr_display()
        # Update heading arrows
        for c_name in ("cmdr","qty","name","category","cmc","type"):
            display = {"cmdr":"*","qty":"Qty","name":"Name","category":"Category",
                       "cmc":"CMC","type":"Type"}[c_name]
            arrow = ""
            if c_name == col:
                arrow = " ▼" if rev else " ▲"
            self.tree.heading(c_name, text=display + arrow)
        # Update grid view if visible
        if hasattr(self, '_deck_view_mode') and self._deck_view_mode.get() == "grid":
            self._render_deck_grid()

    def _sort_tree(self, col):
        """Sort the treeview by clicked column. Click again to reverse."""
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        self._refresh_tree()

    def _remove_card(self):
        """Remove the selected card from the deck (not the original decklist)."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a card in the list first.")
            return
        iid = sel[0]
        if iid == "__sep__":
            return
        try:
            idx = int(iid)
        except ValueError:
            return
        if 0 <= idx < len(self.cards):
            removed = self.cards[idx]
            # Confirm removal
            if removed.is_commander:
                msg = f"Remove commander '{removed.name}'?"
            else:
                msg = f"Remove '{removed.name}' (x{removed.quantity})?"
            if messagebox.askyesno("Remove Card", msg):
                self.cards.pop(idx)
                self.sel_idx = None
                self._refresh_tree(); self._refresh_summary()
                self._sts(f"Removed: {removed.name}")

    # ---- ADD CARD (Scryfall Search Popup) ----
    def _open_add_card_dialog(self):
        """Open a popup dialog for searching and adding cards from Scryfall."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Add Card from Scryfall")
        dlg.geometry("500x420")
        dlg.configure(bg="#1a1a2e")
        dlg.transient(self.root)
        dlg.grab_set()

        # Search
        ttk.Label(dlg, text="Search Scryfall", style="S.TLabel").pack(anchor=tk.W, padx=15, pady=(10,3))
        search_frame = ttk.Frame(dlg); search_frame.pack(fill=tk.X, padx=15)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, font=("Segoe UI",11))
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        search_entry.focus_set()

        # Results listbox
        ttk.Label(dlg, text="Suggestions (click to select):", style="TLabel").pack(anchor=tk.W, padx=15, pady=(8,2))
        lb_frame = ttk.Frame(dlg); lb_frame.pack(fill=tk.BOTH, expand=True, padx=15)
        listbox = tk.Listbox(lb_frame, font=("Segoe UI",10), bg="#16213e", fg="#e0e0e0",
                             selectbackground="#0f3460", height=8, exportselection=False)
        lb_sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=lb_sb.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_sb.pack(side=tk.LEFT, fill=tk.Y)

        # Qty + buttons
        bottom = ttk.Frame(dlg); bottom.pack(fill=tk.X, padx=15, pady=10)
        ttk.Label(bottom, text="Qty:").pack(side=tk.LEFT)
        qty_var = tk.StringVar(value="1")
        ttk.Spinbox(bottom, from_=1, to=99, width=4, textvariable=qty_var,
                     font=("Segoe UI",10)).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom, text="Add to Deck", command=lambda: do_add()).pack(side=tk.LEFT, padx=5)
        status_var = tk.StringVar(value="")
        tk.Label(bottom, textvariable=status_var, bg="#1a1a2e", fg="#2ECC71",
                 font=("Segoe UI",9)).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom, text="Close", command=dlg.destroy).pack(side=tk.RIGHT)

        # Debounce state
        after_id = [None]

        def on_key(event):
            if after_id[0]:
                dlg.after_cancel(after_id[0])
            after_id[0] = dlg.after(300, do_autocomplete)

        def do_autocomplete():
            query = search_var.get().strip()
            if len(query) < 2:
                listbox.delete(0, tk.END)
                return
            def go():
                suggestions = self.scry.autocomplete(query)
                dlg.after(0, lambda: show_results(suggestions))
            threading.Thread(target=go, daemon=True).start()

        def show_results(suggestions):
            listbox.delete(0, tk.END)
            for name in suggestions[:15]:
                listbox.insert(tk.END, name)

        def on_listbox_select(event):
            sel = listbox.curselection()
            if sel:
                search_var.set(listbox.get(sel[0]))

        def do_add():
            name = search_var.get().strip()
            if not name:
                status_var.set("Type a card name first")
                return
            try: qty = int(qty_var.get())
            except ValueError: qty = 1
            qty = max(1, qty)

            # Check if already in deck
            for c in self.cards:
                if c.name.lower() == name.lower() or (
                    " // " in c.name and name.lower() in c.name.lower()):
                    c.quantity += qty
                    self._refresh_tree(); self._refresh_summary()
                    status_var.set(f"+{qty} {c.name} (now x{c.quantity})")
                    search_var.set(""); listbox.delete(0, tk.END)
                    return

            status_var.set("Looking up...")
            def go():
                sc = self.scry.fetch_by_name(name)
                dlg.after(0, lambda: finish_add(sc, name, qty))
            threading.Thread(target=go, daemon=True).start()

        def finish_add(sc, search_name, qty):
            if not sc:
                status_var.set(f"Not found: {search_name}")
                return

            full_name = sc.get("name", search_name)
            layout = sc.get("layout", "normal")
            has_faces = "card_faces" in sc and sc["card_faces"]

            # Check duplicate again with full Scryfall name
            for c in self.cards:
                if c.name.lower() == full_name.lower():
                    c.quantity += qty
                    self._refresh_tree(); self._refresh_summary()
                    status_var.set(f"+{qty} {c.name} (now x{c.quantity})")
                    search_var.set(""); listbox.delete(0, tk.END)
                    return

            # Build Card object
            img = ""
            if "image_uris" in sc:
                img = sc["image_uris"].get("normal", sc["image_uris"].get("small", ""))
            elif has_faces:
                face = sc["card_faces"][0]
                if "image_uris" in face:
                    img = face["image_uris"].get("normal", face["image_uris"].get("small", ""))

            cmc = sc.get("cmc", 0)
            if has_faces and cmc == 0:
                cmc = self._calc_face_cmc_total(sc["card_faces"])

            oracle = sc.get("oracle_text", "")
            if not oracle and has_faces:
                oracle = " // ".join(f.get("oracle_text", "") for f in sc["card_faces"])

            card = Card(name=full_name, quantity=qty,
                        mana_cost=sc.get("mana_cost", ""), cmc=cmc,
                        type_line=sc.get("type_line", ""), image_uri=img,
                        scryfall_id=sc.get("id", ""), oracle_text=oracle,
                        layout=layout)

            # Auto-categorize
            tl = card.type_line.lower()
            if " // " in tl:
                faces = [f.strip() for f in tl.split(" // ")]
                if any("land" in f for f in faces):
                    card.category = "Land"
            elif "land" in tl and "creature" not in tl:
                card.category = "Land"

            if card.category == "Other":
                for cat, tag in CATEGORY_TAGS.items():
                    matched = self.scry.search_names_with_tag(tag, [full_name])
                    if full_name.lower() in matched:
                        card.category = cat
                        break
                else:
                    if "creature" in tl:
                        card.category = "Creature"

            self.cards.append(card)
            self._refresh_tree(); self._refresh_summary()
            status_var.set(f"Added: {full_name} x{qty}")
            search_var.set(""); listbox.delete(0, tk.END)
            self._sts(f"Added {full_name} x{qty} to deck")

            # Preload image in background
            if full_name not in self.pil_cache:
                def load_img():
                    pil_img = self.scry.fetch_image(img, full_name)
                    if pil_img:
                        self.pil_cache[full_name] = pil_img
                threading.Thread(target=load_img, daemon=True).start()

        search_entry.bind("<KeyRelease>", on_key)
        search_entry.bind("<Return>", lambda e: do_add())
        listbox.bind("<<ListboxSelect>>", on_listbox_select)
        listbox.bind("<Double-Button-1>", lambda e: do_add())

    def _update_cmdr_info(self):
        cmdrs = [c.name for c in self.cards if c.is_commander]
        if cmdrs:
            self.cmdr_info.set("Commander: " + ", ".join(cmdrs))
        else:
            self.cmdr_info.set("No commander set (tip: select a card and click 'Set as Commander')")

    def _toggle_commander(self):
        if self.sel_idx is None: return
        card = self.cards[self.sel_idx]
        card.is_commander = not card.is_commander
        self._refresh_tree(); self._refresh_summary()
        idx = str(self.sel_idx); self.tree.selection_set(idx); self.tree.see(idx)

    def _clear_commanders(self):
        for c in self.cards: c.is_commander = False
        self._refresh_tree(); self._refresh_summary()

    def _get_deck_cards(self):
        """Return only non-commander cards for simulation."""
        return [c for c in self.cards if not c.is_commander]

    def _refresh_summary(self):
        self.summary.configure(state=tk.NORMAL); self.summary.delete("1.0", tk.END)
        if not self.cards:
            self.summary.insert(tk.END, "No cards loaded."); self.summary.configure(state=tk.DISABLED); return
        cmdrs = [c for c in self.cards if c.is_commander]
        deck = [c for c in self.cards if not c.is_commander]
        total = sum(c.quantity for c in deck)
        if cmdrs:
            self.summary.insert(tk.END, "Commander(s):\n")
            for c in cmdrs:
                self.summary.insert(tk.END, f"  {c.name} (CMC {int(c.cmc)})\n")
            self.summary.insert(tk.END, "\n")
        self.summary.insert(tk.END, f"Deck: {total} cards\n")
        cc = Counter()
        for c in deck: cc[c.category] += c.quantity
        self.summary.insert(tk.END, "-"*40 + "\n")
        for cat in ALL_CATEGORIES:
            n = cc.get(cat, 0); pct = n/total*100 if total else 0
            self.summary.insert(tk.END, f"{cat:12s} {n:3d} ({pct:5.1f}%) {'#'*int(pct/2.5)}\n")
        self.summary.insert(tk.END, "-"*40 + "\n\nMana Curve (nonland):\n")
        cmc_c = Counter()
        for c in deck:
            if c.category != "Land": cmc_c[min(int(c.cmc), 7)] += c.quantity
        for cmc in range(8):
            lb = f"{cmc}" if cmc < 7 else "7+"
            self.summary.insert(tk.END, f"  CMC {lb}: {cmc_c.get(cmc,0):3d} {'#'*cmc_c.get(cmc,0)}\n")
        self.summary.configure(state=tk.DISABLED)

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        iid = sel[0]
        if iid == "__sep__":
            return  # ignore separator click
        try:
            idx = int(iid)
        except ValueError:
            return
        if idx < 0 or idx >= len(self.cards):
            return
        self.sel_idx = idx; card = self.cards[idx]; self.cat_var.set(card.category)
        if card.name in self.pil_cache:
            # Create preview-sized photo from cached PIL image
            preview = self.pil_cache[card.name].resize((250, 349), Image.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
            self.imgs[card.name] = photo  # prevent GC
            self.img_label.configure(image=photo, text="")
        else:
            self.img_label.configure(image="", text="Loading image...")
            threading.Thread(target=self._load_img, args=(card,), daemon=True).start()

    def _load_img(self, card):
        img = self.scry.fetch_image(card.image_uri, card.name)
        if img:
            self.pil_cache[card.name] = img
            preview = img.resize((250, 349), Image.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
            self.imgs[card.name] = photo
            self.root.after(0, lambda: self.img_label.configure(image=photo, text=""))
        else:
            self.root.after(0, lambda: self.img_label.configure(text=f"No image: {card.name}"))

    def _set_cat(self):
        if self.sel_idx is None: return
        self.cards[self.sel_idx].category = self.cat_var.get()
        self._refresh_tree(); self._refresh_summary()
        idx = str(self.sel_idx); self.tree.selection_set(idx); self.tree.see(idx)

    def _add_category(self):
        """Prompt user to create a new custom category with a hex color picker."""
        import math

        palette = [
            ["#FF0000","#FF4500","#FF8C00","#FFD700","#ADFF2F","#32CD32","#00CED1","#1E90FF","#6A5ACD","#9400D3","#FF1493","#FF69B4"],
            ["#DC143C","#E74C3C","#F39C12","#DAA520","#2ECC71","#2E8B57","#4169E1","#0f3460","#8B008B","#9370DB","#C71585","#DB7093"],
            ["#FFB6C1","#FFA07A","#FFE4B5","#F0E68C","#98FB98","#AFEEEE","#87CEEB","#B0C4DE","#DDA0DD","#D8BFD8","#FFE4E1","#FFDAB9"],
            ["#8B0000","#A0522D","#8B7355","#556B2F","#006400","#008080","#191970","#2C2C54","#4B0082","#483D8B","#800020","#708090"],
            ["#FFFFFF","#D3D3D3","#A9A9A9","#808080","#696969","#505050","#383838","#2D2D2D","#1C1C1C","#111111","#F5F5DC","#C0C0C0"],
        ]

        cols = len(palette[0])
        hex_r = 14
        hex_w = math.sqrt(3) * hex_r       # pointy-top width
        row_h = 2 * hex_r * 0.75           # vertical spacing between row centers
        # Canvas size: grid centered with padding
        grid_w = cols * hex_w + hex_w * 0.5  # extra half for odd-row offset
        canvas_w = int(grid_w + 24)          # 12px padding each side
        canvas_h = int(len(palette) * row_h + hex_r + 16)
        dlg_w = canvas_w + 40                # dialog padding
        dlg_h = canvas_h + 230               # room for name, preview, button

        dlg = tk.Toplevel(self.root); dlg.title("New Category")
        dlg.geometry(f"{dlg_w}x{dlg_h}")
        dlg.configure(bg="#1a1a2e"); dlg.transient(self.root); dlg.grab_set()

        ttk.Label(dlg, text="Category Name:").pack(padx=15, pady=(12,3), anchor=tk.W)
        name_var = tk.StringVar()
        ttk.Entry(dlg, textvariable=name_var, font=("Segoe UI",11)).pack(fill=tk.X, padx=15)

        ttk.Label(dlg, text="Pick a Color:").pack(padx=15, pady=(10,3), anchor=tk.W)

        color_var = tk.StringVar(value="#9370DB")

        preview_frame = tk.Frame(dlg, bg="#1a1a2e"); preview_frame.pack(fill=tk.X, padx=15, pady=(0,5))
        swatch = tk.Canvas(preview_frame, width=36, height=36, bg="#9370DB", highlightthickness=2,
                           highlightbackground="#555")
        swatch.pack(side=tk.LEFT)
        hex_label = tk.Label(preview_frame, text="#9370DB", bg="#1a1a2e", fg="#e0e0e0",
                             font=("Consolas",12,"bold"))
        hex_label.pack(side=tk.LEFT, padx=10)

        canvas = tk.Canvas(dlg, width=canvas_w, height=canvas_h, bg="#0d1117",
                           highlightthickness=1, highlightbackground="#30363d")
        canvas.pack(padx=15, pady=5)

        def hex_points(cx, cy, r):
            pts = []
            for i in range(6):
                angle = math.radians(60 * i - 30)
                pts.append(cx + r * math.cos(angle))
                pts.append(cy + r * math.sin(angle))
            return pts

        def on_hex_click(color):
            color_var.set(color)
            swatch.configure(bg=color)
            hex_label.configure(text=color)

        # Center the grid horizontally in the canvas
        pad_x = (canvas_w - cols * hex_w) / 2 + hex_w / 2

        for row_idx, row_colors in enumerate(palette):
            x_shift = hex_w / 2 if row_idx % 2 == 1 else 0
            cy = 8 + hex_r + row_idx * row_h
            for col_idx, color in enumerate(row_colors):
                cx = pad_x + col_idx * hex_w + x_shift
                pts = hex_points(cx, cy, hex_r - 1)  # slight inset for gaps
                hex_id = canvas.create_polygon(pts, fill=color, outline="#333", width=1,
                                                activefill=color, activeoutline="white", activewidth=2)
                canvas.tag_bind(hex_id, "<Button-1>", lambda e, c=color: on_hex_click(c))

        def do_add():
            n = name_var.get().strip()
            c = color_var.get().strip()
            if not n: return
            if n in ALL_CATEGORIES:
                messagebox.showwarning("Exists", f"'{n}' already exists.", parent=dlg); return
            ALL_CATEGORIES.append(n)
            CATEGORY_COLORS[n] = c if c.startswith("#") else "#9370DB"
            idx = len(ALL_CATEGORIES) - 1
            rb = tk.Radiobutton(self.cat_frame, text=n, variable=self.cat_var, value=n,
                command=self._set_cat, bg=self.bg, fg=CATEGORY_COLORS[n], selectcolor=self.bg,
                activebackground=self.bg, activeforeground=CATEGORY_COLORS[n],
                font=("Segoe UI",10,"bold"), indicatoron=True)
            rb.grid(row=(idx)//3, column=(idx)%3, sticky=tk.W, padx=5, pady=2)
            self._refresh_summary()
            dlg.destroy()
            self._sts(f"Added category: {n}")
        ttk.Button(dlg, text="Add Category", command=do_add).pack(pady=10)

    def _show_hand(self):
        if not self.cards: messagebox.showwarning("No Deck", "Import first!"); return
        deck_cards = self._get_deck_cards()
        if not deck_cards: messagebox.showwarning("Empty Deck", "All cards are set as commander!"); return
        SampleHandWindow(self.root, deck_cards, self.scry, self.pil_cache)

    def _show_goldfish_game(self):
        if not self.cards: messagebox.showwarning("No Deck", "Import first!"); return
        deck_cards = self._get_deck_cards()
        if not deck_cards: messagebox.showwarning("Empty Deck", "All cards are set as commander!"); return
        cmdrs = [c for c in self.cards if c.is_commander]
        GoldfishGameWindow(self.root, deck_cards, cmdrs, self.scry, self.pil_cache)

    # ---- SIMULATION ----
    def _get_land_range(self):
        """Auto-calculate keepable land range from commander CMC."""
        cmdrs = [c for c in self.cards if c.is_commander]
        cmdr_cmc = max((int(c.cmc) for c in cmdrs), default=0)
        # Individual commander info: {name: cmc}
        cmdr_info = {c.name: int(c.cmc) for c in cmdrs}

        # Auto from commander CMC
        if cmdr_cmc <= 3:
            lmin, lmax = 1, 3
        elif cmdr_cmc <= 6:
            lmin, lmax = 3, 5
        else:
            lmin, lmax = 3, 5
        return lmin, lmax, cmdr_cmc, cmdr_info

    def _pick_tutor_targets(self):
        """Open a visual dialog to pick cards from deck to track."""
        if not self.cards:
            messagebox.showwarning("No Deck", "Import a deck first!"); return
        pick_win = tk.Toplevel(self.root)
        pick_win.title("Card Priority List")
        pick_win.geometry("1050x750")
        pick_win.configure(bg="#1a1a2e")

        top = ttk.Frame(pick_win); top.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(top, text="Select cards to track avg turn seen/tutored:",
                  font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        ttk.Button(top, text="Apply", command=lambda: apply_and_close()).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Clear All", command=lambda: clear_all()).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Select All", command=lambda: select_all()).pack(side=tk.RIGHT, padx=5)

        # Scrollable card grid
        canvas = tk.Canvas(pick_win, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(pick_win, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg="#1a1a2e")
        canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Mousewheel scrolling
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        inner.bind("<MouseWheel>", _on_mousewheel)

        # Pre-parse current targets
        current = {t.strip() for t in self.tutor_targets_var.get().split(",") if t.strip()}
        check_vars = {}
        card_widgets = []
        _img_refs = []  # GC protection

        THUMB_W, THUMB_H = 146, 204
        COLS = 5

        sorted_cards = sorted(
            [c for c in self.cards if not c.is_commander],
            key=lambda x: x.name.lower())

        for idx, card in enumerate(sorted_cards):
            row_n = idx // COLS
            col_n = idx % COLS

            cell = tk.Frame(inner, bg="#1a1a2e", padx=4, pady=4)
            cell.grid(row=row_n, column=col_n, padx=3, pady=3, sticky="n")
            cell.bind("<MouseWheel>", _on_mousewheel)

            # Card image thumbnail
            img_label = tk.Label(cell, bg="#2a2a4a", width=THUMB_W, height=THUMB_H)
            img_label.pack()
            img_label.bind("<MouseWheel>", _on_mousewheel)

            # Load image async
            if card.name in self.pil_cache:
                try:
                    thumb = self.pil_cache[card.name].resize((THUMB_W, THUMB_H), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(thumb)
                    _img_refs.append(photo)
                    img_label.configure(image=photo, width=THUMB_W, height=THUMB_H)
                except Exception:
                    img_label.configure(text=card.name[:20], fg="#888")
            else:
                img_label.configure(text=card.name[:20], fg="#888")
                # Load async
                def load_img(c=card, lbl=img_label):
                    img = self.scry.fetch_image(c.image_uri, c.name)
                    if img:
                        self.pil_cache[c.name] = img
                        try:
                            thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
                            photo = ImageTk.PhotoImage(thumb)
                            _img_refs.append(photo)
                            pick_win.after(0, lambda p=photo, l=lbl: l.configure(image=p, width=THUMB_W, height=THUMB_H))
                        except Exception:
                            pass
                threading.Thread(target=load_img, daemon=True).start()

            # Checkbox with card name
            var = tk.BooleanVar(value=card.name in current)
            check_vars[card.name] = var

            cb_frame = tk.Frame(cell, bg="#1a1a2e")
            cb_frame.pack(fill=tk.X)
            cb = tk.Checkbutton(cb_frame, variable=var, bg="#1a1a2e",
                               activebackground="#1a1a2e", selectcolor="#2a2a4a")
            cb.pack(side=tk.LEFT)
            tk.Label(cb_frame, text=card.name[:22], bg="#1a1a2e", fg="#e0e0e0",
                    font=("Segoe UI", 8), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X)
            cb_frame.bind("<MouseWheel>", _on_mousewheel)

            # Toggle on image click too
            def toggle(v=var):
                v.set(not v.get())
            img_label.bind("<Button-1>", lambda e, v=var: toggle(v))

            # Highlight if selected
            def update_highlight(cell=cell, var=var):
                if var.get():
                    cell.configure(bg="#2E8B57", highlightbackground="#2ECC71", highlightthickness=2)
                else:
                    cell.configure(bg="#1a1a2e", highlightbackground="#1a1a2e", highlightthickness=0)
            var.trace_add("write", lambda *a, c=cell, v=var: update_highlight(c, v))
            update_highlight()

            card_widgets.append((card, var, cell))

        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Store refs on the window to prevent GC
        pick_win._img_refs = _img_refs

        def select_all():
            for _, var, _ in card_widgets:
                var.set(True)

        def clear_all():
            for _, var, _ in card_widgets:
                var.set(False)

        def apply_and_close():
            selected = [name for name, var in check_vars.items() if var.get()]
            self.tutor_targets_var.set(", ".join(selected))
            pick_win.destroy()

    def _run_sim(self):
        if not self.cards: messagebox.showwarning("No Deck", "Import first!"); return
        deck_cards = self._get_deck_cards()
        if not deck_cards: messagebox.showwarning("Empty Deck", "All cards are set as commander!"); return
        try: ns = int(self.sim_n.get()); hs = int(self.hand_sz.get())
        except ValueError: messagebox.showerror("Error", "Enter valid numbers."); return
        try: min_mull = int(self.min_mull.get())
        except ValueError: min_mull = 4
        min_mull = max(1, min(hs, min_mull))
        land_min, land_max, cmdr_cmc, cmdr_info = self._get_land_range()

        # Parse tutor target list
        tutor_targets = None
        tt_str = self.tutor_targets_var.get().strip()
        if tt_str:
            tutor_targets = [t.strip() for t in tt_str.split(",") if t.strip()]

        # Parse ideal hand configuration
        ideal_hand = {}
        for cat, var in self.ideal_hand.items():
            val = var.get().strip()
            if val:
                try:
                    ideal_hand[cat] = int(val)
                except ValueError:
                    pass

        self.sim_btn.configure(state=tk.DISABLED); self.sim_prog["value"] = 0
        self._sts(f"Running {ns:,} simulations (hands {hs} down to {min_mull})...")
        def pcb(p): self.root.after(0, lambda: self.sim_prog.configure(value=p*100))
        def go():
            results = SimEngine.sim_hands(deck_cards, ns, hs, pcb,
                                          min_mull=min_mull, land_min=land_min,
                                          land_max=land_max, commander_cmc=cmdr_cmc,
                                          commander_cmcs=cmdr_info,
                                          tutor_targets=tutor_targets,
                                          ideal_hand=ideal_hand)
            self.root.after(0, lambda: self._show_sim(results, land_min, land_max, cmdr_cmc, ideal_hand, cmdr_info))
        threading.Thread(target=go, daemon=True).start()

    def _show_sim(self, results, land_min, land_max, cmdr_cmc, ideal_hand=None, cmdr_info=None):
        self.sim_btn.configure(state=tk.NORMAL); self._sts("Simulation complete!")
        t = self.sim_text; t.configure(state=tk.NORMAL); t.delete("1.0", tk.END)

        # Show settings
        t.insert(tk.END, "OPENING HAND SIMULATION\n", "header")
        cmdrs = [c for c in self.cards if c.is_commander]
        if cmdrs:
            cmdr_names = ", ".join(f"{c.name} (CMC {int(c.cmc)})" for c in cmdrs)
            t.insert(tk.END, f"Commander: {cmdr_names}\n", "info")
        t.insert(tk.END, f"Keepable: {land_min}-{land_max} lands + early play + draw/ramp for low-land hands\n")

        # Show ideal hand config if set
        if ideal_hand:
            parts = [f"{cat}: {n}" for cat, n in ideal_hand.items() if n > 0]
            t.insert(tk.END, f"Ideal hand: {', '.join(parts)}\n", "info")
        t.insert(tk.END, "\n")

        # Get the primary (7-card) result for the chart
        primary_sz = max(results.keys())
        primary = results[primary_sz]

        # Commander cast turn
        if cmdr_cmc > 0 and primary.avg_cmdr_turn > 0:
            t.insert(tk.END, "COMMANDER CAST TURN\n", "header")

            # Per-commander breakdown (for partners/dual commanders)
            if cmdr_info and len(cmdr_info) > 1 and hasattr(primary, 'per_cmdr_turns') and primary.per_cmdr_turns:
                for cname, ccmc in sorted(cmdr_info.items(), key=lambda x: x[1]):
                    avg_t = primary.per_cmdr_turns.get(cname, 0)
                    if avg_t > 0:
                        tg = "good" if avg_t <= ccmc else "warn" if avg_t <= ccmc + 2 else "bad"
                        t.insert(tk.END, f"  {cname} ", "info")
                        t.insert(tk.END, f"(CMC {ccmc})", "")
                        t.insert(tk.END, f": Turn {avg_t:.1f}\n", tg)
                # Both commanders
                if primary.avg_both_turn > 0:
                    total_cmc = sum(cmdr_info.values())
                    tg = "good" if primary.avg_both_turn <= total_cmc else "warn"
                    t.insert(tk.END, f"  Both commanders castable: ", "info")
                    t.insert(tk.END, f"Turn {primary.avg_both_turn:.1f}\n", tg)
                t.insert(tk.END, "\n")
            else:
                # Single commander
                t.insert(tk.END, f"  Avg earliest cast turn: ", "info")
                turn = primary.avg_cmdr_turn
                tg = "good" if turn <= cmdr_cmc else "warn" if turn <= cmdr_cmc + 2 else "bad"
                t.insert(tk.END, f"Turn {turn:.1f}\n", tg)

            qual = primary.hand_quality
            qual_tg = "good" if qual >= 30 else "warn" if qual >= 15 else "bad"
            t.insert(tk.END, f"  On-curve rate (cast by T{cmdr_cmc}): ", "info")
            t.insert(tk.END, f"{qual:.1f}%\n", qual_tg)
            t.insert(tk.END, f"  (factors: land drops, ramp, draw spells, tutors)\n\n")

        # Mulligan summary table — Commander mulligan rules
        if len(results) > 1:
            t.insert(tk.END, "MULLIGAN ANALYSIS (Commander rules)\n", "header")
            t.insert(tk.END, "  Draw 7 → free mull → draw 7 put 1 back → etc.\n\n", "info")
            t.insert(tk.END, f"  {'Hand':>12s}  {'Keepable':>10s}  {'Avg Lands':>10s}  {'Avg Ramp':>10s}", "info")
            if cmdr_cmc > 0:
                t.insert(tk.END, f"  {'Cmdr Turn':>10s}  {'On Curve':>9s}", "info")
            t.insert(tk.END, "\n")
            t.insert(tk.END, "  " + "-" * (55 + (24 if cmdr_cmc > 0 else 0)) + "\n")
            for sz in sorted(results.keys(), reverse=True):
                r = results[sz]
                if sz == primary_sz:
                    label = "Opening + Free"
                else:
                    put_back = primary_sz - sz
                    label = f"Mull to {sz} (-{put_back})"
                tag = "good" if r.keepable >= 80 else "warn" if r.keepable >= 60 else "bad"
                line = f"  {label:>12s}  {r.keepable:>9.1f}%  {r.cat_avgs.get('Land',0):>10.2f}  {r.cat_avgs.get('Ramp',0):>10.2f}"
                if cmdr_cmc > 0:
                    line += f"  {r.avg_cmdr_turn:>10.1f}  {r.hand_quality:>8.1f}%"
                t.insert(tk.END, line + "\n", tag)
            t.insert(tk.END, "\n")

        # Ideal hand probability
        if ideal_hand and hasattr(r, 'ideal_or_better'):
            t.insert(tk.END, "\nIDEAL HAND PROBABILITY\n", "header")
            parts = [f"{cat}: {n}" for cat, n in ideal_hand.items() if n > 0]
            t.insert(tk.END, f"  Target (at least): {', '.join(parts)}\n\n", "info")

            # Show per-mulligan breakdown
            if len(results) > 1:
                t.insert(tk.END, f"  {'Hand':>14s}  {'Probability':>12s}\n", "info")
                t.insert(tk.END, "  " + "-" * 30 + "\n")
                for sz in sorted(results.keys(), reverse=True):
                    ri = results[sz]
                    ab = getattr(ri, 'ideal_or_better', 0)
                    if sz == primary_sz:
                        label = "Opening + Free"
                    else:
                        label = f"Mull to {sz} (-{primary_sz - sz})"
                    tg = "good" if ab >= 25 else "warn" if ab >= 10 else "bad"
                    t.insert(tk.END, f"  {label:>14s}  {ab:>11.1f}%\n", tg)
        t.insert(tk.END, "\n")

        # Tutor Priority Tracker results
        if primary.tutor_tracker:
            t.insert(tk.END, "\nCARD PRIORITY LIST (avg turn seen or tutored)\n", "header")
            t.insert(tk.END, f"  {'Card':<30s}  {'Avg Turn':>9s}  {'Found %':>8s}\n", "info")
            t.insert(tk.END, "  " + "-" * 52 + "\n")
            for tname, data in sorted(primary.tutor_tracker.items(),
                                       key=lambda x: x[1]["avg_turn"] if x[1]["avg_turn"] >= 0 else 99):
                avg = data["avg_turn"]
                pct = data["found_pct"]
                if avg >= 0:
                    tg = "good" if avg <= 5 else "warn" if avg <= 10 else "bad"
                    t.insert(tk.END, f"  {tname:<30s}  {'T'+f'{avg:.1f}':>9s}  {pct:>7.1f}%\n", tg)
                else:
                    t.insert(tk.END, f"  {tname:<30s}  {'never':>9s}  {pct:>7.1f}%\n", "bad")
            t.insert(tk.END, "\n  (includes natural draws + tutor searches within 20 turns)\n", "info")
            t.insert(tk.END, "  (draw engines on battlefield also contribute extra draws)\n", "info")

        t.configure(state=tk.DISABLED)
        self._draw_chart(r.land_dist, land_min, land_max)

    def _draw_chart(self, ld, land_min=2, land_max=5):
        c = self.chart; c.delete("all"); c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 80 or h < 80 or not ld: return
        m = 50; cw = w-2*m; ch = h-2*m; mx = max(ld.values(), default=1)
        n = len(ld); bw = min(cw/max(n,1)*0.7, 60); gap = (cw-bw*n)/max(n+1,1)
        c.create_line(m, m, m, h-m, fill="#555", width=2)
        c.create_line(m, h-m, w-m, h-m, fill="#555", width=2)
        for i, (lands, pct) in enumerate(sorted(ld.items())):
            x = m + gap*(i+1) + bw*i; bh = (pct/max(mx,1))*ch*0.9; yt = h-m-bh
            col = "#2ECC71" if land_min<=lands<=land_max else "#F39C12" if abs(lands-land_min)<=1 or abs(lands-land_max)<=1 else "#E74C3C"
            c.create_rectangle(x, yt, x+bw, h-m, fill=col, outline="")
            c.create_text(x+bw/2, h-m+15, text=str(lands), fill="#e0e0e0", font=("Segoe UI",9))
            c.create_text(x+bw/2, yt-12, text=f"{pct:.1f}%", fill="#e0e0e0", font=("Segoe UI",8))
        c.create_text(w/2, h-5, text="Lands in Hand", fill="#999", font=("Segoe UI",9))

    # ---- GOLDFISH ----
    def _run_gf(self):
        if not self.cards: messagebox.showwarning("No Deck", "Import first!"); return
        deck_cards = self._get_deck_cards()
        if not deck_cards: messagebox.showwarning("Empty Deck", "All cards are set as commander!"); return
        try: ns = int(self.gf_n.get()); nt = int(self.gf_t.get())
        except ValueError: messagebox.showerror("Error", "Enter valid numbers."); return
        self.gf_btn.configure(state=tk.DISABLED); self.gf_prog["value"] = 0
        self._sts(f"Goldfishing {ns:,} games over {nt} turns...")

        # Gather cached combo data if available
        combo_pieces = getattr(self, '_cached_combo_pieces', None)

        # Pass commander info for combo checks (commanders are always accessible from command zone)
        commanders = [c for c in self.cards if c.is_commander]

        def pcb(p): self.root.after(0, lambda: self.gf_prog.configure(value=p*100))
        def go():
            try:
                r = SimEngine.sim_goldfish(deck_cards, ns, nt, pcb, combo_pieces=combo_pieces,
                                           commanders=commanders)
            except Exception as e:
                import traceback
                err_msg = traceback.format_exc()
                self.root.after(0, lambda: self._show_gf_error(err_msg, nt))
                return
            self.root.after(0, lambda: self._show_gf(r, nt))
        threading.Thread(target=go, daemon=True).start()

    def _show_gf_error(self, err_msg, nt):
        """Display goldfish simulation error to user."""
        self.gf_btn.configure(state=tk.NORMAL); self._sts("Goldfish error!")
        t = self.gf_text; t.configure(state=tk.NORMAL); t.delete("1.0", tk.END)
        t.insert(tk.END, "GOLDFISH SIMULATION ERROR\n", "header")
        t.insert(tk.END, f"\nAn error occurred during simulation:\n\n{err_msg}\n")
        t.insert(tk.END, "\nPlease report this error.", "info")
        t.configure(state=tk.DISABLED)

    def _show_gf(self, r, nt):
        self.gf_btn.configure(state=tk.NORMAL); self._sts("Goldfish complete!")
        t = self.gf_text; t.configure(state=tk.NORMAL); t.delete("1.0", tk.END)
        if not r: t.insert(tk.END, "Deck too small."); t.configure(state=tk.DISABLED); return

        ld = r["land_drops"]; am = r["avail_mana"]; hs = r["hand_size"]
        bd = r["bonus_draws"]; tc = r["total_cards"]
        cdt = r["cards_drawn_this_turn"]
        bfc = r["bf_creatures"]; bfa = r["bf_artifacts"]; bfe = r["bf_enchantments"]; bft = r["bf_total"]
        tkc = r["token_count"]; cp = r["combat_power"]; cd = r["cumul_damage"]
        gs = r["gy_size"]
        draw_sources = r.get("draw_sources", [])
        combo_stats = r.get("combo_stats", [])
        w1 = r.get("win_turn_1opp"); w3 = r.get("win_turn_3opp")

        # Diagnostic: detect all-zero results (should never happen with valid deck)
        if hs.get(0, 0) == 0 and ld.get(1, 0) == 0:
            deck_cards = self._get_deck_cards()
            deck_size = sum(c.quantity for c in deck_cards)
            cats = {}
            for c in deck_cards: cats[c.category] = cats.get(c.category, 0) + c.quantity
            t.insert(tk.END, "⚠ DIAGNOSTIC: All simulation values are zero!\n", "header")
            t.insert(tk.END, f"  Deck: {len(deck_cards)} unique cards, {deck_size} total\n")
            t.insert(tk.END, f"  Categories: {cats}\n")
            t.insert(tk.END, f"  hand_size[0]={hs.get(0)}, land_drops[1]={ld.get(1)}\n")
            t.insert(tk.END, f"  Result keys: {list(r.keys())}\n\n")
            t.insert(tk.END, "This may indicate a bug. Please try re-importing your deck.\n\n")

        nsims = int(self.gf_n.get())
        combo_note = ""
        if hasattr(self, '_cached_combo_pieces') and self._cached_combo_pieces:
            combo_note = f" | {len(self._cached_combo_pieces)} combo(s) tracked"

        # ═══════════════════════════════════════════════════════════
        # SECTION 1: MANA DEVELOPMENT
        # ═══════════════════════════════════════════════════════════
        t.insert(tk.END, "MANA DEVELOPMENT\n", "header")
        t.insert(tk.END, f"(Average across {nsims:,} games{combo_note})\n\n")

        hdr = f"  {'Turn':>5s}  {'Lands':>6s}  {'Ramp':>6s}  {'Total':>6s}  {'Mana Bar':>20s}"
        t.insert(tk.END, hdr + "\n", "info")
        t.insert(tk.END, "  " + "-" * 55 + "\n")
        for turn in range(nt + 1):
            lb = "Open" if turn == 0 else f"T{turn}"
            lands = ld[turn]; total_m = am[turn]; ramp = max(0, total_m - lands)
            bar = "█" * int(total_m) + "░" * max(0, int(10 - total_m))
            tag = "good" if total_m >= turn + 1 else "info"
            t.insert(tk.END, f"  {lb:>5s}  {lands:>6.1f}  {ramp:>+5.1f}  {total_m:>6.1f}  {bar}\n", tag)

        # ═══════════════════════════════════════════════════════════
        # SECTION 2: CARD FLOW
        # ═══════════════════════════════════════════════════════════
        t.insert(tk.END, "\n")
        t.insert(tk.END, "═" * 70 + "\n")
        t.insert(tk.END, "CARD FLOW\n", "header")
        t.insert(tk.END, "(Cards drawn per turn, hand size, total cards seen)\n\n")

        hdr2 = f"  {'Turn':>5s}  {'Drawn':>6s}  {'Hand':>6s}  {'In Play':>7s}  {'GY':>5s}  {'Total Seen':>10s}"
        t.insert(tk.END, hdr2 + "\n", "info")
        t.insert(tk.END, "  " + "-" * 55 + "\n")
        for turn in range(nt + 1):
            lb = "Open" if turn == 0 else f"T{turn}"
            drawn = cdt.get(turn, 0) if turn > 0 else 7
            hand_ct = hs.get(turn, 0)
            in_play = bft.get(turn, 0) + ld.get(turn, 0)
            gy = gs.get(turn, 0)
            total = tc.get(turn, 0)
            tag = "good" if drawn > 1.5 else "info"
            t.insert(tk.END, f"  {lb:>5s}  {drawn:>6.1f}  {hand_ct:>6.1f}  {in_play:>7.1f}  {gy:>5.1f}  {total:>10.1f}\n", tag)

        final_bonus = bd.get(nt, 0)
        t.insert(tk.END, f"\n  By turn {nt}: {final_bonus:.1f} bonus cards from draw/tutor effects\n", "good")

        # ═══════════════════════════════════════════════════════════
        # SECTION 3: BOARD STATE
        # ═══════════════════════════════════════════════════════════
        t.insert(tk.END, "\n")
        t.insert(tk.END, "═" * 70 + "\n")
        t.insert(tk.END, "BOARD STATE\n", "header")
        t.insert(tk.END, "(Average permanents on battlefield by turn)\n\n")

        hdr3 = f"  {'Turn':>5s}  {'Creatures':>9s}  {'Artifacts':>9s}  {'Enchant':>9s}  {'Lands':>6s}  {'Tokens':>7s}  {'Total':>6s}"
        t.insert(tk.END, hdr3 + "\n", "info")
        t.insert(tk.END, "  " + "-" * 70 + "\n")
        for turn in range(nt + 1):
            lb = "Open" if turn == 0 else f"T{turn}"
            cr = bfc.get(turn, 0); ar = bfa.get(turn, 0); en = bfe.get(turn, 0)
            ln = ld.get(turn, 0); tk_ct = tkc.get(turn, 0)
            tot = cr + ar + en + ln
            tag = "good" if cr >= 3 else "info"
            t.insert(tk.END, f"  {lb:>5s}  {cr:>9.1f}  {ar:>9.1f}  {en:>9.1f}  {ln:>6.1f}  {tk_ct:>7.1f}  {tot:>6.1f}\n", tag)

        # ═══════════════════════════════════════════════════════════
        # SECTION 4: DAMAGE CLOCK & WIN ESTIMATION
        # ═══════════════════════════════════════════════════════════
        t.insert(tk.END, "\n")
        t.insert(tk.END, "═" * 70 + "\n")
        t.insert(tk.END, "DAMAGE CLOCK\n", "header")
        t.insert(tk.END, "(Combat damage from T3 + drain/ping from permanents)\n\n")

        hdr4 = f"  {'Turn':>5s}  {'Power':>7s}  {'Tokens':>7s}  {'Dmg/Turn':>8s}  {'Cumul':>7s}  {'Progress':>20s}"
        t.insert(tk.END, hdr4 + "\n", "info")
        t.insert(tk.END, "  " + "-" * 65 + "\n")
        prev_cum = 0
        for turn in range(nt + 1):
            lb = "Open" if turn == 0 else f"T{turn}"
            pw = cp.get(turn, 0); cum = cd.get(turn, 0)
            tok = tkc.get(turn, 0)
            dmg_this = cum - prev_cum; prev_cum = cum
            pct40 = min(cum / 40, 1.0) if cum > 0 else 0
            bar = "▓" * int(pct40 * 20) + "░" * (20 - int(pct40 * 20))
            tag = "good" if cum >= 40 else "sub" if cum >= 20 else "info"
            t.insert(tk.END, f"  {lb:>5s}  {pw:>7.1f}  {tok:>7.1f}  {dmg_this:>8.1f}  {cum:>7.1f}  {bar}\n", tag)

        # Win estimation
        t.insert(tk.END, "\n")
        t.insert(tk.END, "  WIN ESTIMATION\n", "sub")
        if w1 is not None:
            t.insert(tk.END, f"    Kill 1 opponent (40 life):   ~Turn {w1}\n", "good")
        else:
            t.insert(tk.END, f"    Kill 1 opponent (40 life):   Not reached in {nt} turns\n", "warn")
        if w3 is not None:
            t.insert(tk.END, f"    Kill all 3 opponents (120):  ~Turn {w3}\n", "good")
        else:
            t.insert(tk.END, f"    Kill all 3 opponents (120):  Not reached in {nt} turns\n", "warn")

        # Combo win estimation
        if combo_stats:
            best_combo = max(combo_stats, key=lambda x: x["found_pct"])
            if best_combo["avg_turn"] is not None:
                t.insert(tk.END, f"    Combo win (best):            ~Turn {best_combo['avg_turn']:.1f} "
                                 f"({best_combo['found_pct']:.1f}% of games)\n", "good")

        # Alternate win condition estimation
        alt_win = r.get("alt_win_stats")
        if alt_win and alt_win["win_count"] > 0:
            type_labels = {
                "empty_library": "Library Empty", "treasure_count": "Treasure Win",
                "life_total_high": "Life Total Win", "life_total_exact": "Near-Death Win",
                "creature_count": "Creature Count", "artifact_count": "Artifact Count",
                "graveyard_creatures": "Graveyard Win", "counter_self": "Counter Win",
                "hand_size": "Hand Size Win", "power_threshold": "Power Win",
                "demon_count": "Demon Win", "poison": "Poison/Infect",
                "second_cast": "Second Casting", "instant_win": "Alt Win Condition",
                "gate_count": "Maze's End", "five_color": "5-Color Win",
                "type_count": "Tribal Win", "opponent_loses": "Opponent Loses",
            }
            type_str = ", ".join(type_labels.get(wt, wt) for wt in alt_win["types"])
            t.insert(tk.END, f"    Alt win ({type_str}): ~Turn {alt_win['avg_turn']:.1f} "
                             f"({alt_win['win_pct']:.1f}% of games)\n", "good")
            t.insert(tk.END, f"      Cards: {', '.join(alt_win['cards'])}\n", "info")
        elif alt_win and alt_win["win_count"] == 0:
            type_labels = {
                "empty_library": "Library Empty", "treasure_count": "Treasure Win",
                "life_total_high": "Life Total Win", "life_total_exact": "Near-Death Win",
                "creature_count": "Creature Count", "artifact_count": "Artifact Count",
                "graveyard_creatures": "Graveyard Win", "counter_self": "Counter Win",
                "hand_size": "Hand Size Win", "power_threshold": "Power Win",
                "demon_count": "Demon Win", "poison": "Poison/Infect",
                "second_cast": "Second Casting", "instant_win": "Alt Win Condition",
                "gate_count": "Maze's End", "five_color": "5-Color Win",
                "type_count": "Tribal Win", "opponent_loses": "Opponent Loses",
            }
            type_str = ", ".join(type_labels.get(wt, wt) for wt in alt_win["types"])
            t.insert(tk.END, f"    Alt win ({type_str}): Not achieved in {nt} turns\n", "warn")
            t.insert(tk.END, f"      Cards: {', '.join(alt_win['cards'])}\n", "info")

        t.insert(tk.END, "    (Includes combat + drain/ping effects; combat starts T3)\n", "info")

        # ═══════════════════════════════════════════════════════════
        # SECTION 5: DRAW SOURCES
        # ═══════════════════════════════════════════════════════════
        if draw_sources:
            t.insert(tk.END, "\n")
            t.insert(tk.END, "═" * 70 + "\n")
            t.insert(tk.END, "DRAW SOURCES IN DECK\n", "header")
            engines = [d for d in draw_sources if d["repeating"]]
            one_shots = [d for d in draw_sources if not d["repeating"]]
            if engines:
                t.insert(tk.END, "\n  Draw Engines (repeating):\n", "info")
                for d in sorted(engines, key=lambda x: x["draws"], reverse=True):
                    lbl = f" [{d['label']}]" if d.get("label") else ""
                    t.insert(tk.END, f"    {d['name']} (x{d['qty']}) — ~{d['draws']}/turn{lbl}\n", "good")
            if one_shots:
                t.insert(tk.END, f"\n  One-Shot ({len(one_shots)} cards):\n", "info")
                for d in sorted(one_shots, key=lambda x: x["draws"], reverse=True)[:10]:
                    lbl = f" [{d['label']}]" if d.get("label") else ""
                    t.insert(tk.END, f"    {d['name']} (x{d['qty']}) — draws {d['draws']}{lbl}\n")
                if len(one_shots) > 10:
                    t.insert(tk.END, f"    ... and {len(one_shots) - 10} more\n", "info")

        # ═══════════════════════════════════════════════════════════
        # SECTION 6: COMBO ASSEMBLY
        # ═══════════════════════════════════════════════════════════
        if combo_stats:
            t.insert(tk.END, "\n")
            t.insert(tk.END, "═" * 70 + "\n")
            t.insert(tk.END, "COMBO ASSEMBLY\n", "header")
            t.insert(tk.END, f"(Tracked across {nsims:,} games — includes tutors, draw, commander)\n\n")

            milestones = [t2 for t2 in [3, 5, 7, 10, 15, 20] if t2 <= nt]
            hdr5 = f"  {'Combo':<40s} {'Pcs':>3s}"
            for mt in milestones: hdr5 += f"  {'T'+str(mt):>6s}"
            hdr5 += f"  {'Avg Turn':>8s}  {'Total %':>7s}"
            t.insert(tk.END, hdr5 + "\n", "info")
            t.insert(tk.END, "  " + "-" * (len(hdr5) - 2) + "\n")

            sorted_combos = sorted(combo_stats, key=lambda x: x["found_pct"], reverse=True)
            for cs in sorted_combos:
                label = cs["label"]
                if len(label) > 38: label = label[:35] + "..."
                row = f"  {label:<40s} {cs['pieces']:>3d}"
                for mt in milestones:
                    pct = cs["pct_by_turn"].get(mt, 0)
                    row += f"  {pct:>5.1f}%"
                if cs["avg_turn"] is not None:
                    avg_label = "~T" + f"{cs['avg_turn']:.1f}"
                    row += f"  {avg_label:>8s}"
                else:
                    row += f"  {'never':>8s}"
                row += f"  {cs['found_pct']:>6.1f}%"
                tag = "good" if cs["found_pct"] >= 50 else "sub" if cs["found_pct"] >= 20 else "info" if cs["found_pct"] > 0 else "warn"
                t.insert(tk.END, row + "\n", tag)

        t.configure(state=tk.DISABLED)

    # ---- ANALYSIS TAB ----
    def _ui_analysis(self, f):
        ctrl = ttk.LabelFrame(f, text="Deck Analysis & External Data", padding=10)
        ctrl.pack(fill=tk.X, padx=10, pady=10)
        row = ttk.Frame(ctrl); row.pack(fill=tk.X)
        self.analysis_btn = ttk.Button(row, text="Analyze Deck Effects", command=self._run_analysis)
        self.analysis_btn.pack(side=tk.LEFT, padx=(0,10))
        #self.edhrec_btn = ttk.Button(row, text="EDHREC Recommendations", command=self._run_edhrec)
        #self.edhrec_btn.pack(side=tk.LEFT, padx=(0,10))
        self.analysis_prog = ttk.Progressbar(ctrl, mode="determinate", length=500)
        self.analysis_prog.pack(fill=tk.X, pady=5)
        self.analysis_text = tk.Text(f, font=("Consolas", 10), bg="#16213e", fg="#e0e0e0",
                                      state=tk.DISABLED, wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.analysis_text.tag_configure("header", foreground="#e94560", font=("Consolas", 11, "bold"))
        self.analysis_text.tag_configure("sub", foreground="#FFD700", font=("Consolas", 10, "bold"))
        self.analysis_text.tag_configure("info", foreground="#00d2ff")
        self.analysis_text.tag_configure("good", foreground="#2ECC71")
        self.analysis_text.tag_configure("warn", foreground="#FF6B6B")
        self.analysis_text.tag_configure("combo", foreground="#E040FB")
        self.analysis_text.tag_configure("rec", foreground="#64FFDA")

    # ---- EFFECT CLASSIFIER (Feature 8) ----
    @staticmethod
    def _classify_effects(cards):
        """Classify all cards by their effect types for impact analysis."""
        effects = {
            "draw": [], "scry": [], "surveil": [], "brainstorm": [],
            "tutor": [], "cascade": [], "discover": [], "shuffle": [],
            "topdeck_manip": [], "impulse_draw": [], "wheel": [],
            "extra_turns": [], "extra_combat": [],
        }

        for card in cards:
            oracle = card.oracle_text.lower() if card.oracle_text else ""
            tl = card.type_line.lower() if card.type_line else ""
            name_l = card.name.lower()
            if not oracle: continue

            # Draw (exclude gift-a-card opponent draws)
            is_gift_only = False
            if "gift" in oracle and "draw" in oracle:
                oracle_no_gift = re.sub(r'gift\s+a\s+card\s*\([^)]*\)\s*', '', oracle)
                if "draw" not in oracle_no_gift:
                    is_gift_only = True
            if not is_gift_only:
                if "draw" in oracle and card.category == "Draw":
                    effects["draw"].append(card)
                elif "draw" in oracle and re.search(r'(?:you\s+)?draw\s+\w+\s+cards?|(?:you\s+)?draw\s+a\s+card', oracle):
                    # Make sure it's not just "they draw" / "opponent draws"
                    if not re.search(r'^(?:they|that player|each opponent)\s+draws?\s+', oracle):
                        effects["draw"].append(card)

            # Scry
            if re.search(r'\bscry\b', oracle):
                effects["scry"].append(card)

            # Surveil
            if re.search(r'\bsurveil\b', oracle):
                effects["surveil"].append(card)

            # Brainstorm / draw-then-put-back
            if re.search(r'draw\s+\w+\s+cards?.*?then\s+put\s+\w+\s+cards?\s+.*?(?:on\s+top|back)', oracle):
                effects["brainstorm"].append(card)

            # Tutor: "search your library" — classify by what it can find
            if re.search(r'search\s+your\s+library', oracle):
                effects["tutor"].append(card)
                # Subtype classification for combo probability
                tutor_type = "any"  # default: can find anything
                # Land-only tutors (fetchlands, Crop Rotation, etc.)
                if re.search(r'search\s+your\s+library\s+for\s+(?:a|an|up\s+to\s+\w+)\s+(?:basic\s+)?(?:land|mountain|forest|swamp|plains|island|gate|desert)', oracle):
                    if not re.search(r'search\s+your\s+library\s+for\s+(?:a|an)\s+(?:creature|instant|sorcery|enchantment|artifact|card)', oracle):
                        tutor_type = "land"
                # Creature-only tutors
                if re.search(r'search\s+your\s+library\s+for\s+(?:a|an|up\s+to\s+\w+)\s+creature', oracle):
                    tutor_type = "creature"
                # Artifact-only tutors
                if re.search(r'search\s+your\s+library\s+for\s+(?:a|an|up\s+to\s+\w+)\s+artifact', oracle):
                    tutor_type = "artifact"
                # Enchantment-only tutors (Idyllic Tutor, Enlightened Tutor)
                if re.search(r'search\s+your\s+library\s+for\s+(?:a|an|up\s+to\s+\w+)\s+enchantment', oracle):
                    tutor_type = "enchantment"
                # Combined artifact or enchantment (Enlightened Tutor)
                if re.search(r'search\s+your\s+library\s+for\s+(?:a|an)\s+(?:artifact\s+or\s+enchantment|enchantment\s+or\s+artifact)', oracle):
                    tutor_type = "artifact_enchantment"
                # Instant/sorcery tutors (Mystical Tutor, Spellseeker)
                if re.search(r'search\s+your\s+library\s+for\s+(?:a|an)\s+(?:instant|sorcery|instant\s+or\s+sorcery|instant\s+card\s+or\s+a\s+sorcery)', oracle):
                    tutor_type = "instant_sorcery"
                # "search your library for a card" = any card tutor
                if re.search(r'search\s+your\s+library\s+for\s+a\s+card\b', oracle):
                    tutor_type = "any"
                # Store subtype on the card object for later use
                card._tutor_subtype = tutor_type

            # Cascade
            if "cascade" in oracle:
                effects["cascade"].append(card)

            # Discover
            if re.search(r'\bdiscover\b', oracle):
                effects["discover"].append(card)

            # Shuffle (library manipulation)
            if re.search(r'shuffle\s+your\s+library', oracle) and "search" not in oracle:
                effects["shuffle"].append(card)

            # Topdeck manipulation: "look at the top", "reveal the top"
            if re.search(r'(?:look\s+at|reveal)\s+the\s+top', oracle):
                effects["topdeck_manip"].append(card)

            # Impulse draw: "exile the top ... you may play/cast"
            if re.search(r'exile\s+(?:the\s+)?top.*?(?:you\s+may\s+(?:play|cast)|until\s+(?:end\s+of|the\s+end))', oracle, re.DOTALL):
                effects["impulse_draw"].append(card)

            # Wheel: "each player ... draws" or "discard your hand ... draw"
            if re.search(r'(?:each\s+player|all\s+players)[^.]*?draws?\s+', oracle):
                effects["wheel"].append(card)
            elif re.search(r'discard\s+(?:your|their)\s+hand[^.]*?draw', oracle):
                effects["wheel"].append(card)

            # Extra turns
            if re.search(r'(?:take|takes?)\s+an?\s+extra\s+turn', oracle):
                effects["extra_turns"].append(card)

            # Extra combat
            if re.search(r'additional\s+combat\s+phase', oracle):
                effects["extra_combat"].append(card)

        return effects

    @staticmethod
    def _estimate_effect_impact(effects, total_cards):
        """Estimate the best-case average impact of effects on card advantage."""
        impacts = {}
        deck_size = total_cards

        # Draw: count total potential, separating free vs life-cost engines
        draw_cards = effects["draw"]
        total_draw_free = 0
        total_draw_life = 0
        life_cost_engines = []
        for c in draw_cards:
            est = SimEngine._estimate_card_draws(c)
            draws, rep, lbl, lc = est
            if lc > 0:
                # Life-costing engine: use 50% effectiveness
                total_draw_life += draws * c.quantity
                life_cost_engines.append((c, draws, rep, lbl, lc))
            else:
                total_draw_free += draws * c.quantity
        total_draw_adjusted = total_draw_free + int(total_draw_life * 0.5)
        impacts["draw"] = {"count": len(draw_cards),
                           "total_qty": sum(c.quantity for c in draw_cards),
                           "potential": total_draw_adjusted,
                           "desc": f"~{total_draw_adjusted} bonus cards (free: {total_draw_free}, life-cost engines at 50%: ~{int(total_draw_life * 0.5)})"}
        # Store life-cost info for detail display
        impacts["_life_cost_engines"] = life_cost_engines

        # Scry: improves draw quality, ~equivalent to seeing 0.3 extra useful cards per scry
        scry_cards = effects["scry"]
        scry_total = 0
        for c in scry_cards:
            oracle = c.oracle_text.lower() if c.oracle_text else ""
            m = re.search(r'scry\s+(\w+)', oracle)
            if m:
                NUMBER_WORDS = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5}
                w = m.group(1)
                n = NUMBER_WORDS.get(w, int(w) if w.isdigit() else 1)
                scry_total += n * c.quantity
        impacts["scry"] = {"count": len(scry_cards),
                           "total_qty": sum(c.quantity for c in scry_cards),
                           "potential": scry_total,
                           "desc": f"~{scry_total} total scry depth (improves top ~{scry_total*30}% of draws)"}

        # Surveil: like scry but also fills graveyard (relevant for recursion)
        surv_cards = effects["surveil"]
        surv_total = 0
        for c in surv_cards:
            oracle = c.oracle_text.lower() if c.oracle_text else ""
            m = re.search(r'surveil\s+(\w+)', oracle)
            if m:
                NUMBER_WORDS = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5}
                w = m.group(1)
                surv_total += NUMBER_WORDS.get(w, int(w) if w.isdigit() else 1) * c.quantity
        impacts["surveil"] = {"count": len(surv_cards),
                              "total_qty": sum(c.quantity for c in surv_cards),
                              "potential": surv_total,
                              "desc": f"~{surv_total} surveil depth (scry + graveyard fuel)"}

        # Brainstorm: improves quality significantly, net ~1 card but sees 3
        bs_cards = effects["brainstorm"]
        impacts["brainstorm"] = {"count": len(bs_cards),
                                 "total_qty": sum(c.quantity for c in bs_cards),
                                 "potential": sum(c.quantity for c in bs_cards) * 3,
                                 "desc": f"See ~{sum(c.quantity for c in bs_cards)*3} extra cards (net draw after put-back)"}

        # Tutors: equivalent to drawing your best card, massive impact
        tutor_cards = effects["tutor"]
        impacts["tutor"] = {"count": len(tutor_cards),
                            "total_qty": sum(c.quantity for c in tutor_cards),
                            "potential": sum(c.quantity for c in tutor_cards),
                            "desc": f"{sum(c.quantity for c in tutor_cards)} tutor effects (each ≈ finding your best card)"}

        # Cascade: casts a free spell, avg CMC value ≈ (cascade CMC - 1) / 2
        casc_cards = effects["cascade"]
        casc_value = sum(max(1, int(c.cmc) - 1) for c in casc_cards)
        impacts["cascade"] = {"count": len(casc_cards),
                              "total_qty": sum(c.quantity for c in casc_cards),
                              "potential": casc_value,
                              "desc": f"~{casc_value} total free mana value from cascade"}

        # Discover: similar to cascade
        disc_cards = effects["discover"]
        disc_value = sum(max(1, int(c.cmc) - 1) for c in disc_cards)
        impacts["discover"] = {"count": len(disc_cards),
                               "total_qty": sum(c.quantity for c in disc_cards),
                               "potential": disc_value,
                               "desc": f"~{disc_value} total free mana value from discover"}

        # Impulse draw: exile-play, roughly 1 extra card per effect
        imp_cards = effects["impulse_draw"]
        impacts["impulse_draw"] = {"count": len(imp_cards),
                                   "total_qty": sum(c.quantity for c in imp_cards),
                                   "potential": sum(c.quantity for c in imp_cards),
                                   "desc": f"~{sum(c.quantity for c in imp_cards)} impulse draw effects"}

        # Wheels: massive draw reset, typically draw 7
        whl_cards = effects["wheel"]
        impacts["wheel"] = {"count": len(whl_cards),
                            "total_qty": sum(c.quantity for c in whl_cards),
                            "potential": sum(c.quantity for c in whl_cards) * 7,
                            "desc": f"~{sum(c.quantity for c in whl_cards)*7} cards from wheel effects (draw 7 each)"}

        return impacts

    def _run_analysis(self):
        if not self.cards:
            messagebox.showwarning("No Deck", "Import a deck first!"); return
        deck_cards = self._get_deck_cards()
        t = self.analysis_text; t.configure(state=tk.NORMAL); t.delete("1.0", tk.END)

        effects = self._classify_effects(deck_cards)
        total = sum(c.quantity for c in deck_cards)
        impacts = self._estimate_effect_impact(effects, total)

        t.insert(tk.END, "DECK EFFECT ANALYSIS\n", "header")
        t.insert(tk.END, f"({total} cards in deck, best-case scenario averages)\n\n")

        # Summary table
        labels = [
            ("draw", "Card Draw"),  ("scry", "Scry"),
            ("surveil", "Surveil"), ("brainstorm", "Brainstorm/Put-Back"),
            ("tutor", "Tutors (Search Library)"), ("cascade", "Cascade"),
            ("discover", "Discover"), ("impulse_draw", "Impulse Draw (Exile-Play)"),
            ("wheel", "Wheel Effects"),
        ]
        t.insert(tk.END, f"{'Effect':<28s} {'Cards':>6s} {'Qty':>5s}  Impact\n", "info")
        t.insert(tk.END, "-" * 80 + "\n")
        total_advantage = 0
        for key, label in labels:
            imp = impacts.get(key, {})
            cnt = imp.get("count", 0)
            qty = imp.get("total_qty", 0)
            if cnt > 0:
                tag = "good"
                t.insert(tk.END, f"  {label:<26s} {cnt:>6d} {qty:>5d}  {imp['desc']}\n", tag)
                total_advantage += imp.get("potential", 0)

        # Cards with no effects detected
        effect_card_names = set()
        for lst in effects.values():
            for c in lst:
                effect_card_names.add(c.name)
        no_effect = [c for c in deck_cards if c.name not in effect_card_names and c.category != "Land"]

        t.insert(tk.END, f"\n  TOTAL CARD ADVANTAGE POTENTIAL: ~{total_advantage} bonus cards/interactions\n", "sub")
        t.insert(tk.END, f"  (This represents best-case cumulative impact over a full game)\n\n", "info")

        # Detail each effect category with card names
        t.insert(tk.END, "EFFECT DETAILS\n", "header")
        for key, label in labels:
            cards_list = effects.get(key, [])
            if not cards_list: continue
            t.insert(tk.END, f"\n  {label} ({len(cards_list)} cards):\n", "sub")
            for c in sorted(cards_list, key=lambda x: x.cmc):
                est = SimEngine._estimate_card_draws(c) if key == "draw" else (0, False, "", 0)
                extra = ""
                if key == "draw" and est[0] > 0:
                    rep_str = "repeating" if est[1] else "one-shot"
                    life_note = ""
                    if est[3] > 0:
                        life_note = f" ⚠ COSTS LIFE (~{est[3]}/cycle, counted at 50%)"
                    extra = f" — ~{est[0]} draw [{est[2]}] ({rep_str}){life_note}"
                elif key == "tutor":
                    oracle = c.oracle_text.lower() if c.oracle_text else ""
                    subtype = getattr(c, '_tutor_subtype', 'any')
                    subtype_label = TUTOR_SUBTYPES.get(subtype, subtype)
                    if subtype == "artifact_enchantment":
                        subtype_label = "Artifact/Enchantment Tutor"
                    dest = ""
                    if "to the top" in oracle or "on top" in oracle:
                        dest = " → top of library"
                    elif "to your hand" in oracle or "into your hand" in oracle:
                        dest = " → hand"
                    elif "onto the battlefield" in oracle or "into play" in oracle:
                        dest = " → battlefield"
                    elif "into your graveyard" in oracle or "into a graveyard" in oracle:
                        dest = " → graveyard"
                    else:
                        dest = " → hand/library"
                    extra = f" — [{subtype_label}]{dest}"
                t.insert(tk.END, f"    {c.name} (CMC {int(c.cmc)}){extra}\n")

        # Topdeck manipulation and shuffle summary
        td_cards = effects.get("topdeck_manip", [])
        sh_cards = effects.get("shuffle", [])
        if td_cards or sh_cards:
            t.insert(tk.END, "\n  Library Manipulation:\n", "sub")
            if td_cards:
                names = ", ".join(c.name for c in td_cards)
                t.insert(tk.END, f"    Topdeck ({len(td_cards)}): {names}\n", "info")
            if sh_cards:
                names = ", ".join(c.name for c in sh_cards)
                t.insert(tk.END, f"    Shuffle ({len(sh_cards)}): {names}\n", "info")

        # ---- GRAVEYARD SYNERGY ANALYSIS ----
        gy_recursion = []  # cards that return things from graveyard
        gy_selfmill = []   # cards that mill yourself
        gy_entomb = []     # cards that tutor to graveyard
        gy_flashback = []  # cards castable from graveyard
        gy_dredge = []     # dredge cards
        gy_discard_outlets = []  # discard outlets

        for c in deck_cards:
            oracle = (c.oracle_text or "").lower()
            tl = (c.type_line or "").lower()
            # Recursion
            if re.search(r'return\s+(?:a|target|up\s+to|all)\s+[^.]*?from\s+(?:your\s+|a\s+)?graveyard', oracle):
                gy_recursion.append(c)
            elif re.search(r'(?:put|return)\s+(?:it|that card|target\s+\w+\s+card)\s+from\s+(?:your\s+|a\s+)?graveyard\s+(?:onto|on|to)\s+the\s+battlefield', oracle):
                gy_recursion.append(c)
            elif re.search(r'(?:cast|play)\s+[^.]*?from\s+(?:your\s+)?graveyard', oracle):
                gy_recursion.append(c)
            # Self-mill
            if re.search(r'mill\s+\w+', oracle) and "opponent" not in oracle.split("mill")[0][-20:]:
                gy_selfmill.append(c)
            elif re.search(r'put\s+the\s+top\s+\w+\s+cards?\s+(?:of\s+your\s+library\s+)?into\s+(?:your\s+)?graveyard', oracle):
                gy_selfmill.append(c)
            # Entomb / tutor to graveyard
            if re.search(r'search\s+your\s+library[^.]*?put\s+(?:it|that card)\s+into\s+(?:your\s+)?graveyard', oracle):
                gy_entomb.append(c)
            # Flashback / retrace / escape / cast from graveyard
            if "flashback" in oracle or "retrace" in oracle or "escape" in oracle:
                gy_flashback.append(c)
            elif re.search(r'(?:you may\s+)?(?:cast|play)\s+(?:this card|it)\s+from\s+(?:your\s+)?graveyard', oracle):
                gy_flashback.append(c)
            # Dredge
            if "dredge" in oracle:
                gy_dredge.append(c)
            # Discard outlets
            if re.search(r'discard\s+(?:a|an|one|two|three)\s+cards?(?:\s*:)', oracle):
                gy_discard_outlets.append(c)
            elif re.search(r'as\s+an\s+additional\s+cost[^.]*?discard', oracle):
                gy_discard_outlets.append(c)

        # Also check commanders
        cmdr_recursion = []
        for c in self.cards:
            if c.is_commander:
                oracle = (c.oracle_text or "").lower()
                if re.search(r'(?:cast|play|return)\s+(?:\w+\s+)?(?:card|creature|artifact|permanent)?\s*(?:from|in)\s+(?:your\s+)?graveyard', oracle):
                    cmdr_recursion.append(c)
                elif re.search(r'(?:put|return)\s+(?:it|that card|target)[^.]*?(?:from\s+(?:your\s+)?graveyard|graveyard\s+to)', oracle):
                    cmdr_recursion.append(c)

        total_gy = len(gy_recursion) + len(gy_selfmill) + len(gy_entomb) + len(gy_flashback) + len(gy_dredge) + len(gy_discard_outlets) + len(cmdr_recursion)
        if total_gy > 0:
            t.insert(tk.END, "\n")
            t.insert(tk.END, "═" * 80 + "\n")
            t.insert(tk.END, "GRAVEYARD SYNERGY ANALYSIS\n", "header")
            # Graveyard density rating
            gy_pct = total_gy / len(deck_cards) * 100 if deck_cards else 0
            if gy_pct >= 20:
                gy_rating = "HEAVY — graveyard is a primary game zone"
                gy_tag = "good"
            elif gy_pct >= 10:
                gy_rating = "MODERATE — significant graveyard interaction"
                gy_tag = "sub"
            elif gy_pct >= 5:
                gy_rating = "LIGHT — some graveyard utility"
                gy_tag = "info"
            else:
                gy_rating = "MINIMAL — incidental graveyard use"
                gy_tag = "info"
            t.insert(tk.END, f"  Graveyard density: {total_gy} cards ({gy_pct:.1f}%) — {gy_rating}\n", gy_tag)

            if cmdr_recursion:
                names = ", ".join(c.name for c in cmdr_recursion)
                t.insert(tk.END, f"\n  ⭐ Commander recursion: {names}\n", "sub")
                t.insert(tk.END, "     Your graveyard acts as a second hand. Combo sim treats GY pieces as accessible.\n", "info")

            if gy_recursion:
                t.insert(tk.END, f"\n  Recursion ({len(gy_recursion)}):\n", "sub")
                for c in sorted(gy_recursion, key=lambda x: x.cmc):
                    t.insert(tk.END, f"    {c.name} (CMC {int(c.cmc)})\n")
            if gy_selfmill:
                t.insert(tk.END, f"\n  Self-Mill ({len(gy_selfmill)}):\n", "sub")
                for c in sorted(gy_selfmill, key=lambda x: x.cmc):
                    t.insert(tk.END, f"    {c.name} (CMC {int(c.cmc)})\n")
            if gy_entomb:
                t.insert(tk.END, f"\n  Tutor-to-Graveyard ({len(gy_entomb)}):\n", "sub")
                for c in sorted(gy_entomb, key=lambda x: x.cmc):
                    t.insert(tk.END, f"    {c.name} (CMC {int(c.cmc)})\n")
            if gy_flashback:
                t.insert(tk.END, f"\n  Castable from Graveyard ({len(gy_flashback)}):\n", "sub")
                for c in sorted(gy_flashback, key=lambda x: x.cmc):
                    t.insert(tk.END, f"    {c.name} (CMC {int(c.cmc)})\n")
            if gy_dredge:
                t.insert(tk.END, f"\n  Dredge ({len(gy_dredge)}):\n", "sub")
                for c in sorted(gy_dredge, key=lambda x: x.cmc):
                    t.insert(tk.END, f"    {c.name} (CMC {int(c.cmc)})\n")
            if gy_discard_outlets:
                t.insert(tk.END, f"\n  Discard Outlets ({len(gy_discard_outlets)}):\n", "sub")
                for c in sorted(gy_discard_outlets, key=lambda x: x.cmc):
                    t.insert(tk.END, f"    {c.name} (CMC {int(c.cmc)})\n")

            # Advice
            if cmdr_recursion and len(gy_selfmill) < 3:
                t.insert(tk.END, "\n  ⚠ Recursion commander but few self-mill cards. Consider adding mill/dredge to fuel graveyard.\n", "warn")
            if len(gy_recursion) + len(cmdr_recursion) > 0 and len(gy_entomb) == 0:
                t.insert(tk.END, "  💡 No tutor-to-graveyard effects. Entomb/Buried Alive can turbo-charge recursion combos.\n", "info")

        # ---- MANA BASE ANALYSIS (Frank Karsten methodology) ----
        t.insert(tk.END, "\n")
        t.insert(tk.END, "═" * 80 + "\n")
        t.insert(tk.END, "MANA BASE ANALYSIS\n", "header")

        # Karsten's 99-card Commander table (with free mulligan + turn 1 draw)
        # Key: (turn_to_cast, pips_of_color) -> required_sources (for ~41 land deck)
        # Sources: Frank Karsten 2022, Peasant Magic mirror, EPIC EDH, Gist analysis
        # Format: KARSTEN_99[cmc][pips] = sources needed
        KARSTEN_99 = {
            1: {1: 15},                              # C on T1: 15 sources
            2: {1: 13, 2: 21},                       # 1C: 13, CC: 21
            3: {1: 10, 2: 18, 3: 27},               # 2C: 10, 1CC: 18, CCC: 27
            4: {1: 9, 2: 15, 3: 23, 4: 31},         # 3C: 9, 2CC: 15, 1CCC: 23, CCCC: 31
            5: {1: 8, 2: 13, 3: 19, 4: 26, 5: 33},  # 4C: 8, 3CC: 13, 2CCC: 19, 1CCCC: 26
            6: {1: 7, 2: 11, 3: 17, 4: 22, 5: 29},
            7: {1: 7, 2: 10, 3: 15, 4: 20, 5: 26},
        }

        all_cards = [c for c in self.cards]
        color_names = {'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green'}

        # Per-card analysis: find the most demanding card for each color
        # For each color, track: required_sources and the card that drives it
        color_demand = {}  # color -> (required_sources, card_name, cmc, pips)
        multicolor_demand = {}  # frozenset(colors) -> (required_sources, card_name)

        for c in all_cards:
            if c.category == "Land":
                continue
            reqs = SimEngine._parse_color_requirements(c.mana_cost)
            if not reqs:
                continue
            cmc = max(1, int(c.cmc))
            cmc_key = min(cmc, 7)  # cap at 7 for table lookup

            # Check if multicolored (pips of more than one color)
            real_colors = {k: v for k, v in reqs.items() if k in 'WUBRG'}
            is_gold = len(real_colors) > 1

            for color, pips in real_colors.items():
                pips_key = min(pips, 5)
                if cmc_key in KARSTEN_99 and pips_key in KARSTEN_99[cmc_key]:
                    needed = KARSTEN_99[cmc_key][pips_key]
                    # Gold card rule: +1 to all requirements
                    if is_gold:
                        needed += 1
                    if color not in color_demand or needed > color_demand[color][0]:
                        color_demand[color] = (needed, c.name, cmc, pips)

            # Track combined color requirement for gold cards
            if is_gold:
                total_colored_pips = sum(real_colors.values())
                combo_key = frozenset(real_colors.keys())
                pips_key = min(total_colored_pips, 5)
                if cmc_key in KARSTEN_99 and pips_key in KARSTEN_99[cmc_key]:
                    needed = KARSTEN_99[cmc_key][pips_key] + 1  # +1 for gold
                    if combo_key not in multicolor_demand or needed > multicolor_demand[combo_key][0]:
                        multicolor_demand[combo_key] = (needed, c.name)

        # Count actual color sources in deck (Karsten weighting)
        # Lands = 1 full source, Mana rocks = 3/4 source, Mana dorks (creatures) = 1/2 source
        color_sources = {'W': 0.0, 'U': 0.0, 'B': 0.0, 'R': 0.0, 'G': 0.0, 'any': 0.0}
        source_details = {'lands': {}, 'rocks': {}, 'dorks': {}}  # color -> count

        for col in 'WUBRG':
            source_details['lands'][col] = 0
            source_details['rocks'][col] = 0
            source_details['dorks'][col] = 0
        source_details['lands']['any'] = 0
        source_details['rocks']['any'] = 0
        source_details['dorks']['any'] = 0

        fetch_lands = []
        for c in all_cards:
            if c.category == "Land":
                colors = SimEngine._parse_color_production(c)
                oracle = (c.oracle_text or "").lower()
                # Detect fetch lands (count as full source for all fetchable colors)
                if re.search(r'search\s+your\s+library\s+for\s+(?:a|an)\s+(?:basic\s+)?(?:land|mountain|forest|swamp|plains|island)', oracle):
                    fetch_lands.append(c)
                if 'any' in colors or len(colors) > 1:
                    color_sources['any'] += c.quantity
                    source_details['lands']['any'] += c.quantity
                    # Also count toward each specific color
                    if 'any' in colors:
                        for col in 'WUBRG':
                            color_sources[col] += c.quantity
                    else:
                        for col in colors:
                            if col in color_sources:
                                color_sources[col] += c.quantity
                else:
                    for col in colors:
                        if col in color_sources:
                            color_sources[col] += c.quantity
                            source_details['lands'][col] += c.quantity
            elif c.category == "Ramp":
                colors = SimEngine._parse_color_production(c)
                tl = (c.type_line or "").lower()
                is_creature = "creature" in tl
                # Karsten: dorks = 0.5, rocks = 0.75
                weight = 0.5 if is_creature else 0.75
                src_key = 'dorks' if is_creature else 'rocks'
                if 'any' in colors or len(colors) > 1:
                    color_sources['any'] += c.quantity * weight
                    source_details[src_key]['any'] += c.quantity
                    if 'any' in colors:
                        for col in 'WUBRG':
                            color_sources[col] += c.quantity * weight
                    else:
                        for col in colors:
                            if col in color_sources:
                                color_sources[col] += c.quantity * weight
                else:
                    for col in colors:
                        if col in color_sources:
                            color_sources[col] += c.quantity * weight
                            source_details[src_key][col] += c.quantity

        active_colors = [col for col in 'WUBRG' if col in color_demand]

        if not active_colors:
            t.insert(tk.END, "  No colored mana requirements detected.\n", "info")
        else:
            # Per-color requirements vs sources
            t.insert(tk.END, f"  {'Color':>8s}  {'Need':>6s}  {'Have':>6s}  {'Gap':>5s}  {'Status':>8s}  Most Demanding Card\n", "info")
            t.insert(tk.END, "  " + "-" * 78 + "\n")

            shortfalls = []
            for col in active_colors:
                needed, card_name, cmc, pips = color_demand[col]
                have = color_sources.get(col, 0)
                gap = have - needed
                pip_str = col * pips
                cost_str = f"({cmc}CMC, {pip_str})"

                if gap >= 2:
                    status, tag = "Good", "good"
                elif gap >= 0:
                    status, tag = "OK", "sub"
                elif gap >= -3:
                    status, tag = "Low", "warn"
                else:
                    status, tag = "SHORT!", "bad"
                    shortfalls.append((col, needed, have, card_name))

                t.insert(tk.END, f"  {color_names[col]:>8s}  {needed:>6d}  {have:>6.1f}  {gap:>+5.1f}  ")
                t.insert(tk.END, f"{status:>8s}", tag)
                t.insert(tk.END, f"  {card_name} {cost_str}\n")

            # Multi-color combined requirements
            if multicolor_demand:
                t.insert(tk.END, f"\n  Gold Card Combined Requirements:\n", "sub")
                for combo_key, (needed, card_name) in sorted(multicolor_demand.items(),
                        key=lambda x: x[1][0], reverse=True):
                    colors_str = "/".join(sorted(combo_key))
                    # Total sources of any of these colors
                    # For 2-color: count sources that produce at least one of the colors
                    combined = 0
                    for c2 in all_cards:
                        if c2.category == "Land":
                            cp = SimEngine._parse_color_production(c2)
                            if 'any' in cp or any(col in cp for col in combo_key):
                                combined += c2.quantity
                        elif c2.category == "Ramp":
                            cp = SimEngine._parse_color_production(c2)
                            tl2 = (c2.type_line or "").lower()
                            wt = 0.5 if "creature" in tl2 else 0.75
                            if 'any' in cp or any(col in cp for col in combo_key):
                                combined += c2.quantity * wt
                    gap = combined - needed
                    if gap >= 0:
                        tag = "good"
                    elif gap >= -3:
                        tag = "warn"
                    else:
                        tag = "bad"
                    t.insert(tk.END, f"    {colors_str}: Need {needed} sources of ({colors_str}), have {combined:.1f} ")
                    t.insert(tk.END, f"({gap:+.1f})", tag)
                    t.insert(tk.END, f"  ← {card_name}\n")

            # Source breakdown
            t.insert(tk.END, f"\n  Source Breakdown (Karsten weighting: lands=1, rocks=¾, dorks=½):\n", "sub")
            for col in active_colors:
                lands = source_details['lands'].get(col, 0)
                any_lands = source_details['lands'].get('any', 0)
                rocks = source_details['rocks'].get(col, 0)
                any_rocks = source_details['rocks'].get('any', 0)
                dorks = source_details['dorks'].get(col, 0)
                any_dorks = source_details['dorks'].get('any', 0)
                total_have = color_sources.get(col, 0)
                t.insert(tk.END, f"    {color_names[col]:>6s}: {lands} dedicated lands")
                if any_lands:
                    t.insert(tk.END, f" + {any_lands} multi/any lands")
                if rocks + any_rocks:
                    t.insert(tk.END, f" + {rocks+any_rocks} rocks (×¾)")
                if dorks + any_dorks:
                    t.insert(tk.END, f" + {dorks+any_dorks} dorks (×½)")
                t.insert(tk.END, f" = {total_have:.1f} effective\n")

            if fetch_lands:
                t.insert(tk.END, f"\n    Fetch lands ({len(fetch_lands)}): ", "info")
                t.insert(tk.END, ", ".join(c.name for c in fetch_lands[:8]) + "\n")

            # Suggestions
            t.insert(tk.END, f"\n  Recommendations:\n", "sub")
            any_suggestion = False
            for col in active_colors:
                needed = color_demand[col][0]
                have = color_sources.get(col, 0)
                gap = have - needed
                if gap < 0:
                    deficit = int(-gap + 0.5)
                    card_name = color_demand[col][1]
                    t.insert(tk.END, f"    ⚠ {color_names[col]}: {deficit} more source(s) needed ", "warn")
                    t.insert(tk.END, f"(driven by {card_name}). Add {col}-producing lands/rocks.\n")
                    any_suggestion = True
            if not any_suggestion:
                t.insert(tk.END, "    ✓ Mana base meets Karsten requirements for all colors.\n", "good")

        # Combo Finder — integrated into analysis
        t.insert(tk.END, "\n")
        t.insert(tk.END, "═" * 80 + "\n")
        t.insert(tk.END, "COMBO FINDER (Commander Spellbook)\n", "header")
        t.insert(tk.END, "  Searching for combos...\n", "info")
        t.configure(state=tk.DISABLED)
        self._sts("Searching Commander Spellbook for combos...")

        def combo_search():
            card_names = [c.name for c in self.cards]
            data = self._fetch_combos(card_names)
            self.root.after(0, lambda: self._show_analysis_combos(t, data))

        threading.Thread(target=combo_search, daemon=True).start()

    def _show_analysis_combos(self, t, data):
        """Display combo results inline in the analysis text widget."""
        t.configure(state=tk.NORMAL)
        # Remove the "Searching..." line
        t.delete("end-2l", "end-1l")

        if "error" in data:
            t.insert(tk.END, f"  Error: {data['error']}\n", "warn")
            t.insert(tk.END, "  Note: backend.commanderspellbook.com must be reachable.\n", "info")
            t.configure(state=tk.DISABLED)
            self._sts("Combo search failed.")
            return

        # Extract combos
        raw_results = data.get("results", {})
        if isinstance(raw_results, dict):
            results = raw_results.get("included", [])
            almost_results = raw_results.get("almostIncluded", [])
        elif isinstance(raw_results, list):
            results = raw_results
            almost_results = []
        else:
            results = []
            almost_results = []

        my_cards = {c.name.lower() for c in self.cards}
        in_deck = []
        almost = []

        for combo in results if isinstance(results, list) else []:
            combo_info = self._parse_combo(combo, my_cards)
            if combo_info:
                in_deck.append(combo_info)

        for combo in almost_results if isinstance(almost_results, list) else []:
            combo_info = self._parse_combo(combo, my_cards)
            if combo_info:
                almost.append(combo_info)

        # Display combos in deck
        t.insert(tk.END, f"\n  Combos in your deck: {len(in_deck)}\n", "sub")

        # Cache combo piece sets for goldfish sim
        if in_deck:
            self._cached_combo_pieces = [set(c["cards"]) for c in in_deck]
        else:
            self._cached_combo_pieces = None
        if in_deck:
            for i, combo in enumerate(in_deck[:20], 1):
                cards_str = " + ".join(combo["cards"])
                t.insert(tk.END, f"\n  {i}. ", "combo")
                t.insert(tk.END, f"{cards_str}\n", "combo")
                if combo["results"]:
                    results_str = ", ".join(combo["results"][:5])
                    t.insert(tk.END, f"     → {results_str}\n", "good")
                if combo["prereqs"]:
                    prereq_str = ", ".join(combo["prereqs"][:3])
                    t.insert(tk.END, f"     Requires: {prereq_str}\n", "info")
                t.insert(tk.END, "     ")
                for card_name in combo["cards"]:
                    self._insert_combo_thumb(t, card_name)
                    t.insert(tk.END, " ")
                t.insert(tk.END, "\n")
            if len(in_deck) > 20:
                t.insert(tk.END, f"\n  ... and {len(in_deck)-20} more combos\n", "info")
        else:
            t.insert(tk.END, "  No complete combos found in your deck.\n")

        t.configure(state=tk.DISABLED)
        self._sts(f"Analysis complete! Found {len(in_deck)} combos.")

        # Note: Combo assembly probability is now tracked in the Goldfish Turns tab
        if in_deck:
            t.configure(state=tk.NORMAL)
            t.insert(tk.END, "\n  → Run the Goldfish sim for combo assembly probability by turn\n", "info")
            t.insert(tk.END, "    (tracks tutors, draw, commander, and mana curve)\n", "info")
            t.configure(state=tk.DISABLED)

    def _open_combo_window(self):
        if not self.cards:
            messagebox.showwarning("No Deck", "Import a deck first!"); return

        win = tk.Toplevel(self.root)
        win.title("Combo Finder — Commander Spellbook")
        win.geometry("1000x800")
        win.configure(bg="#1a1a2e")

        ctrl = ttk.LabelFrame(win, text="Combo Finder", padding=10)
        ctrl.pack(fill=tk.X, padx=10, pady=(10, 5))
        row = ttk.Frame(ctrl); row.pack(fill=tk.X)
        search_btn = ttk.Button(row, text="Search Commander Spellbook", 
                                command=lambda: self._combo_win_search(win, search_btn, prog, txt))
        search_btn.pack(side=tk.LEFT, padx=(0, 10))
        prog = ttk.Progressbar(ctrl, mode="determinate", length=500)
        prog.pack(fill=tk.X, pady=5)

        txt = tk.Text(win, font=("Consolas", 10), bg="#16213e", fg="#e0e0e0",
                      state=tk.DISABLED, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        txt.tag_configure("header", foreground="#e94560", font=("Consolas", 11, "bold"))
        txt.tag_configure("sub", foreground="#FFD700", font=("Consolas", 10, "bold"))
        txt.tag_configure("info", foreground="#00d2ff")
        txt.tag_configure("good", foreground="#2ECC71")
        txt.tag_configure("warn", foreground="#FF6B6B")
        txt.tag_configure("combo", foreground="#E040FB")

        # Store refs for image GC prevention
        if not hasattr(self, '_combo_img_refs'):
            self._combo_img_refs = []

        # Auto-search on open
        win.after(100, lambda: self._combo_win_search(win, search_btn, prog, txt))

    def _combo_win_search(self, win, btn, prog, txt):
        btn.configure(state=tk.DISABLED)
        prog["value"] = 10
        txt.configure(state=tk.NORMAL)
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, "Searching Commander Spellbook for combos...\n", "info")
        txt.configure(state=tk.DISABLED)
        self._sts("Searching Commander Spellbook for combos...")

        def go():
            card_names = [c.name for c in self.cards]
            combos = self._fetch_combos(card_names)
            self.root.after(0, lambda: self._combo_win_show(win, btn, prog, txt, combos))

        threading.Thread(target=go, daemon=True).start()

    def _combo_win_show(self, win, btn, prog, txt, data):
        """Display combo results and probability in the combo finder window."""
        btn.configure(state=tk.NORMAL)
        prog["value"] = 100
        t = txt
        t.configure(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        t.insert(tk.END, "COMBO FINDER (Commander Spellbook)\n", "header")

        if "error" in data:
            t.insert(tk.END, f"  Error: {data['error']}\n", "warn")
            t.insert(tk.END, "  Note: backend.commanderspellbook.com must be reachable.\n", "info")
            t.configure(state=tk.DISABLED)
            self._sts("Combo search failed.")
            return

        # Extract combos from the nested structure
        raw_results = data.get("results", {})
        if isinstance(raw_results, dict):
            results = raw_results.get("included", [])
            almost_results = raw_results.get("almostIncluded", [])
        elif isinstance(raw_results, list):
            results = raw_results
            almost_results = []
        else:
            results = []
            almost_results = []

        my_cards = {c.name.lower() for c in self.cards}
        in_deck = []
        almost = []

        for combo in results if isinstance(results, list) else []:
            combo_info = self._parse_combo(combo, my_cards)
            if combo_info:
                in_deck.append(combo_info)

        for combo in almost_results if isinstance(almost_results, list) else []:
            combo_info = self._parse_combo(combo, my_cards)
            if combo_info:
                almost.append(combo_info)

        # Display combos in deck
        t.insert(tk.END, f"\nCombos in your deck: {len(in_deck)}\n", "sub")

        # Cache combo piece sets for goldfish sim
        if in_deck:
            self._cached_combo_pieces = [set(c["cards"]) for c in in_deck]
        if in_deck:
            for i, combo in enumerate(in_deck[:20], 1):
                cards_str = " + ".join(combo["cards"])
                t.insert(tk.END, f"\n  {i}. ", "combo")
                t.insert(tk.END, f"{cards_str}\n", "combo")
                if combo["results"]:
                    results_str = ", ".join(combo["results"][:5])
                    t.insert(tk.END, f"     → {results_str}\n", "good")
                if combo["prereqs"]:
                    prereq_str = ", ".join(combo["prereqs"][:3])
                    t.insert(tk.END, f"     Requires: {prereq_str}\n", "info")
                t.insert(tk.END, "     ")
                for card_name in combo["cards"]:
                    self._insert_combo_thumb(t, card_name)
                    t.insert(tk.END, " ")
                t.insert(tk.END, "\n")
            if len(in_deck) > 20:
                t.insert(tk.END, f"\n  ... and {len(in_deck)-20} more combos\n", "info")
        else:
            t.insert(tk.END, "  No complete combos found in your deck.\n")

        # Display almost-combos
        t.insert(tk.END, f"\nAlmost-combos (add 1 card): {len(almost)}\n", "sub")
        if almost:
            for i, combo in enumerate(almost[:15], 1):
                cards_str = " + ".join(combo["cards"])
                missing_str = combo["missing"][0] if combo["missing"] else "?"
                t.insert(tk.END, f"\n  {i}. {cards_str}\n", "info")
                t.insert(tk.END, f"     Missing: {missing_str}\n", "warn")
                if combo["results"]:
                    results_str = ", ".join(combo["results"][:3])
                    t.insert(tk.END, f"     → {results_str}\n", "good")
                t.insert(tk.END, "     ")
                for card_name in combo["cards"]:
                    is_missing = card_name.lower() in {m.lower() for m in combo["missing"]}
                    self._insert_combo_thumb(t, card_name, missing=is_missing)
                    t.insert(tk.END, " ")
                t.insert(tk.END, "\n")
            if len(almost) > 15:
                t.insert(tk.END, f"\n  ... and {len(almost)-15} more near-combos\n", "info")

        t.configure(state=tk.DISABLED)
        self._sts(f"Found {len(in_deck)} combos in deck, {len(almost)} need 1 card.")

        # Run probability analysis if we have combos
        if in_deck:
            self._show_combo_probabilities_in(t, in_deck)

    def _fetch_combos(self, card_names):
        """Query Commander Spellbook's find-my-combos endpoint.
        
        Based on the confirmed working Combinator project pattern (Dec 2024):
        requests.get("https://backend.commanderspellbook.com/find-my-combos",
                     json=data).json()["results"]
        """
        payload = {
            "main": [{"card": name} for name in card_names],
            "commanders": [{"card": c.name} for c in self.cards if c.is_commander],
        }
        base_url = "https://backend.commanderspellbook.com/find-my-combos"
        attempts = []  # track status codes for error reporting

        # Attempt 1: GET with JSON body, NO extra headers (exact Combinator pattern)
        try:
            r = requests.get(base_url, json=payload, timeout=30)
            attempts.append(f"GET {base_url} → {r.status_code}")
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 400:
                # 400 = payload format wrong; capture response body for debugging
                try:
                    err_body = r.text[:300]
                except Exception:
                    err_body = "?"
                attempts[-1] += f" body={err_body}"
        except Exception as e:
            attempts.append(f"GET {base_url} → {e}")

        # Attempt 2: Same but with trailing slash (Django APPEND_SLASH)
        try:
            r = requests.get(base_url + "/", json=payload, timeout=30)
            attempts.append(f"GET {base_url}/ → {r.status_code}")
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            attempts.append(f"GET {base_url}/ → {e}")

        # Attempt 3: POST without trailing slash
        try:
            r = requests.post(base_url, json=payload, timeout=30)
            attempts.append(f"POST {base_url} → {r.status_code}")
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            attempts.append(f"POST {base_url} → {e}")

        # Attempt 4: POST with trailing slash
        try:
            r = requests.post(base_url + "/", json=payload, timeout=30)
            attempts.append(f"POST {base_url}/ → {r.status_code}")
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            attempts.append(f"POST {base_url}/ → {e}")

        # All attempts failed — report which status codes we got
        detail = " | ".join(attempts) if attempts else "No attempts made"
        return {"error": f"Commander Spellbook API error. Attempts: {detail}"}

    def _insert_combo_thumb(self, text_widget, card_name, missing=False):
        """Insert a small card thumbnail into a tk.Text widget at current position.
        Images load asynchronously; a placeholder is shown immediately.
        If missing=True, the border will be red to indicate the card is not in deck."""
        THUMB_W, THUMB_H = 80, 112  # small thumbnails

        # Create a placeholder image (solid color)
        border_color = "#ff4444" if missing else "#3a506b"
        ph_img = Image.new("RGB", (THUMB_W, THUMB_H), border_color)
        ph_photo = ImageTk.PhotoImage(ph_img)
        if not hasattr(self, '_combo_img_refs'):
            self._combo_img_refs = []
        self._combo_img_refs.append(ph_photo)

        # Insert placeholder into text widget
        text_widget.image_create(tk.END, image=ph_photo)
        # Save the index of the image so we can replace it
        img_index = text_widget.index(f"{tk.END}-2c")

        # Check PIL cache first
        if card_name in self.pil_cache:
            pil = self.pil_cache[card_name].resize((THUMB_W, THUMB_H), Image.LANCZOS)
            if missing:
                pil = self._add_missing_border(pil)
            photo = ImageTk.PhotoImage(pil)
            self._combo_img_refs.append(photo)
            try:
                text_widget.image_configure(img_index, image=photo)
            except tk.TclError:
                pass
            return

        # Load asynchronously
        safe_name = requests.utils.quote(card_name)
        url = f"https://api.scryfall.com/cards/named?format=image&exact={safe_name}&version=small"

        def go():
            try:
                resp = requests.get(url, headers={"User-Agent": "FasterFishing/1.0"}, timeout=15)
                if resp.status_code == 200:
                    pil_img = Image.open(io.BytesIO(resp.content))
                    self.pil_cache[card_name] = pil_img
                    resized = pil_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
                    if missing:
                        resized = self._add_missing_border(resized)
                    self.root.after(0, lambda: self._replace_combo_thumb(
                        text_widget, img_index, resized))
            except Exception:
                pass

        threading.Thread(target=go, daemon=True).start()

    def _replace_combo_thumb(self, text_widget, img_index, pil_img):
        """Replace a placeholder image in the text widget with the loaded image."""
        try:
            photo = ImageTk.PhotoImage(pil_img)
            if not hasattr(self, '_combo_img_refs'):
                self._combo_img_refs = []
            self._combo_img_refs.append(photo)
            text_widget.configure(state=tk.NORMAL)
            text_widget.image_configure(img_index, image=photo)
            text_widget.configure(state=tk.DISABLED)
        except tk.TclError:
            pass  # widget may have been destroyed

    def _add_missing_border(self, pil_img):
        """Add a red border to indicate a missing card."""
        from PIL import ImageDraw
        img = pil_img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i in range(3):  # 3px red border
            draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline="#ff4444")
        return img

    def _parse_combo(self, combo, my_cards):
        """Parse a single combo from Commander Spellbook API response."""
        if not isinstance(combo, dict):
            return None

        # Extract card names — try multiple known key patterns
        card_names = []
        uses = combo.get("uses", combo.get("cards", []))
        if isinstance(uses, list):
            for u in uses:
                if isinstance(u, dict):
                    # Nested: {"card": {"name": "Sol Ring"}} or {"card": "Sol Ring"} or {"name": "Sol Ring"}
                    card_data = u.get("card", u)
                    if isinstance(card_data, dict):
                        card_names.append(card_data.get("name", ""))
                    elif isinstance(card_data, str):
                        card_names.append(card_data)
                elif isinstance(u, str):
                    card_names.append(u)

        card_names = [n for n in card_names if n]
        if not card_names:
            return None

        card_names_lower = {n.lower() for n in card_names}
        missing = card_names_lower - my_cards

        combo_info = {
            "cards": card_names,
            "missing": list(missing),
            "results": [],
            "prereqs": [],
            "id": combo.get("id", ""),
        }

        # Extract results/effects
        produces = combo.get("produces", combo.get("results", []))
        if isinstance(produces, list):
            for p in produces:
                if isinstance(p, dict):
                    feat = p.get("feature", p)
                    name = feat.get("name", str(feat)) if isinstance(feat, dict) else str(feat)
                    combo_info["results"].append(name)
                elif isinstance(p, str):
                    combo_info["results"].append(p)

        # Extract prerequisites
        prereqs = combo.get("requires", combo.get("prerequisites", []))
        if isinstance(prereqs, list):
            for p in prereqs:
                if isinstance(p, dict):
                    tmpl = p.get("template", p)
                    name = tmpl.get("name", str(tmpl)) if isinstance(tmpl, dict) else str(tmpl)
                    combo_info["prereqs"].append(name)
                elif isinstance(p, str):
                    combo_info["prereqs"].append(p)

        return combo_info

    def _show_combo_probabilities_in(self, t, combos):
        """Calculate and display probability of assembling each combo by turn N.
        Uses Monte Carlo goldfish simulation with mana, ramp, draw, and tutor sequencing."""

        t.configure(state=tk.NORMAL)
        t.insert(tk.END, "\n")
        t.insert(tk.END, "═" * 80 + "\n")
        t.insert(tk.END, "COMBO ASSEMBLY PROBABILITY (Monte Carlo)\n", "header")

        deck_cards = self._get_deck_cards()
        deck_size = sum(c.quantity for c in deck_cards)

        N_SIMS = 5000
        MAX_TURN = 20
        turns_display = [3, 5, 7, 10, 15]

        t.insert(tk.END, f"  Deck: {deck_size} cards | {N_SIMS:,} simulated games per combo\n", "info")
        t.insert(tk.END, f"  Simulates: land drops, mana curve, ramp, draw spells, tutors\n\n", "info")

        t.insert(tk.END, f"  {'Combo':<40s} {'Pcs':>3s}  ", "info")
        for turn in turns_display:
            t.insert(tk.END, f"{'T'+str(turn):>6s}", "info")
        t.insert(tk.END, f"  {'~50% Turn':>9s}\n", "info")
        t.insert(tk.END, "  " + "-" * 95 + "\n")
        t.configure(state=tk.DISABLED)

        # Run simulations in background thread
        def run_all_combos():
            # Build caches once
            full_deck = SimEngine.build_deck(deck_cards)
            mana_cache = {}
            for c in deck_cards:
                if c.category == "Ramp" and c.name not in mana_cache:
                    mana_cache[c.name] = SimEngine._estimate_mana_produced(c)
            draw_cache = {}
            for c in deck_cards:
                if c.name not in draw_cache:
                    draw_cache[c.name] = SimEngine._estimate_card_draws(c)
            # Color production cache: card_name -> set of colors produced
            color_prod_cache = {}
            for c in deck_cards:
                if c.name not in color_prod_cache:
                    if c.category == "Land" or c.category == "Ramp":
                        color_prod_cache[c.name] = SimEngine._parse_color_production(c)
            # Color requirement cache: card_name -> dict of color pips needed
            color_req_cache = {}
            for c in deck_cards:
                if c.name not in color_req_cache:
                    color_req_cache[c.name] = SimEngine._parse_color_requirements(c.mana_cost)
            # Also cache commander color requirements
            for c in self.cards:
                if c.is_commander and c.name not in color_req_cache:
                    color_req_cache[c.name] = SimEngine._parse_color_requirements(c.mana_cost)
            # Build tutor info: name -> (subtype, destination)
            # destination: "hand", "top", "battlefield"
            tutor_cards = {}
            for c in deck_cards:
                oracle = (c.oracle_text or "").lower()
                if re.search(r'search\s+your\s+library', oracle):
                    subtype = getattr(c, '_tutor_subtype', None)
                    if subtype is None:
                        if re.search(r'search\s+your\s+library\s+for\s+a\s+card\b', oracle):
                            subtype = "any"
                        elif re.search(r'creature', oracle.split("search your library")[1][:30] if "search your library" in oracle else ""):
                            subtype = "creature"
                        elif re.search(r'artifact', oracle.split("search your library")[1][:30] if "search your library" in oracle else ""):
                            subtype = "artifact"
                        elif re.search(r'enchantment', oracle.split("search your library")[1][:30] if "search your library" in oracle else ""):
                            subtype = "enchantment"
                        elif re.search(r'(?:instant|sorcery)', oracle.split("search your library")[1][:30] if "search your library" in oracle else ""):
                            subtype = "instant_sorcery"
                        elif re.search(r'(?:basic\s+)?(?:land|forest|mountain|swamp|plains|island)', oracle.split("search your library")[1][:30] if "search your library" in oracle else ""):
                            subtype = "land"
                        else:
                            subtype = "any"
                    # Determine destination
                    dest = "hand"  # default
                    if re.search(r'(?:put|place)\s+(?:it|that card)\s+on\s+top', oracle):
                        dest = "top"
                    elif "to the top of your library" in oracle or "on top of your library" in oracle:
                        dest = "top"
                    elif "onto the battlefield" in oracle or "into play" in oracle:
                        dest = "battlefield"
                    elif "reveal" in oracle and "on top" in oracle:
                        dest = "top"
                    tutor_cards[c.name] = (subtype, dest)

            # Build combo piece type map (include commanders)
            piece_type_map = {}
            for c in deck_cards:
                tl = (c.type_line or "").lower()
                if "creature" in tl: piece_type_map[c.name.lower()] = "creature"
                elif "instant" in tl or "sorcery" in tl: piece_type_map[c.name.lower()] = "instant_sorcery"
                elif "artifact" in tl: piece_type_map[c.name.lower()] = "artifact"
                elif "enchantment" in tl: piece_type_map[c.name.lower()] = "enchantment"
                elif "land" in tl: piece_type_map[c.name.lower()] = "land"
                else: piece_type_map[c.name.lower()] = "other"
            # Also add commanders to piece_type_map (they're not in deck_cards)
            commander_names = {}  # name_lower -> cmc
            for c in self.cards:
                if c.is_commander:
                    commander_names[c.name.lower()] = c.cmc
                    tl = (c.type_line or "").lower()
                    if "creature" in tl: piece_type_map[c.name.lower()] = "creature"
                    elif "artifact" in tl: piece_type_map[c.name.lower()] = "artifact"
                    elif "enchantment" in tl: piece_type_map[c.name.lower()] = "enchantment"
                    else: piece_type_map[c.name.lower()] = "other"

            # ---- GRAVEYARD RECURSION DETECTION ----
            # Detect cards that can recover pieces from graveyard
            recursion_cards = {}  # card_name -> set of types it can recur ("any","creature", etc.)
            self_mill_cards = set()  # cards that mill yourself
            entomb_cards = {}  # cards that tutor to graveyard: name -> subtype

            # Check commanders for recursion ability
            for c in self.cards:
                if c.is_commander:
                    oracle = (c.oracle_text or "").lower()
                    recur_types = set()
                    if re.search(r'(?:cast|play)\s+(?:a\s+)?(?:creature\s+)?(?:card\s+)?from\s+(?:your\s+)?graveyard', oracle):
                        recur_types.add("any") if "card from" in oracle else recur_types.add("creature")
                    if re.search(r'return\s+(?:a\s+|target\s+)?(?:creature\s+)?card\s+from\s+(?:your\s+)?graveyard', oracle):
                        if "creature" in oracle.split("from")[0][-30:]:
                            recur_types.add("creature")
                        else:
                            recur_types.add("any")
                    if re.search(r'(?:put|return)\s+(?:it|that card|target\s+\w+\s+card)\s+from\s+(?:your\s+|a\s+)?graveyard\s+(?:onto|on|to)\s+the\s+battlefield', oracle):
                        recur_types.add("any")
                    if recur_types:
                        recursion_cards[c.name] = recur_types

            for c in deck_cards:
                oracle = (c.oracle_text or "").lower()
                # Recursion: return/cast from graveyard
                recur_types = set()
                if re.search(r'return\s+(?:a|target|up\s+to)\s+[^.]*?from\s+(?:your\s+|a\s+)?graveyard', oracle):
                    if "creature" in oracle.split("graveyard")[0][-30:]:
                        recur_types.add("creature")
                    elif "artifact" in oracle.split("graveyard")[0][-30:]:
                        recur_types.add("artifact")
                    else:
                        recur_types.add("any")
                if re.search(r'(?:cast|play)\s+(?:\w+\s+)?(?:cards?\s+)?from\s+(?:your\s+)?graveyard', oracle):
                    recur_types.add("any")
                if re.search(r'(?:put|return)\s+(?:it|that card|target\s+\w+\s+card)\s+from\s+(?:your\s+|a\s+)?graveyard\s+(?:onto|on|to)\s+the\s+battlefield', oracle):
                    recur_types.add("any")
                if recur_types:
                    recursion_cards[c.name] = recur_types

                # Self-mill: cards that put your own cards into graveyard
                if re.search(r'(?:mill|put the top)\s+\w+\s+cards?\s+(?:of your library\s+)?into\s+(?:your\s+)?graveyard', oracle):
                    self_mill_cards.add(c.name)
                elif "mill" in oracle and ("you" in oracle.split("mill")[0][-20:] or "your" in oracle):
                    self_mill_cards.add(c.name)
                elif re.search(r'mill\s+\w+', oracle) and "opponent" not in oracle and "target opponent" not in oracle:
                    self_mill_cards.add(c.name)

                # Entomb-style: search library and put into graveyard
                if re.search(r'search\s+your\s+library\s+for\s+(?:a\s+)?(?:\w+\s+)?card[^.]*?(?:put\s+(?:it|that card)\s+into\s+(?:your\s+)?graveyard)', oracle):
                    entomb_cards[c.name] = "any"

            has_recursion = len(recursion_cards) > 0

            for combo in combos[:20]:
                card_names = combo["cards"]
                piece_names_lower = {n.lower() for n in card_names}
                n_pieces = len(card_names)

                # Identify which combo pieces are commanders (always in command zone)
                cmdr_pieces = {}  # name_lower -> cmc for commander combo pieces
                non_cmdr_pieces = set()
                for pname in piece_names_lower:
                    if pname in commander_names:
                        cmdr_pieces[pname] = commander_names[pname]
                    else:
                        non_cmdr_pieces.add(pname)

                # Simulate N_SIMS games, track which turn all pieces assembled
                assembled_by_turn = [0] * (MAX_TURN + 1)

                for _sim in range(N_SIMS):
                    deck_copy = list(full_deck)
                    random.shuffle(deck_copy)
                    hand = deck_copy[:7]
                    lib = deck_copy[7:]
                    lands_in_play = 0
                    lands_played = []  # track actual land cards for color
                    ramp_in_play = []
                    battlefield = []
                    graveyard = []  # track graveyard zone
                    found_pieces = set()
                    cmdr_cast = set()
                    assembled_turn = None
                    recursion_on = False  # is recursion source active?

                    # Commander recursion is always available from command zone
                    for rn in recursion_cards:
                        if rn.lower() in commander_names:
                            recursion_on = True

                    # Check opening hand for non-commander pieces
                    for c in hand:
                        if c.name.lower() in non_cmdr_pieces:
                            found_pieces.add(c.name.lower())

                    # Commander pieces are always accessible — just need mana + colors
                    for cname, ccmc in cmdr_pieces.items():
                        if ccmc <= 0:
                            found_pieces.add(cname)
                            cmdr_cast.add(cname)

                    if len(found_pieces) >= n_pieces:
                        assembled_turn = 0

                    for turn in range(1, MAX_TURN + 1):
                        if assembled_turn is not None:
                            break

                        # Draw for turn (Commander: all players draw on turn 1)
                        if lib:
                            drawn = lib.pop(0)
                            hand.append(drawn)
                            if drawn.name.lower() in non_cmdr_pieces:
                                found_pieces.add(drawn.name.lower())

                        # Play a land (prefer lands that produce needed colors)
                        land_in_hand = [c for c in hand if c.category == "Land"]
                        if land_in_hand:
                            hand.remove(land_in_hand[0])
                            lands_in_play += 1
                            lands_played.append(land_in_hand[0])

                        # Build color mana pool from lands + ramp
                        mana_pool = {}  # color -> count, 'any' for wildcard
                        available_mana = 0
                        for lnd in lands_played:
                            lcolors = color_prod_cache.get(lnd.name, {'C'})
                            if 'any' in lcolors:
                                mana_pool['any'] = mana_pool.get('any', 0) + 1
                            elif len(lcolors) > 1:
                                # Multi-color land: treat as 'any' of those colors
                                mana_pool['any'] = mana_pool.get('any', 0) + 1
                            else:
                                for col in lcolors:
                                    mana_pool[col] = mana_pool.get(col, 0) + 1
                            available_mana += 1
                        for rc, cast_turn in ramp_in_play:
                            if turn > cast_turn:
                                mana_val = mana_cache.get(rc.name, 1)
                                available_mana += mana_val
                                rcolors = color_prod_cache.get(rc.name, {'C'})
                                if 'any' in rcolors:
                                    mana_pool['any'] = mana_pool.get('any', 0) + mana_val
                                else:
                                    for col in rcolors:
                                        mana_pool[col] = mana_pool.get(col, 0) + mana_val

                        # Check if commander combo pieces can be cast (mana + colors)
                        for cname, ccmc in cmdr_pieces.items():
                            if cname not in cmdr_cast and available_mana >= ccmc:
                                creqs = color_req_cache.get(cname, {})
                                if SimEngine._can_pay_colors(creqs, mana_pool):
                                    found_pieces.add(cname)
                                    cmdr_cast.add(cname)

                        # Cast ramp (cheapest first)
                        mana_left = available_mana
                        castable_ramp = sorted(
                            [c for c in hand if c.category == "Ramp" and 0 < c.cmc <= mana_left],
                            key=lambda x: x.cmc)
                        for rc in castable_ramp:
                            if rc.cmc <= mana_left:
                                hand.remove(rc)
                                mana_left -= int(rc.cmc)
                                ramp_in_play.append((rc, turn))
                                tl_r = (rc.type_line or "").lower()
                                if "creature" not in tl_r:
                                    mana_left += mana_cache.get(rc.name, 1)

                        # Re-check commander pieces after ramp (mana + colors may have changed)
                        total_mana_now = len(lands_played)
                        mana_pool_now = {}
                        for lnd in lands_played:
                            lc2 = color_prod_cache.get(lnd.name, {'C'})
                            if 'any' in lc2 or len(lc2) > 1:
                                mana_pool_now['any'] = mana_pool_now.get('any', 0) + 1
                            else:
                                for col in lc2:
                                    mana_pool_now[col] = mana_pool_now.get(col, 0) + 1
                        for rc2, ct2 in ramp_in_play:
                            tl2 = (rc2.type_line or "").lower()
                            mv2 = mana_cache.get(rc2.name, 1)
                            if "creature" in tl2:
                                if turn > ct2:
                                    total_mana_now += mv2
                                    rc2_colors = color_prod_cache.get(rc2.name, {'C'})
                                    if 'any' in rc2_colors:
                                        mana_pool_now['any'] = mana_pool_now.get('any', 0) + mv2
                                    else:
                                        for col in rc2_colors:
                                            mana_pool_now[col] = mana_pool_now.get(col, 0) + mv2
                            else:
                                total_mana_now += mv2
                                rc2_colors = color_prod_cache.get(rc2.name, {'C'})
                                if 'any' in rc2_colors:
                                    mana_pool_now['any'] = mana_pool_now.get('any', 0) + mv2
                                else:
                                    for col in rc2_colors:
                                        mana_pool_now[col] = mana_pool_now.get(col, 0) + mv2
                        for cname, ccmc in cmdr_pieces.items():
                            if cname not in cmdr_cast and total_mana_now >= ccmc:
                                creqs = color_req_cache.get(cname, {})
                                if SimEngine._can_pay_colors(creqs, mana_pool_now):
                                    found_pieces.add(cname)
                                    cmdr_cast.add(cname)

                        # Cast draw/tutor spells with remaining mana
                        castable_dt = sorted(
                            [c for c in hand if c.category in ("Draw", "Tutor")
                             and 0 < c.cmc <= mana_left],
                            key=lambda x: x.cmc)
                        for dc in castable_dt:
                            if dc.cmc <= mana_left:
                                hand.remove(dc)
                                mana_left -= int(dc.cmc)
                                dc_tl = (dc.type_line or "").lower()
                                dc_is_permanent = "instant" not in dc_tl and "sorcery" not in dc_tl

                                # Self-mill effect: mill top cards to graveyard
                                if dc.name in self_mill_cards:
                                    mill_n = 3  # default mill amount
                                    dc_oracle = (dc.oracle_text or "").lower()
                                    mm = re.search(r'mill\s+(\w+)', dc_oracle)
                                    if mm:
                                        mw = mm.group(1)
                                        mill_n = {"one":1,"two":2,"three":3,"four":4,"five":5}.get(mw, int(mw) if mw.isdigit() else 3)
                                    for _ in range(mill_n):
                                        if lib:
                                            milled = lib.pop(0)
                                            graveyard.append(milled)
                                            if milled.name.lower() in non_cmdr_pieces:
                                                if recursion_on:
                                                    found_pieces.add(milled.name.lower())

                                # Entomb-style: tutor card to graveyard
                                if dc.name in entomb_cards:
                                    for pname in card_names:
                                        if pname.lower() in found_pieces or pname.lower() in cmdr_pieces:
                                            continue
                                        for lc2 in lib:
                                            if lc2.name.lower() == pname.lower():
                                                lib.remove(lc2)
                                                graveyard.append(lc2)
                                                if recursion_on:
                                                    found_pieces.add(pname.lower())
                                                break
                                        break

                                # Draw spell
                                dn, rep, _, _lc = draw_cache.get(dc.name, (0, False, "", 0))
                                if dn > 0 and not rep:
                                    for _ in range(int(dn)):
                                        if lib:
                                            drawn = lib.pop(0)
                                            hand.append(drawn)
                                            if drawn.name.lower() in non_cmdr_pieces:
                                                found_pieces.add(drawn.name.lower())
                                # Tutor: search library for a combo piece
                                if dc.name in tutor_cards:
                                    tsub, tdest = tutor_cards[dc.name]
                                    # Find an unfound non-commander combo piece this tutor can fetch
                                    for pname in card_names:
                                        if pname.lower() in found_pieces:
                                            continue
                                        if pname.lower() in cmdr_pieces:
                                            continue  # can't tutor for commander (command zone)
                                        ptype = piece_type_map.get(pname.lower(), "other")
                                        can_find = (tsub == "any" or tsub == ptype
                                                    or (tsub == "artifact_enchantment"
                                                        and ptype in ("artifact", "enchantment")))
                                        if can_find:
                                            for lc2 in lib:
                                                if lc2.name.lower() == pname.lower():
                                                    lib.remove(lc2)
                                                    if tdest == "top":
                                                        # Tutor to top: insert at position 0 (next draw)
                                                        lib.insert(0, lc2)
                                                        # Don't mark as found yet — need to draw it
                                                    elif tdest == "battlefield":
                                                        # Tutor to battlefield: it's in play, count as found
                                                        found_pieces.add(pname.lower())
                                                        battlefield.append(lc2)
                                                    else:
                                                        # Tutor to hand: immediately available
                                                        hand.append(lc2)
                                                        found_pieces.add(pname.lower())
                                                    break
                                            break  # one tutor finds one card
                                if rep and dn > 0:
                                    battlefield.append(dc)
                                elif dc_is_permanent:
                                    battlefield.append(dc)
                                else:
                                    graveyard.append(dc)  # instant/sorcery → graveyard

                        # Repeating draw engines on battlefield
                        for perm in battlefield:
                            dn, rep, _, _lc = draw_cache.get(perm.name, (0, False, "", 0))
                            if rep and dn > 0:
                                for _ in range(int(dn)):
                                    if lib:
                                        drawn = lib.pop(0)
                                        hand.append(drawn)
                                        if drawn.name.lower() in non_cmdr_pieces:
                                            found_pieces.add(drawn.name.lower())

                        # ---- GRAVEYARD RECURSION CHECK ----
                        # Check if any recursion sources are now on battlefield
                        if not recursion_on:
                            for perm in battlefield:
                                if perm.name in recursion_cards:
                                    recursion_on = True
                                    break
                        # If recursion is available, pieces in graveyard count as found
                        if recursion_on and has_recursion:
                            for gc in graveyard:
                                if gc.name.lower() in non_cmdr_pieces and gc.name.lower() not in found_pieces:
                                    gtype = piece_type_map.get(gc.name.lower(), "other")
                                    # Check if any recursion source can recur this type
                                    for rn, rtypes in recursion_cards.items():
                                        if "any" in rtypes or gtype in rtypes:
                                            found_pieces.add(gc.name.lower())
                                            break

                        # Check if assembled
                        if len(found_pieces) >= n_pieces:
                            assembled_turn = turn

                    # Record result
                    if assembled_turn is not None and assembled_turn <= MAX_TURN:
                        for tt in range(assembled_turn, MAX_TURN + 1):
                            assembled_by_turn[tt] += 1

                # Convert to probabilities
                probs_by_turn = [assembled_by_turn[tt] / N_SIMS for tt in range(MAX_TURN + 1)]

                # Display result for this combo
                label = " + ".join(n[:18] for n in card_names)
                if len(label) > 38:
                    label = label[:35] + "..."

                def show_combo_row(lbl, n_pcs, probs):
                    t.configure(state=tk.NORMAL)
                    t.insert(tk.END, f"  {lbl:<40s} {n_pcs:>3d}  ", "combo")
                    ft = None
                    for turn in turns_display:
                        p = probs[turn] if turn <= MAX_TURN else probs[MAX_TURN]
                        if p >= 0.9: tag = "good"
                        elif p >= 0.5: tag = "sub"
                        elif p >= 0.2: tag = "info"
                        else: tag = "warn"
                        t.insert(tk.END, f"{p*100:>5.1f}%", tag)
                        t.insert(tk.END, " ")
                        if ft is None and p >= 0.5:
                            ft = turn
                    # Find exact 50% turn
                    if ft is None:
                        for et in range(1, MAX_TURN + 1):
                            if probs[et] >= 0.5:
                                ft = et
                                break
                    if ft is not None:
                        t.insert(tk.END, f" {'~T'+str(ft):>9s}\n", "sub")
                    else:
                        t.insert(tk.END, f" {'  T20+':>9s}\n", "warn")
                    t.configure(state=tk.DISABLED)

                self.root.after(0, lambda l=label, n=n_pieces, p=probs_by_turn: show_combo_row(l, n, p))

            # Legend
            def show_legend():
                t.configure(state=tk.NORMAL)
                t.insert(tk.END, f"\n  Probability = % of {N_SIMS:,} games where ALL pieces were in hand by that turn\n", "info")
                t.insert(tk.END, "  Simulates: land drops → ramp (with mana cost) → draw spells → tutors (type-aware)\n", "info")
                t.insert(tk.END, "  ~50% Turn = earliest turn with ≥50% assembly chance\n", "info")
                t.configure(state=tk.DISABLED)
                self._sts("Combo probability simulation complete!")
            self.root.after(0, show_legend)

        threading.Thread(target=run_all_combos, daemon=True).start()

    # ---- EDHREC RECOMMENDATIONS (Feature 11) ----
    def _run_edhrec(self):
        cmdrs = [c for c in self.cards if c.is_commander]
        if not cmdrs:
            messagebox.showwarning("No Commander", "Set a commander first!"); return
        self.edhrec_btn.configure(state=tk.DISABLED)
        cmdr = cmdrs[0]
        self._sts(f"Fetching EDHREC data for {cmdr.name}...")
        self.analysis_prog["value"] = 10

        def go():
            recs = self._fetch_edhrec(cmdr.name)
            self.root.after(0, lambda: self._show_edhrec(recs, cmdr.name))

        threading.Thread(target=go, daemon=True).start()

    def _fetch_edhrec(self, commander_name):
        """Fetch EDHREC recommendation data for a commander."""
        # EDHREC uses slugified names: "Atraxa, Praetors' Voice" -> "atraxa-praetors-voice"
        slug = re.sub(r"[^a-z0-9]+", "-", commander_name.lower()).strip("-")
        urls = [
            f"https://json.edhrec.com/pages/commanders/{slug}.json",
            # Try alternate slug without possessives
            f"https://json.edhrec.com/pages/commanders/{slug.replace('-s-', '-')}.json",
        ]
        headers = {"User-Agent": "FasterFishing/1.0", "Accept": "application/json"}

        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                continue
        return {"error": f"Could not find EDHREC data for '{commander_name}' (tried slug: {slug})"}

    def _show_edhrec(self, data, cmdr_name):
        self.edhrec_btn.configure(state=tk.NORMAL)
        self.analysis_prog["value"] = 100
        t = self.analysis_text; t.configure(state=tk.NORMAL)

        t.insert(tk.END, "\n\n")
        t.insert(tk.END, "═" * 80 + "\n")
        t.insert(tk.END, f"EDHREC RECOMMENDATIONS — {cmdr_name}\n", "header")

        if "error" in data:
            t.insert(tk.END, f"  Error: {data['error']}\n", "warn")
            t.insert(tk.END, "  Note: json.edhrec.com must be reachable from your machine.\n", "info")
            t.configure(state=tk.DISABLED)
            self._sts("EDHREC fetch failed.")
            return

        my_cards = {c.name.lower() for c in self.cards}

        # Parse the EDHREC JSON structure
        container = data.get("container", data)
        card_lists = container.get("json_dict", container).get("cardlists", [])

        total_decks = container.get("json_dict", {}).get("num_decks",
                      data.get("num_decks", "?"))
        t.insert(tk.END, f"Based on {total_decks} decks on EDHREC\n\n", "info")

        for cardlist in card_lists:
            header = cardlist.get("header", "Cards")
            cards = cardlist.get("cardviews", [])
            if not cards: continue

            # Filter to only cards NOT already in our deck
            new_recs = []
            in_deck_recs = []
            for cv in cards:
                name = cv.get("name", "")
                if not name: continue
                inclusion = cv.get("inclusion", cv.get("num_decks", 0))
                synergy = cv.get("synergy", 0)
                salt = cv.get("salt", 0)
                price = cv.get("price", 0)
                primary_type = cv.get("primary_type", "")
                cmc_val = cv.get("cmc", 0)
                info = {
                    "name": name, "inclusion": inclusion,
                    "synergy": synergy, "salt": salt,
                    "price": price, "type": primary_type, "cmc": cmc_val,
                }
                if name.lower() in my_cards:
                    in_deck_recs.append(info)
                else:
                    new_recs.append(info)

            if not new_recs and not in_deck_recs: continue

            t.insert(tk.END, f"\n  {header}", "sub")
            if in_deck_recs:
                t.insert(tk.END, f" ({len(in_deck_recs)} in deck, ", "info")
                t.insert(tk.END, f"{len(new_recs)} suggestions)\n", "rec")
            else:
                t.insert(tk.END, f" ({len(new_recs)} suggestions)\n", "rec")

            # Show top suggestions NOT in deck
            for rec in new_recs[:10]:
                syn_str = f"+{rec['synergy']:.0%}" if rec['synergy'] > 0 else f"{rec['synergy']:.0%}"
                inc_pct = rec['inclusion']
                if isinstance(inc_pct, (int, float)) and inc_pct > 1:
                    inc_pct = f"{inc_pct} decks"
                elif isinstance(inc_pct, (int, float)):
                    inc_pct = f"{inc_pct:.0%}"
                price_str = f"${rec['price']:.2f}" if rec['price'] else ""
                t.insert(tk.END, f"    + {rec['name']:<30s}", "rec")
                t.insert(tk.END, f" Synergy: {syn_str:<7s} In: {inc_pct:<12s} {price_str}\n")

            if len(new_recs) > 10:
                t.insert(tk.END, f"    ... and {len(new_recs)-10} more suggestions\n", "info")

        # Combo data from EDHREC
        combos_data = container.get("json_dict", {}).get("combos", [])
        if combos_data:
            t.insert(tk.END, f"\n  EDHREC Combos ({len(combos_data)}):\n", "sub")
            for combo in combos_data[:10]:
                if isinstance(combo, dict):
                    cards_list = combo.get("cards", combo.get("value", ""))
                    if isinstance(cards_list, list):
                        cards_str = " + ".join(cards_list)
                    else:
                        cards_str = str(cards_list)
                    t.insert(tk.END, f"    {cards_str}\n", "combo")

        t.insert(tk.END, f"\n  View full page: https://edhrec.com/commanders/{re.sub(r'[^a-z0-9]+', '-', cmdr_name.lower()).strip('-')}\n", "info")

        t.configure(state=tk.DISABLED)
        self._sts(f"EDHREC data loaded for {cmdr_name}.")
def main():
    root = tk.Tk()
    FasterFishing(root)
    root.mainloop()

if __name__ == "__main__":
    main()