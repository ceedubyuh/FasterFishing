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
}
ALL_CATEGORIES = ["Land", "Ramp", "Draw", "Removal", "Board Wipe", "Creature", "Other"]
CATEGORY_COLORS = {
    "Land":"#8B7355", "Ramp":"#2E8B57", "Draw":"#4169E1",
    "Removal":"#DC143C", "Board Wipe":"#FF4500", "Creature":"#DAA520", "Other":"#708090",
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
    def sim_hands(cards, n=10000, hs=7, pcb=None, min_mull=4, land_min=2, land_max=5,
                  commander_cmc=0):
        """Run opening hand simulation with mulligan support.
        
        Args:
            cards: list of Card objects (non-commander deck)
            n: number of simulations
            hs: starting hand size (usually 7)
            min_mull: minimum hand size to mulligan to
            land_min/land_max: range for a keepable hand
            commander_cmc: CMC of commander (0 = no commander cast tracking)
        """
        deck = SimEngine.build_deck(cards)
        results = {}  # hand_size -> SimResult

        for mull_sz in range(hs, min_mull - 1, -1):
            r = SimResult(num_sims=n, hand_size=mull_sz)
            if len(deck) < mull_sz:
                results[mull_sz] = r; continue
            lc = Counter(); ct = {c: 0 for c in ALL_CATEGORIES}
            cd = {c: Counter() for c in ALL_CATEGORIES}; keep = 0
            cmdr_turns = []
            for i in range(n):
                hand = random.sample(deck, mull_sz)
                hc = Counter(c.category for c in hand)
                l = hc.get("Land", 0); lc[l] += 1
                if land_min <= l <= land_max: keep += 1
                for cat in ALL_CATEGORIES:
                    x = hc.get(cat, 0); ct[cat] += x; cd[cat][x] += 1
                # Commander cast turn estimate: simulate draws to find turn with enough lands
                if commander_cmc > 0:
                    # Simulate a goldfish from this opening hand to find earliest cast turn
                    remaining = [c for c in deck if c not in hand]
                    random.shuffle(remaining)
                    lands_played = 0; turn = 0
                    hand_copy = list(hand)
                    lib = list(remaining)
                    while turn < 20:
                        turn += 1
                        # Draw for the turn (skip turn 1 on the play)
                        if turn > 1 and lib:
                            hand_copy.append(lib.pop(0))
                        # Play a land if we have one
                        land_in_hand = [c for c in hand_copy if c.category == "Land"]
                        if land_in_hand:
                            hand_copy.remove(land_in_hand[0])
                            lands_played += 1
                        # Count ramp already "played" (simplified: each ramp = +1 mana)
                        ramp_in_hand = [c for c in hand_copy if c.category == "Ramp"
                                        and c.cmc <= lands_played]
                        ramp_mana = 0
                        for rc in ramp_in_hand:
                            if rc.cmc <= lands_played - ramp_mana:
                                ramp_mana += 1
                        if lands_played + ramp_mana >= commander_cmc:
                            break
                    cmdr_turns.append(turn)

                if pcb and i % 1000 == 0: pcb(i / n)
            r.land_dist = {k: v / n * 100 for k, v in sorted(lc.items())}
            r.cat_avgs = {c: t / n for c, t in ct.items()}
            r.cat_dists = {c: {k: v / n * 100 for k, v in sorted(d.items())} for c, d in cd.items()}
            r.keepable = keep / n * 100
            r.avg_cmdr_turn = sum(cmdr_turns) / len(cmdr_turns) if cmdr_turns else 0
            results[mull_sz] = r
        if pcb: pcb(1.0)
        return results

    @staticmethod
    def _estimate_card_draws(card):
        """Estimate how many bonus cards a card draws when played.
        Returns (draw_count, is_repeating) where is_repeating means it draws each turn."""
        oracle = card.oracle_text.lower() if card.oracle_text else ""
        if not oracle or "draw" not in oracle: return (0, False)

        NUMBER_WORDS = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,
                        "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        def parse_n(text):
            m = re.search(r'draws?\s+(\w+)\s+cards?', text)
            if m:
                w = m.group(1)
                if w in NUMBER_WORDS: return NUMBER_WORDS[w]
                if w.isdigit(): return int(w)
            if "draw cards equal to" in text: return 2
            if "draw a card" in text: return 1
            return 0

        tl = card.type_line.lower() if card.type_line else ""
        is_permanent = "instant" not in tl and "sorcery" not in tl

        # Upkeep/repeating draw (Phyrexian Arena, Sylvan Library)
        if re.search(r'(?:at the beginning of|during) your (?:upkeep|draw step)[^.]*?draw', oracle):
            n = parse_n(oracle)
            if "two additional" in oracle: n = max(n, 2)
            return (max(n, 1), True)  # repeating

        # Opponent-cast triggers (Rhystic Study ~1-2/turn)
        if re.search(r'whenever\s+an?\s+opponent\s+casts[^.]*?draw', oracle):
            if "unless" in oracle or "may pay" in oracle:
                return (1, True)  # ~1 draw/turn conservatively
            return (2, True)

        # "Whenever you cast" triggers on permanents (Beast Whisperer) — ~1 per turn avg
        if is_permanent and re.search(r'whenever\s+you\s+cast[^.]*?draw', oracle):
            return (1, True)

        # Activated abilities: "{T}: draw" — 1 per turn
        if re.search(r'\{t\}[^:]*:\s*[^.]*?draw', oracle):
            return (parse_n(oracle) or 1, True)

        # ETB draw
        if re.search(r'when\s+.{0,30}\s+enters?[^.]*?draw', oracle):
            return (parse_n(oracle) or 1, False)

        # Instant/sorcery cantrip or draw spell
        if not is_permanent:
            n = parse_n(oracle)
            # Subtract put-back (Brainstorm: draw 3 put 2 = net 1)
            putback = re.search(r'put\s+(\w+)\s+(?:of them|cards?)\s+(?:back\s+)?(?:on\s+top|from your hand)', oracle)
            if putback:
                pb_word = putback.group(1)
                pb = NUMBER_WORDS.get(pb_word, int(pb_word) if pb_word.isdigit() else 0)
                n = max(0, n - pb)
            return (n, False)

        return (0, False)

    @staticmethod
    def sim_goldfish(cards, n=1000, turns=10, pcb=None):
        deck = SimEngine.build_deck(cards)
        if len(deck) < 7 + turns: return {}
        td = {t:{c:0 for c in ALL_CATEGORIES} for t in range(turns+1)}
        ld = {t:0 for t in range(turns+1)}
        # New: track bonus draws and total cards in hand per turn
        bonus_draws = {t: 0 for t in range(turns+1)}
        total_cards = {t: 0 for t in range(turns+1)}

        # Pre-compute draw estimates for each unique card
        draw_cache = {}
        for c in cards:
            if c.name not in draw_cache:
                draw_cache[c.name] = SimEngine._estimate_card_draws(c)

        for i in range(n):
            sh = random.sample(deck, len(deck))
            hand = list(sh[:7]); lib = list(sh[7:])
            battlefield = []; mana = 0; ramp_mana = 0
            turn_bonus = 0  # bonus draws this game for turn 0

            cs = Counter(c.category for c in hand)
            for cat in ALL_CATEGORIES: td[0][cat] += cs.get(cat, 0)
            ld[0] += min(cs.get("Land", 0), 1)
            total_cards[0] += len(hand)

            for t in range(1, turns+1):
                # Draw for turn
                if lib: hand.append(lib.pop(0))

                # Simplified play: play a land, cast what we can
                # Calculate mana: lands on bf + ramp
                mana = sum(1 for c in battlefield if c.category == "Land") + \
                       sum(1 for c in battlefield if c.category == "Ramp")
                spent = 0

                # Play a land
                land_in_hand = [c for c in hand if c.category == "Land"]
                if land_in_hand:
                    chosen = land_in_hand[0]
                    hand.remove(chosen); battlefield.append(chosen)
                    mana += 1  # immediate mana from the land we just played

                # Cast spells from hand, cheapest draw/ramp first, then others
                # Sort: draw and ramp first (by CMC), then others by CMC
                castable = sorted(hand, key=lambda c: (
                    0 if c.category == "Draw" else 1 if c.category == "Ramp" else 2, c.cmc))

                cards_to_cast = []
                for card in castable:
                    if card.category == "Land": continue
                    cost = int(card.cmc)
                    if cost <= mana - spent and cost >= 0:
                        cards_to_cast.append(card)
                        spent += cost

                for card in cards_to_cast:
                    hand.remove(card)
                    battlefield.append(card)
                    draws_n, repeating = draw_cache.get(card.name, (0, False))
                    # One-shot draw (ETB, cantrip)
                    if draws_n > 0 and not repeating:
                        for _ in range(draws_n):
                            if lib:
                                hand.append(lib.pop(0))
                                turn_bonus += 1

                # Repeating draw from permanents already on battlefield
                for perm in battlefield:
                    draws_n, repeating = draw_cache.get(perm.name, (0, False))
                    if repeating and draws_n > 0:
                        # Only trigger if it was on bf before this turn (not just cast)
                        if perm not in cards_to_cast:
                            for _ in range(draws_n):
                                if lib:
                                    hand.append(lib.pop(0))
                                    turn_bonus += 1

                # Record stats: count all cards seen (hand + battlefield)
                all_seen = hand + battlefield
                cs = Counter(c.category for c in all_seen)
                for cat in ALL_CATEGORIES: td[t][cat] += cs.get(cat, 0)
                ld[t] += min(sum(1 for c in battlefield if c.category == "Land"), t + 1)
                bonus_draws[t] += turn_bonus
                total_cards[t] += len(all_seen)

            if pcb and i % 100 == 0: pcb(i / n)

        for t in td:
            for cat in td[t]: td[t][cat] /= n
            ld[t] /= n
            bonus_draws[t] /= n
            total_cards[t] /= n
        if pcb: pcb(1.0)

        # Count draw sources in deck
        draw_sources = []
        for c in cards:
            dn, rep = draw_cache.get(c.name, (0, False))
            if dn > 0:
                draw_sources.append({"name": c.name, "qty": c.quantity, "draws": dn,
                                      "repeating": rep})

        return {"turn_avgs": td, "land_drops": ld, "bonus_draws": bonus_draws,
                "total_cards": total_cards, "draw_sources": draw_sources}

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
        for perm in self.battlefield:
            oracle = perm.oracle_text.lower() if perm.oracle_text else ""
            if not oracle or "draw" not in oracle: continue

            # Pattern: "{T}: draw a card" or "{N}, {T}: draw a card"
            # Also: "{T}: draw" without mana cost
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

            # Pattern: "{N}: draw a card" (no tap, repeatable — like Thrasios)
            notap_draw = re.search(r'\{(\d+)\}[^{]*?:\s*[^.]*?draw[^.]*', oracle)
            if notap_draw and not tap_draw:
                trigger_text = notap_draw.group(0)
                n = self._parse_draw_count(trigger_text)
                mana_cost = int(notap_draw.group(1))
                if n > 0:
                    # Activate as many times as we can afford
                    avail = self.mana_available - self.mana_spent
                    activations = avail // max(mana_cost, 1) if mana_cost > 0 else 1
                    activations = min(activations, 3)  # cap at 3 to avoid infinite
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
class MTGGoldfishApp:
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
            ("  Deck & Categories  ", self._ui_deck), ("  Simulations  ", self._ui_sim),
            ("  Goldfish Turns  ", self._ui_gf)]:
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
        tb = ttk.Frame(left); tb.pack(fill=tk.X, pady=(0,5))
        ttk.Label(tb, text="Deck", style="H.TLabel").pack(side=tk.LEFT)
        self.dcnt = tk.StringVar(value="0 cards"); ttk.Label(tb, textvariable=self.dcnt).pack(side=tk.LEFT, padx=10)
        ttk.Button(tb, text="Re-Categorize (Scryfall Tags)", command=self._recat).pack(side=tk.RIGHT, padx=5)
        ttk.Button(tb, text="Remove Selected", command=self._remove_card).pack(side=tk.RIGHT, padx=5)
        ttk.Button(tb, text="Add from Scryfall", command=self._open_add_card_dialog).pack(side=tk.RIGHT, padx=5)
        # Debounce tracking for add card dialog
        self._add_search_after_id = None
        cols = ("cmdr","qty","name","category","cmc","type")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=25)
        self._sort_col = "name"; self._sort_rev = False
        for c, w, a, display in [("cmdr",30,tk.CENTER,"*"),("qty",40,tk.CENTER,"Qty"),
                      ("name",200,tk.W,"Name"),("category",100,tk.CENTER,"Category"),
                      ("cmc",50,tk.CENTER,"CMC"),("type",180,tk.W,"Type")]:
            self.tree.heading(c, text=display, command=lambda col=c: self._sort_tree(col))
            self.tree.column(c, width=w, anchor=a)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right = ttk.Frame(p); p.add(right, weight=2)
        self.img_label = ttk.Label(right, text="Select a card to preview"); self.img_label.pack(pady=10)

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
        self.cat_frame = cf  # store reference for dynamic category adds
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
        ttk.Button(row1, text="Play Goldfish", command=self._show_goldfish_game).pack(side=tk.RIGHT, padx=5)
        ttk.Button(row1, text="Draw Sample Hand", command=self._show_hand).pack(side=tk.RIGHT, padx=5)

        # Row 2: Mulligan and Land range
        row2 = ttk.Frame(ctrl); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Mulligan down to:").pack(side=tk.LEFT)
        self.min_mull = tk.StringVar(value="4"); ttk.Entry(row2, textvariable=self.min_mull, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="(Commander free mull at 7)").pack(side=tk.LEFT, padx=(0,20))

        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        ttk.Label(row2, text="Ideal lands in hand:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Label(row2, text="Min:").pack(side=tk.LEFT)
        self.land_min = tk.StringVar(value=""); ttk.Entry(row2, textvariable=self.land_min, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Label(row2, text="Max:").pack(side=tk.LEFT)
        self.land_max = tk.StringVar(value=""); ttk.Entry(row2, textvariable=self.land_max, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Label(row2, text="(blank = auto from commander CMC)").pack(side=tk.LEFT, padx=(5,0))

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
        self.gf_prog = ttk.Progressbar(ctrl, mode="determinate", length=500); self.gf_prog.pack(fill=tk.X, pady=5)
        self.gf_text = tk.Text(f, font=("Consolas",10), bg="#16213e", fg="#e0e0e0", state=tk.DISABLED, wrap=tk.WORD)
        self.gf_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.gf_text.tag_configure("header", foreground="#e94560", font=("Consolas",11,"bold"))
        self.gf_text.tag_configure("info", foreground="#00d2ff")
        self.gf_text.tag_configure("good", foreground="#2ECC71")

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
            self._sts("Deck fetched! Click 'Import Text' to load with Scryfall data.")
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
        # Apply current sort
        sorted_cards = list(enumerate(self.cards))
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
        sorted_cards.sort(key=sort_key, reverse=rev)

        for orig_idx, c in sorted_cards:
            cmdr_mark = "CMDR" if c.is_commander else ""
            self.tree.insert("", tk.END, iid=str(orig_idx),
                values=(cmdr_mark, c.quantity, c.name, c.category, int(c.cmc), c.type_line))
        total = sum(c.quantity for c in self.cards)
        n_cmdrs = sum(1 for c in self.cards if c.is_commander)
        deck_size = sum(c.quantity for c in self.cards if not c.is_commander)
        self.dcnt.set(f"{deck_size} in deck + {n_cmdrs} commander(s) ({len(self.cards)} unique)")
        self._update_cmdr_info()
        # Update heading arrows
        for c_name in ("cmdr","qty","name","category","cmc","type"):
            display = {"cmdr":"*","qty":"Qty","name":"Name","category":"Category",
                       "cmc":"CMC","type":"Type"}[c_name]
            arrow = ""
            if c_name == col:
                arrow = " ▼" if rev else " ▲"
            self.tree.heading(c_name, text=display + arrow)

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
        idx = int(sel[0])
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
        idx = int(sel[0]); self.sel_idx = idx; card = self.cards[idx]; self.cat_var.set(card.category)
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
        """Get land range, auto-calculating from commander CMC if not specified."""
        cmdrs = [c for c in self.cards if c.is_commander]
        cmdr_cmc = max((int(c.cmc) for c in cmdrs), default=0)

        # Parse user input or auto-calculate
        try: lmin = int(self.land_min.get())
        except (ValueError, AttributeError):
            # Auto from commander CMC
            if cmdr_cmc <= 3: lmin = 1
            elif cmdr_cmc <= 6: lmin = 3
            else: lmin = 3
        try: lmax = int(self.land_max.get())
        except (ValueError, AttributeError):
            if cmdr_cmc <= 3: lmax = 3
            elif cmdr_cmc <= 6: lmax = 5
            else: lmax = 5
        return max(0, lmin), max(lmin, lmax), cmdr_cmc

    def _run_sim(self):
        if not self.cards: messagebox.showwarning("No Deck", "Import first!"); return
        deck_cards = self._get_deck_cards()
        if not deck_cards: messagebox.showwarning("Empty Deck", "All cards are set as commander!"); return
        try: ns = int(self.sim_n.get()); hs = int(self.hand_sz.get())
        except ValueError: messagebox.showerror("Error", "Enter valid numbers."); return
        try: min_mull = int(self.min_mull.get())
        except ValueError: min_mull = 4
        min_mull = max(1, min(hs, min_mull))
        land_min, land_max, cmdr_cmc = self._get_land_range()

        self.sim_btn.configure(state=tk.DISABLED); self.sim_prog["value"] = 0
        self._sts(f"Running {ns:,} simulations (hands {hs} down to {min_mull})...")
        def pcb(p): self.root.after(0, lambda: self.sim_prog.configure(value=p*100))
        def go():
            results = SimEngine.sim_hands(deck_cards, ns, hs, pcb,
                                          min_mull=min_mull, land_min=land_min,
                                          land_max=land_max, commander_cmc=cmdr_cmc)
            self.root.after(0, lambda: self._show_sim(results, land_min, land_max, cmdr_cmc))
        threading.Thread(target=go, daemon=True).start()

    def _show_sim(self, results, land_min, land_max, cmdr_cmc):
        self.sim_btn.configure(state=tk.NORMAL); self._sts("Simulation complete!")
        t = self.sim_text; t.configure(state=tk.NORMAL); t.delete("1.0", tk.END)

        # Show settings
        t.insert(tk.END, "OPENING HAND SIMULATION\n", "header")
        cmdrs = [c for c in self.cards if c.is_commander]
        if cmdrs:
            cmdr_names = ", ".join(f"{c.name} (CMC {int(c.cmc)})" for c in cmdrs)
            t.insert(tk.END, f"Commander: {cmdr_names}\n", "info")
        t.insert(tk.END, f"Keepable range: {land_min}-{land_max} lands in hand\n\n")

        # Get the primary (7-card) result for the chart
        primary_sz = max(results.keys())
        primary = results[primary_sz]

        # Commander cast turn
        if cmdr_cmc > 0 and primary.avg_cmdr_turn > 0:
            t.insert(tk.END, "COMMANDER CAST TURN\n", "header")
            t.insert(tk.END, f"  Avg earliest cast turn: ", "info")
            turn = primary.avg_cmdr_turn
            tg = "good" if turn <= cmdr_cmc else "warn" if turn <= cmdr_cmc + 2 else "bad"
            t.insert(tk.END, f"Turn {turn:.1f}\n", tg)
            t.insert(tk.END, f"  (based on 7-card hands, land drops + ramp)\n\n")

        # Mulligan summary table
        if len(results) > 1:
            t.insert(tk.END, "MULLIGAN ANALYSIS\n", "header")
            t.insert(tk.END, f"  {'Hand':>6s}  {'Keepable':>10s}  {'Avg Lands':>10s}  {'Avg Ramp':>10s}", "info")
            if cmdr_cmc > 0:
                t.insert(tk.END, f"  {'Cmdr Turn':>10s}", "info")
            t.insert(tk.END, "\n")
            t.insert(tk.END, "  " + "-" * (50 + (12 if cmdr_cmc > 0 else 0)) + "\n")
            for sz in sorted(results.keys(), reverse=True):
                r = results[sz]
                free = " (free)" if sz == primary_sz else ""
                tag = "good" if r.keepable >= 80 else "warn" if r.keepable >= 60 else "bad"
                line = f"  {sz:>4d}{free:>6s}  {r.keepable:>9.1f}%  {r.cat_avgs.get('Land',0):>10.2f}  {r.cat_avgs.get('Ramp',0):>10.2f}"
                if cmdr_cmc > 0:
                    line += f"  {r.avg_cmdr_turn:>10.1f}"
                t.insert(tk.END, line + "\n", tag)
            t.insert(tk.END, "\n")

        # Detailed results for primary hand size
        r = primary
        t.insert(tk.END, f"{r.num_sims:,} hands of {r.hand_size} cards\n", "header")
        tag = "good" if r.keepable >= 80 else "warn" if r.keepable >= 60 else "bad"
        t.insert(tk.END, f"Keepable Hands ({land_min}-{land_max} lands): ", "info")
        t.insert(tk.END, f"{r.keepable:.1f}%\n\n", tag)

        t.insert(tk.END, "LANDS IN OPENING HAND\n", "header")
        for lands, pct in r.land_dist.items():
            bar = "#" * int(pct / 2)
            tg = "good" if land_min <= lands <= land_max else "warn" if abs(lands - land_min) <= 1 or abs(lands - land_max) <= 1 else "bad"
            t.insert(tk.END, f"  {lands} lands: ", "info"); t.insert(tk.END, f"{pct:5.1f}%  {bar}\n", tg)

        t.insert(tk.END, "\nAVG CARDS PER CATEGORY IN HAND\n", "header")
        for cat in ALL_CATEGORIES:
            avg = r.cat_avgs.get(cat, 0)
            if avg > 0.01: t.insert(tk.END, f"  {cat:12s}: {avg:.2f}\n")
        for key in ["Land", "Ramp", "Draw", "Removal"]:
            dist = r.cat_dists.get(key, {})
            if dist:
                t.insert(tk.END, f"\n{key.upper()} COUNT DISTRIBUTION\n", "header")
                for cnt, pct in dist.items():
                    if pct > 0.1: t.insert(tk.END, f"  {cnt} {key.lower():10s}: {pct:5.1f}%  {'#' * int(pct / 2)}\n")
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
        def pcb(p): self.root.after(0, lambda: self.gf_prog.configure(value=p*100))
        def go():
            r = SimEngine.sim_goldfish(deck_cards, ns, nt, pcb)
            self.root.after(0, lambda: self._show_gf(r, nt))
        threading.Thread(target=go, daemon=True).start()

    def _show_gf(self, r, nt):
        self.gf_btn.configure(state=tk.NORMAL); self._sts("Goldfish complete!")
        t = self.gf_text; t.configure(state=tk.NORMAL); t.delete("1.0", tk.END)
        if not r: t.insert(tk.END, "Deck too small."); t.configure(state=tk.DISABLED); return
        ta = r["turn_avgs"]; ld = r["land_drops"]
        bd = r.get("bonus_draws", {}); tc = r.get("total_cards", {})
        draw_sources = r.get("draw_sources", [])

        t.insert(tk.END, "GOLDFISH TURN-BY-TURN AVERAGES\n", "header")
        t.insert(tk.END, "(Avg cumulative cards seen by each turn)\n\n")
        cats = [c for c in ALL_CATEGORIES if any(ta[x].get(c,0) > 0.01 for x in ta)]
        hdr = f"{'Turn':>5s}"
        for cat in cats: hdr += f"  {cat[:8]:>8s}"
        hdr += f"  {'Lands':>8s}"
        t.insert(tk.END, hdr + "\n", "info"); t.insert(tk.END, "-"*len(hdr) + "\n")
        for turn in range(nt+1):
            lb = "Open" if turn==0 else str(turn); row = f"{lb:>5s}"
            for cat in cats: row += f"  {ta[turn].get(cat,0):8.2f}"
            row += f"  {ld[turn]:8.2f}"; t.insert(tk.END, row + "\n")

        t.insert(tk.END, "\nLAND DROP PROGRESSION\n", "header")
        t.insert(tk.END, "(Avg lands playable by each turn)\n\n")
        for turn in range(nt+1):
            lb = "Open" if turn==0 else f"T{turn}"; d = ld[turn]
            t.insert(tk.END, f"  {lb:>5s}: {d:5.2f} lands  {'#'*int(d*3)}\n", "good")

        # New: Card Draw Analysis
        t.insert(tk.END, "\nCARD DRAW ANALYSIS\n", "header")
        t.insert(tk.END, "(Natural draws + bonus draws from draw spells/engines)\n\n")
        hdr2 = f"{'Turn':>5s}  {'Natural':>8s}  {'Bonus':>8s}  {'Total':>8s}  {'Cards Seen':>10s}"
        t.insert(tk.END, hdr2 + "\n", "info")
        t.insert(tk.END, "-" * len(hdr2) + "\n")
        for turn in range(nt+1):
            natural = 7 if turn == 0 else 7 + turn  # opening 7 + 1 per turn
            bonus = bd.get(turn, 0)
            total_drawn = natural + bonus
            total_seen = tc.get(turn, 0)
            lb = "Open" if turn == 0 else str(turn)
            tag = "good" if bonus >= 2 else "info"
            t.insert(tk.END, f"  {lb:>5s}  {natural:>8d}  {bonus:>8.1f}  {total_drawn:>8.1f}  {total_seen:>10.1f}\n", tag)

        # Final turn summary
        final_bonus = bd.get(nt, 0)
        final_total = tc.get(nt, 0)
        t.insert(tk.END, f"\n  By turn {nt}: avg {final_bonus:.1f} bonus cards drawn from effects\n", "good")
        t.insert(tk.END, f"  Total cards seen (hand + battlefield): {final_total:.1f}\n", "info")

        # Draw sources list
        if draw_sources:
            t.insert(tk.END, "\nDRAW SOURCES IN DECK\n", "header")
            one_shots = [d for d in draw_sources if not d["repeating"]]
            engines = [d for d in draw_sources if d["repeating"]]
            if engines:
                t.insert(tk.END, "\n  Draw Engines (repeating each turn):\n", "info")
                for d in sorted(engines, key=lambda x: x["draws"], reverse=True):
                    t.insert(tk.END, f"    {d['name']} (x{d['qty']}) — ~{d['draws']}/turn\n", "good")
            if one_shots:
                t.insert(tk.END, "\n  One-Shot Draw (ETB/cantrip):\n", "info")
                for d in sorted(one_shots, key=lambda x: x["draws"], reverse=True):
                    net = d["draws"]
                    t.insert(tk.END, f"    {d['name']} (x{d['qty']}) — draws {net}\n")

            total_one = sum(d["draws"] * d["qty"] for d in one_shots)
            total_eng = sum(d["draws"] * d["qty"] for d in engines)
            t.insert(tk.END, f"\n  Summary: {len(engines)} draw engines (~{total_eng} draws/turn),  "
                             f"{len(one_shots)} one-shots ({total_one} total bonus draws)\n", "info")

        t.configure(state=tk.DISABLED)


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    root = tk.Tk()
    MTGGoldfishApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()