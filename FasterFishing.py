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
Usage:         python FasterFishing.py
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
        try:
            r = requests.get(f"https://api2.moxfield.com/v2/decks/all/{m.group(1)}",
                headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                         "Content-Type":"application/json",
                         "Accept":"application/json"}, timeout=15)
            if r.status_code == 200:
                d = r.json(); lines = []
                # Mark commanders with a comment so parse_text can detect them
                for n, e in d.get("commanders",{}).items():
                    lines.append(f"{e.get('quantity',1)} {n} #CMDR")
                for n, e in d.get("mainboard",{}).items():
                    lines.append(f"{e.get('quantity',1)} {n}")
                return "\n".join(lines)
            elif r.status_code == 403:
                return "__MOXFIELD_BLOCKED__"
        except Exception: pass
        return None

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
        # Step 1: Lands from type_line (careful with MDFCs!)
        for c in cards:
            tl = c.type_line.lower()
            if " // " in tl:
                # Multi-face: only "Land" if ALL faces are land-type
                faces = [f.strip() for f in tl.split(" // ")]
                all_land = all("land" in f for f in faces)
                any_land = any("land" in f for f in faces)
                if all_land:
                    c.category = "Land"
                elif any_land and c.layout == "modal_dfc":
                    # MDFC spell//land — treat as the spell side's category (will be set below)
                    c.category = ""
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
    def sim_goldfish(cards, n=1000, turns=10, pcb=None):
        deck = SimEngine.build_deck(cards)
        if len(deck) < 7 + turns: return {}
        td = {t:{c:0 for c in ALL_CATEGORIES} for t in range(turns+1)}
        ld = {t:0 for t in range(turns+1)}
        for i in range(n):
            sh = random.sample(deck, len(deck))
            seen = list(sh[:7]); lib = sh[7:]
            cs = Counter(c.category for c in seen)
            for cat in ALL_CATEGORIES: td[0][cat] += cs.get(cat,0)
            ld[0] += min(cs.get("Land",0), 1)
            for t in range(1, turns+1):
                if lib: seen.append(lib.pop(0))
                cs = Counter(c.category for c in seen)
                for cat in ALL_CATEGORIES: td[t][cat] += cs.get(cat,0)
                ld[t] += min(cs.get("Land",0), t+1)
            if pcb and i % 100 == 0: pcb(i/n)
        for t in td:
            for cat in td[t]: td[t][cat] /= n
            ld[t] /= n
        if pcb: pcb(1.0)
        return {"turn_avgs": td, "land_drops": ld}

# ============================================================================
# SAMPLE HAND POP-UP
# ============================================================================
class SampleHandWindow:
    def __init__(self, parent, cards, scryfall, pil_cache):
        self.cards = cards; self.scry = scryfall; self.pc = pil_cache; self.refs = []
        self.win = tk.Toplevel(parent); self.win.title("Sample Opening Hand")
        self.win.geometry("1100x520"); self.win.configure(bg="#1a1a2e")
        bf = tk.Frame(self.win, bg="#1a1a2e"); bf.pack(fill=tk.X, padx=10, pady=5)
        for txt, sz in [("Draw New Hand (7)",7),("Mulligan (6)",6),("Mulligan (5)",5)]:
            tk.Button(bf, text=txt, font=("Segoe UI",10,"bold"),
                command=lambda s=sz: self._draw(s), bg="#0f3460", fg="white").pack(side=tk.LEFT, padx=5)
        self.info = tk.StringVar(value="")
        tk.Label(bf, textvariable=self.info, bg="#1a1a2e", fg="#e0e0e0",
                 font=("Segoe UI",10)).pack(side=tk.RIGHT, padx=10)
        self.hf = tk.Frame(self.win, bg="#1a1a2e")
        self.hf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._draw()

    def _draw(self, sz=7):
        for w in self.hf.winfo_children(): w.destroy()
        self.refs.clear()
        deck = SimEngine.build_deck(self.cards)
        if len(deck) < sz: self.info.set("Deck too small!"); return
        hand = random.sample(deck, sz)
        cats = Counter(c.category for c in hand)
        l, r, d = cats.get("Land",0), cats.get("Ramp",0), cats.get("Draw",0)
        kp = "KEEP" if 2<=l<=5 else "MULLIGAN?"
        self.info.set(f"{sz}-card hand  |  Lands: {l}  Ramp: {r}  Draw: {d}  |  {kp}")
        for card in hand:
            cf = tk.Frame(self.hf, bg="#1a1a2e"); cf.pack(side=tk.LEFT, padx=3, fill=tk.Y)
            if card.name in self.pc:
                # PIL image available — create hand-sized PhotoImage
                hand_img = self.pc[card.name].resize((130, 181), Image.LANCZOS)
                photo = ImageTk.PhotoImage(hand_img)
                lbl = tk.Label(cf, image=photo, bg="#1a1a2e"); lbl.pack()
                self.refs.append(photo)
            else:
                col = CATEGORY_COLORS.get(card.category, "#708090")
                ph = tk.Frame(cf, bg=col, width=130, height=181); ph.pack_propagate(False); ph.pack()
                tk.Label(ph, text=card.name, wraplength=120, bg=col, fg="white",
                         font=("Segoe UI",8,"bold")).pack(expand=True)
                # Try to load image async
                threading.Thread(target=self._load, args=(card, cf, ph), daemon=True).start()
            cc = CATEGORY_COLORS.get(card.category, "#708090")
            tk.Label(cf, text=card.category, bg=cc, fg="white",
                     font=("Segoe UI",8,"bold"), padx=4, pady=1).pack(fill=tk.X)

    def _load(self, card, frame, placeholder):
        """Download image and swap the placeholder in the UI."""
        img = self.scry.fetch_image(card.image_uri, card.name)
        if img:
            self.pc[card.name] = img  # cache raw PIL image
            hand_img = img.resize((130, 181), Image.LANCZOS)
            photo = ImageTk.PhotoImage(hand_img)
            def update_ui():
                try:
                    if placeholder.winfo_exists():
                        placeholder.destroy()
                        lbl = tk.Label(frame, image=photo, bg="#1a1a2e")
                        children = frame.winfo_children()
                        if children:
                            lbl.pack(before=children[-1])
                        else:
                            lbl.pack()
                        self.refs.append(photo)
                except tk.TclError:
                    pass  # window was closed
            self.win.after(0, update_ui)


# ============================================================================
# MAIN GUI
# ============================================================================
class FasterFishing:
    def __init__(self, root):
        self.root = root; self.root.title("FasterFishing")
        self.root.geometry("1400x900"); self.root.minsize(1100, 650)
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

        # Add Card controls
        add_frame = ttk.LabelFrame(right, text="Add Card (Scryfall Search)", padding=10)
        add_frame.pack(fill=tk.X, padx=5, pady=(0,5))
        # Search row
        search_row = ttk.Frame(add_frame); search_row.pack(fill=tk.X, pady=(0,3))
        ttk.Label(search_row, text="Search:").pack(side=tk.LEFT)
        self.add_search_var = tk.StringVar()
        self.add_search_entry = ttk.Entry(search_row, textvariable=self.add_search_var, font=("Segoe UI",10))
        self.add_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.add_search_entry.bind("<KeyRelease>", self._on_add_search_key)
        self.add_search_entry.bind("<Return>", lambda e: self._add_card())
        # Autocomplete listbox
        self.add_listbox = tk.Listbox(add_frame, font=("Segoe UI",9), bg="#16213e", fg="#e0e0e0",
                                       selectbackground="#0f3460", height=5, exportselection=False)
        self.add_listbox.pack(fill=tk.X, pady=(0,3))
        self.add_listbox.bind("<<ListboxSelect>>", self._on_add_listbox_select)
        self.add_listbox.bind("<Double-Button-1>", lambda e: self._add_card())
        # Qty + Add button row
        qty_row = ttk.Frame(add_frame); qty_row.pack(fill=tk.X)
        ttk.Label(qty_row, text="Qty:").pack(side=tk.LEFT)
        self.add_qty_var = tk.StringVar(value="1")
        ttk.Spinbox(qty_row, from_=1, to=99, width=4, textvariable=self.add_qty_var,
                     font=("Segoe UI",10)).pack(side=tk.LEFT, padx=5)
        ttk.Button(qty_row, text="Add to Deck", command=self._add_card).pack(side=tk.LEFT, padx=5)
        self.add_status = tk.StringVar(value="")
        ttk.Label(qty_row, textvariable=self.add_status, foreground="#2ECC71").pack(side=tk.LEFT, padx=5)
        # Debounce tracking
        self._add_search_after_id = None

        cf = ttk.LabelFrame(right, text="Override Category", padding=10); cf.pack(fill=tk.X, padx=5, pady=5)
        self.cat_var = tk.StringVar()
        for i, cat in enumerate(ALL_CATEGORIES):
            col = CATEGORY_COLORS[cat]
            rb = tk.Radiobutton(cf, text=cat, variable=self.cat_var, value=cat, command=self._set_cat,
                bg=self.bg, fg=col, selectcolor=self.bg, activebackground=self.bg, activeforeground=col,
                font=("Segoe UI",10,"bold"), indicatoron=True)
            rb.grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)
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
                "Moxfield blocks automated requests via Cloudflare.\n\n"
                "To import your deck:\n"
                "1. Open your deck on Moxfield\n"
                "2. Click the '...' menu (top right)\n"
                "3. Select 'Export' > 'Export for MTGO'\n"
                "4. Copy the text and paste it here\n\n"
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

    # ---- ADD CARD (Scryfall Search) ----
    def _on_add_search_key(self, event):
        """Debounced autocomplete trigger on keypress."""
        if self._add_search_after_id:
            self.root.after_cancel(self._add_search_after_id)
        self._add_search_after_id = self.root.after(300, self._do_autocomplete)

    def _do_autocomplete(self):
        """Fetch autocomplete suggestions from Scryfall in background."""
        query = self.add_search_var.get().strip()
        if len(query) < 2:
            self.add_listbox.delete(0, tk.END)
            return
        def go():
            suggestions = self.scry.autocomplete(query)
            self.root.after(0, lambda: self._show_autocomplete(suggestions))
        threading.Thread(target=go, daemon=True).start()

    def _show_autocomplete(self, suggestions):
        """Populate the autocomplete listbox."""
        self.add_listbox.delete(0, tk.END)
        for name in suggestions[:10]:
            self.add_listbox.insert(tk.END, name)

    def _on_add_listbox_select(self, event):
        """When user clicks a suggestion, fill the search entry."""
        sel = self.add_listbox.curselection()
        if sel:
            name = self.add_listbox.get(sel[0])
            self.add_search_var.set(name)

    def _add_card(self):
        """Look up card on Scryfall and add it to the deck."""
        name = self.add_search_var.get().strip()
        if not name:
            self.add_status.set("Type a card name first")
            return
        try:
            qty = int(self.add_qty_var.get())
        except ValueError:
            qty = 1
        qty = max(1, qty)

        # Check if already in deck
        for c in self.cards:
            if c.name.lower() == name.lower() or (
                " // " in c.name and name.lower() in c.name.lower()):
                c.quantity += qty
                self._refresh_tree(); self._refresh_summary()
                self.add_status.set(f"+{qty} {c.name} (now x{c.quantity})")
                self.add_search_var.set("")
                self.add_listbox.delete(0, tk.END)
                return

        self.add_status.set("Looking up...")
        def go():
            sc = self.scry.fetch_by_name(name)
            self.root.after(0, lambda: self._add_card_done(sc, name, qty))
        threading.Thread(target=go, daemon=True).start()

    def _add_card_done(self, sc, search_name, qty):
        """Process the Scryfall result and add card to deck."""
        if not sc:
            self.add_status.set(f"Not found: {search_name}")
            return

        full_name = sc.get("name", search_name)
        layout = sc.get("layout", "normal")
        has_faces = "card_faces" in sc and sc["card_faces"]

        # Check duplicate again with full Scryfall name
        for c in self.cards:
            if c.name.lower() == full_name.lower():
                c.quantity += qty
                self._refresh_tree(); self._refresh_summary()
                self.add_status.set(f"+{qty} {c.name} (now x{c.quantity})")
                self.add_search_var.set("")
                self.add_listbox.delete(0, tk.END)
                return

        # Build Card object (same logic as _imp_txt)
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
            if all("land" in f for f in faces):
                card.category = "Land"
            elif "land" in tl and layout != "modal_dfc":
                card.category = "Land"
        elif "land" in tl and "creature" not in tl:
            card.category = "Land"

        if card.category == "Other":
            # Quick single-card tag check
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
        self.add_status.set(f"Added: {full_name} x{qty}")
        self.add_search_var.set("")
        self.add_listbox.delete(0, tk.END)
        self._sts(f"Added {full_name} x{qty} to deck")

        # Preload image in background
        if full_name not in self.pil_cache:
            def load_img():
                pil_img = self.scry.fetch_image(img, full_name)
                if pil_img:
                    self.pil_cache[full_name] = pil_img
            threading.Thread(target=load_img, daemon=True).start()

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

    def _show_hand(self):
        if not self.cards: messagebox.showwarning("No Deck", "Import first!"); return
        deck_cards = self._get_deck_cards()
        if not deck_cards: messagebox.showwarning("Empty Deck", "All cards are set as commander!"); return
        SampleHandWindow(self.root, deck_cards, self.scry, self.pil_cache)

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
        t.configure(state=tk.DISABLED)


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    root = tk.Tk()
    FasterFishing(root)
    root.mainloop()

if __name__ == "__main__":
    main()
