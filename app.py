
import streamlit as st
import pandas as pd
from io import StringIO
import json

st.set_page_config(page_title="CSV → Crossword", page_icon="🧩", layout="wide")

# -------------------------- Utilities --------------------------

def normalize_entries(rows):
    cleaned = []
    for w, c in rows:
        if not isinstance(w, str) or not isinstance(c, str):
            continue
        w2 = "".join(ch for ch in w.upper() if ch.isalpha())
        c2 = c.strip()
        if len(w2) >= 2 and c2:
            cleaned.append((w2, c2))
    # De-duplicate by word, keep first clue
    seen = set()
    uniq = []
    for w, c in cleaned:
        if w not in seen:
            uniq.append((w, c))
            seen.add(w)
    return uniq

def empty_grid(n):
    return [["#" for _ in range(n)] for _ in range(n)]

def can_place_word(grid, word, row, col, direction):
    n = len(grid)
    dr, dc = (0, 1) if direction == "H" else (1, 0)
    # Bounds
    r_end = row + dr * (len(word)-1)
    c_end = col + dc * (len(word)-1)
    if r_end < 0 or r_end >= n or c_end < 0 or c_end >= n:
        return False

    # Preceding/Following cells cannot be letters (unless out of bounds)
    pr, pc = row - dr, col - dc
    fr, fc = r_end + dr, c_end + dc
    if 0 <= pr < n and 0 <= pc < n and grid[pr][pc] != "#":
        return False
    if 0 <= fr < n and 0 <= fc < n and grid[fr][fc] != "#":
        return False

    for i, ch in enumerate(word):
        r = row + dr * i
        c = col + dc * i
        cell = grid[r][c]
        # Conflict with different letter
        if cell != "#" and cell != ch:
            return False
        # Check orthogonal neighbors for illegal adjacency (unless crossing on same cell)
        if direction == "H":
            # vertical neighbors cannot be letters unless this is a crossing cell
            up = r-1
            down = r+1
            if cell == "#":  # only care when placing into empty cell
                if 0 <= up < n and grid[up][c] != "#":
                    return False
                if 0 <= down < n and grid[down][c] != "#":
                    return False
        else:
            # horizontal neighbors
            left = c-1
            right = c+1
            if cell == "#":
                if 0 <= left < n and grid[r][left] != "#":
                    return False
                if 0 <= right < n and grid[r][right] != "#":
                    return False
    return True

def place_word(grid, word, row, col, direction):
    dr, dc = (0, 1) if direction == "H" else (1, 0)
    for i, ch in enumerate(word):
        r = row + dr * i
        c = col + dc * i
        grid[r][c] = ch

def find_intersections(grid, word):
    """Return list of (row, col, direction) placements that intersect existing letters."""
    n = len(grid)
    candidates = []
    # Build map of letter positions
    letter_positions = {}
    for r in range(n):
        for c in range(n):
            ch = grid[r][c]
            if ch != "#":
                letter_positions.setdefault(ch, []).append((r,c))

    for wi, wch in enumerate(word):
        for pos in letter_positions.get(wch, []):
            r, c = pos
            # Try horizontal placement intersecting at (r,c) with word[wi]
            start_c = c - wi
            if 0 <= start_c < n:
                if can_place_word(grid, word, r, start_c, "H"):
                    candidates.append((r, start_c, "H"))
            # Try vertical
            start_r = r - wi
            if 0 <= start_r < n:
                if can_place_word(grid, word, start_r, c, "V"):
                    candidates.append((start_r, c, "V"))
    return candidates

def generate_crossword(entries, size=15):
    """
    Very simple greedy/intersection-first placer.
    Returns grid, placed_info(list of dict), skipped(list)
    """
    entries = sorted(entries, key=lambda x: -len(x[0]))  # longest first
    grid = empty_grid(size)
    placed = []
    skipped = []

    if not entries:
        return grid, placed, skipped

    # Place the first word in the middle horizontally
    first_word, first_clue = entries[0]
    mid = size // 2
    start_c = max(0, (size - len(first_word)) // 2)
    if can_place_word(grid, first_word, mid, start_c, "H"):
        place_word(grid, first_word, mid, start_c, "H")
        placed.append({"word": first_word, "clue": first_clue, "row": mid, "col": start_c, "dir": "H"})
    else:
        skipped.append((first_word, first_clue))

    # Place the rest
    for word, clue in entries[1:]:
        candidates = find_intersections(grid, word)
        placed_this = False
        for r, c, d in candidates:
            if can_place_word(grid, word, r, c, d):
                place_word(grid, word, r, c, d)
                placed.append({"word": word, "clue": clue, "row": r, "col": c, "dir": d})
                placed_this = True
                break
        if not placed_this:
            # Try scanning the board for any legal stand-alone placement
            n = len(grid)
            for r in range(n):
                if placed_this: break
                for c in range(n):
                    for d in ("H", "V"):
                        if can_place_word(grid, word, r, c, d):
                            place_word(grid, word, r, c, d)
                            placed.append({"word": word, "clue": clue, "row": r, "col": c, "dir": d})
                            placed_this = True
                            break
                    if placed_this: break
            if not placed_this:
                skipped.append((word, clue))

    return grid, placed, skipped

def build_numbering(grid, placed):
    """
    Compute Across and Down numbering and return clue lists with numbers,
    and a map of (r,c)->number for rendering.
    """
    n = len(grid)
    number_map = [[0]*n for _ in range(n)]
    next_num = 1
    across = []
    down = []
    # Find starts
    for r in range(n):
        for c in range(n):
            if grid[r][c] == "#":
                continue
            start_across = (c == 0 or grid[r][c-1] == "#") and (c+1 < n and grid[r][c+1] != "#")
            start_down = (r == 0 or grid[r-1][c] == "#") and (r+1 < n and grid[r+1][c] != "#")
            if start_across or start_down:
                number_map[r][c] = next_num
                next_num += 1

    # Build words and map to clues using placed info
    placed_lookup = {}
    for pi in placed:
        key = (pi["row"], pi["col"], pi["dir"], len(pi["word"]))
        placed_lookup.setdefault(key, []).append(pi)

    # Across
    for r in range(n):
        c = 0
        while c < n:
            if grid[r][c] != "#":
                start = c
                while c < n and grid[r][c] != "#":
                    c += 1
                end = c - 1
                num = number_map[r][start]
                word = "".join(grid[r][x] for x in range(start, end+1))
                clue = None
                maybe = placed_lookup.get((r, start, "H", len(word)), [])
                if maybe:
                    clue = maybe[0]["clue"]
                across.append({"num": num if num else None, "word": word, "row": r, "col": start, "clue": clue})
            c += 1
    # Down
    for c in range(n):
        r = 0
        while r < n:
            if grid[r][c] != "#":
                start = r
                while r < n and grid[r][c] != "#":
                    r += 1
                end = r - 1
                num = number_map[start][c]
                word = "".join(grid[x][c] for x in range(start, end+1))
                clue = None
                maybe = placed_lookup.get((start, c, "V", len(word)), [])
                if maybe:
                    clue = maybe[0]["clue"]
                down.append({"num": num if num else None, "word": word, "row": start, "col": c, "clue": clue})
            r += 1
    return across, down, number_map

def render_grid_html(grid, number_map):
    n = len(grid)
    html = []
    html.append('<div style="overflow:auto; max-height:70vh; border:1px solid #ddd;">')
    html.append('<table style="border-collapse:collapse;">')
    for r in range(n):
        html.append("<tr>")
        for c in range(n):
            ch = grid[r][c]
            if ch == "#":
                html.append('<td style="width:32px;height:32px;background:#111;border:1px solid #999;"></td>')
            else:
                num = number_map[r][c]
                html.append('<td style="position:relative;width:32px;height:32px;border:1px solid #999;text-align:center;vertical-align:middle;font-family:monospace;font-weight:bold;">')
                if num:
                    html.append(f'<div style="position:absolute;top:1px;left:2px;font-size:10px;font-weight:600;">{num}</div>')
                html.append(f'<div style="font-size:16px;line-height:32px;">{ch}</div>')
                html.append('</td>')
        html.append("</tr>")
    html.append("</table></div>")
    return "\n".join(html)

# -------------------------- UI --------------------------

st.title("🧩 CSV → Crossword Generator")

with st.sidebar:
    st.header("Input")
    size = st.slider("Grid size", min_value=7, max_value=25, value=15, step=1, help="Squares per side")
    uploaded = st.file_uploader("Upload CSV (word, clue)", type=["csv"])
    st.caption("Tips: CSV should have two columns: word, clue. Extra columns are ignored.")
    st.write("---")
    st.subheader("Add entries manually")
    new_word = st.text_input("Word (letters only)")
    new_clue = st.text_input("Clue / Definition")
    add_btn = st.button("Add to list")
    st.write("---")
    bulk_txt = st.text_area("Bulk paste (CSV: word,clue per line)", height=120, placeholder="banana, A yellow fruit\npython, A programming language")
    use_bulk = st.checkbox("Include bulk-pasted entries", value=False)

# Session state for manual list
if "manual_entries" not in st.session_state:
    st.session_state.manual_entries = []

if add_btn and new_word and new_clue:
    st.session_state.manual_entries.append((new_word, new_clue))

# Collect rows from CSV
rows = []
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if df.shape[1] >= 2:
            rows.extend(list(zip(df.iloc[:,0].astype(str), df.iloc[:,1].astype(str))))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Add manual entries
rows.extend(st.session_state.manual_entries)

# Add bulk
if use_bulk and bulk_txt.strip():
    sio = StringIO(bulk_txt)
    try:
        df2 = pd.read_csv(sio, header=None)
        if df2.shape[1] >= 2:
            rows.extend(list(zip(df2.iloc[:,0].astype(str), df2.iloc[:,1].astype(str))))
    except Exception:
        for line in bulk_txt.splitlines():
            if "," in line:
                w, c = line.split(",", 1)
                rows.append((w.strip(), c.strip()))

entries = normalize_entries(rows)

colL, colR = st.columns([1,1])

with colL:
    st.subheader("Your Entries")
    if entries:
        df_show = pd.DataFrame(entries, columns=["Word", "Clue"])
        st.dataframe(df_show, hide_index=True, use_container_width=True)
    else:
        st.info("No valid entries yet. Upload a CSV or add some words and clues in the sidebar.")

go = st.button("🔧 Generate Crossword", type="primary", use_container_width=True)

if go and not entries:
    st.warning("Please add at least one valid word & clue.")
    go = False

if go:
    grid, placed, skipped = generate_crossword(entries, size=size)
    across, down, number_map = build_numbering(grid, placed)

    with colR:
        st.subheader("Preview")
        html = render_grid_html(grid, number_map)
        st.markdown(html, unsafe_allow_html=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Across")
        for item in across:
            if item["num"]:
                clue = item["clue"] or "(no clue)"
                st.write(f'**{item["num"]}.** {clue} ({len(item["word"])})')
    with c2:
        st.subheader("Down")
        for item in down:
            if item["num"]:
                clue = item["clue"] or "(no clue)"
                st.write(f'**{item["num"]}.** {clue} ({len(item["word"])})')
    with c3:
        st.subheader("Status")
        st.write(f"Placed: **{len(placed)}**")
        st.write(f"Skipped: **{len(skipped)}**")
        if skipped:
            with st.expander("Show skipped words"):
                for w, c in skipped:
                    st.write(f"- {w}: {c}")

    puzzle = {
        "size": size,
        "grid": grid,
        "across": across,
        "down": down,
        "placed": placed,
        "skipped": skipped,
    }
    st.download_button("📥 Download puzzle JSON", data=json.dumps(puzzle, indent=2),
                       file_name="crossword.json", mime="application/json")

    def blank_repr(grid, number_map):
        lines = []
        n = len(grid)
        for r in range(n):
            row_chars = []
            for c in range(n):
                if grid[r][c] == "#":
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            lines.append("".join(row_chars))
        return "\n".join(lines)

    st.download_button("📄 Download blank grid (.txt)",
                       data=blank_repr(grid, number_map),
                       file_name="crossword_blank.txt",
                       mime="text/plain")

else:
    with colR:
        st.subheader("Preview")
        st.info("Generate the crossword to see a preview here.")

st.caption("Note: This generator uses a simple greedy placer. For big, theme-heavy crosswords, results may vary. Try adjusting grid size or entry list.")
