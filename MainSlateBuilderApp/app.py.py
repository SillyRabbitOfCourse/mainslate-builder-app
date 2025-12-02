import random
from typing import List, Dict, Tuple
import pandas as pd
import streamlit as st

# ================================================================
#                  ORIGINAL GLOBAL-LIKE CONFIG (DEFAULTS)
#   (These will be overwritten by the UI before building)
# ================================================================

NUM_LINEUPS = 40
SALARY_CAP = 50000
MIN_SALARY = 49000
RANDOM_SEED = 42

SLOT_ORDER = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
FLEX_ELIGIBLE = {"RB", "WR", "TE"}

STACK_TEAMS: List[str] = []
STACK_EXPOSURES: Dict[str, float] = {}
STACK_REQUIRED: Dict[str, List[str]] = {}
STACK_OPTIONAL: Dict[str, Dict[str, float]] = {}
STACK_MIN_MAX: Dict[str, Tuple[int, int]] = {}

STACK_RUNBACK_TEAMS: Dict[str, str] = {}
STACK_RUNBACKS: Dict[str, Dict[str, float]] = {}
STACK_RUNBACK_MIN_MAX: Dict[str, Tuple[int, int]] = {}

STACK_INCLUDE_DST: Dict[str, bool] = {}
STACK_DST_PERCENT: Dict[str, float] = {}

MINI_STACKS: List[Dict] = []

EXCLUDED_PLAYERS: List[str] = []
EXCLUDED_TEAMS: List[str] = []
EXCEPT_PLAYERS_FROM_EXCLUDED_TEAMS: List[str] = []
EXCEPT_DST_FROM_EXCLUDED_TEAMS: List[str] = []

MAX_ATTEMPTS_PER_LINEUP = 5000
MAX_OVERALL_ATTEMPTS = 40 * 100  # will be updated from NUM_LINEUPS


# ================================================================
#                        DATA LOADING
# ================================================================

def load_player_pool(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    df["Salary"] = df["Salary"].astype(int)
    return df


def apply_exclusions(df: pd.DataFrame, excluded_players: List[str]) -> pd.DataFrame:
    if not excluded_players:
        return df
    return df[~df["Name"].isin(excluded_players)].reset_index(drop=True)


def apply_team_exclusions(
    df: pd.DataFrame,
    excluded_teams: List[str],
    except_players: List[str],
    except_dst_teams: List[str],
) -> pd.DataFrame:
    if not excluded_teams:
        return df

    remove_mask = df["TeamAbbrev"].isin(excluded_teams)

    if except_players:
        keep_players = df["Name"].isin(except_players)
        remove_mask = remove_mask & (~keep_players)

    if except_dst_teams:
        is_dst = df["Position"] == "DST"
        keep_dst = df["TeamAbbrev"].isin(except_dst_teams)
        remove_mask = remove_mask & ~(is_dst & keep_dst)

    return df[~remove_mask].reset_index(drop=True)


def position_split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    groups = {}
    for pos, group in df.groupby("Position"):
        groups[pos] = group.reset_index(drop=True)
    return groups


def salary_of_lineup(entries: List[Dict]) -> int:
    return sum(e["Player"]["Salary"] for e in entries)


def lineup_to_key(entries: List[Dict]) -> Tuple:
    ids = sorted(e["Player"]["ID"] for e in entries)
    return tuple(ids)


# ================================================================
#                    STACK + RUNBACK HELPERS
# ================================================================

def sample_optional_players(team: str) -> List[str]:
    chosen = []
    for player_name, pct in STACK_OPTIONAL.get(team, {}).items():
        if random.random() < pct:
            chosen.append(player_name)
    return chosen


def sample_runbacks(team: str) -> List[str]:
    chosen = []
    for player_name, pct in STACK_RUNBACKS.get(team, {}).items():
        if random.random() < pct:
            chosen.append(player_name)
    return chosen


def maybe_add_dst_to_stack(team: str, df: pd.DataFrame):
    if not STACK_INCLUDE_DST.get(team, False):
        return None

    pct = STACK_DST_PERCENT.get(team, 0.0)
    if random.random() >= pct:
        return None

    dst = df[(df["Position"] == "DST") & (df["TeamAbbrev"] == team)]
    if dst.empty:
        return None

    return dst.iloc[0]


# ================================================================
#                    MINI STACK HELPERS
# ================================================================

def init_mini_stack_state(num_lineups: int, mini_stacks: List[Dict]) -> List[Dict]:
    rules = []
    for rule in mini_stacks:
        target = int(num_lineups * rule.get("exposure", 0.0))
        r = dict(rule)
        r["remaining"] = target
        rules.append(r)
    return rules


def mini_rule_applicable_to_team(
    rule: Dict,
    primary_team: str,
    stack_teams: List[str],
    stack_runback_teams: Dict[str, str],
) -> bool:
    stack_teams_set = set(stack_teams)
    runback_teams_set = set(stack_runback_teams.values())

    if rule["type"] == "same_team":
        team = rule["team"]
        if team in stack_teams_set or team in runback_teams_set:
            return False
        return True

    if rule["type"] == "opposing_teams":
        t1 = rule["team1"]
        t2 = rule["team2"]
        if (
            t1 in stack_teams_set or t2 in stack_teams_set or
            t1 in runback_teams_set or t2 in runback_teams_set
        ):
            return False
        return True

    return False


def pick_mini_stack_players(
    rule: Dict,
    df: pd.DataFrame,
    used_ids: set,
    stack_teams: List[str],
    stack_runback_teams: Dict[str, str],
) -> List[pd.Series] | None:
    stack_teams_set = set(stack_teams)
    runback_teams_set = set(stack_runback_teams.values())

    base_pool = df[~df["TeamAbbrev"].isin(stack_teams_set.union(runback_teams_set))]
    base_pool = base_pool[~base_pool["ID"].isin(used_ids)]

    if rule["type"] == "same_team":
        team = rule["team"]
        team_pool = base_pool[base_pool["TeamAbbrev"] == team]
        if team_pool.empty:
            return None

        pair = random.choice(rule["pairs"])
        pos1, pos2 = pair

        p1_pool = team_pool[team_pool["Position"] == pos1]
        if p1_pool.empty:
            return None
        p1 = p1_pool.sample(n=1).iloc[0]

        remaining = team_pool[team_pool["ID"] != p1["ID"]]
        p2_pool = remaining[remaining["Position"] == pos2]
        if p2_pool.empty:
            return None
        p2 = p2_pool.sample(n=1).iloc[0]

        return [p1, p2]

    if rule["type"] == "opposing_teams":
        t1 = rule["team1"]
        t2 = rule["team2"]

        pair = random.choice(rule["pairs"])
        pos1, pos2 = pair

        pool1 = base_pool[
            (base_pool["TeamAbbrev"] == t1) &
            (base_pool["Position"] == pos1)
        ]
        pool2 = base_pool[
            (base_pool["TeamAbbrev"] == t2) &
            (base_pool["Position"] == pos2)
        ]

        if pool1.empty or pool2.empty:
            return None

        p1 = pool1.sample(n=1).iloc[0]
        p2 = pool2.sample(n=1).iloc[0]
        return [p1, p2]

    return None


# ================================================================
#                      STACK LINEUP BUILDER
# ================================================================

def build_stack_lineup(
    df: pd.DataFrame,
    pos_groups: Dict[str, pd.DataFrame],
    team: str,
    mini_rule: Dict | None,
) -> List[Dict] | None:
    """
    Build a single lineup for a given primary stack team.
    Uses global config dicts populated via the UI.
    """

    required = STACK_REQUIRED.get(team, [])
    min_p, max_p = STACK_MIN_MAX.get(team, (2, 5))

    team_players: List[pd.Series] = []

    # Required players
    for name in required:
        row = df[(df["Name"] == name) & (df["TeamAbbrev"] == team)]
        if row.empty:
            return None
        team_players.append(row.iloc[0])

    # Optional players
    optional_names = sample_optional_players(team)
    for name in optional_names:
        row = df[(df["Name"] == name) & (df["TeamAbbrev"] == team)]
        if not row.empty:
            team_players.append(row.iloc[0])

    # ---------------- RUN-BACKS ----------------
    runback_min, runback_max = STACK_RUNBACK_MIN_MAX.get(team, (0, 999))
    opp_team = STACK_RUNBACK_TEAMS.get(team)

    chosen_runbacks: List[pd.Series] = []
    runback_pool: List[pd.Series] = []

    if opp_team is not None and opp_team != "":
        for name in STACK_RUNBACKS.get(team, {}):
            row = df[(df["Name"] == name) & (df["TeamAbbrev"] == opp_team)]
            if not row.empty:
                runback_pool.append(row.iloc[0])

        runback_names = sample_runbacks(team)
        for name in runback_names:
            row = df[(df["Name"] == name) & (df["TeamAbbrev"] == opp_team)]
            if not row.empty:
                chosen_runbacks.append(row.iloc[0])

        if len(chosen_runbacks) < runback_min:
            needed = runback_min - len(chosen_runbacks)
            chosen_ids = {p["ID"] for p in chosen_runbacks}
            available = [p for p in runback_pool if p["ID"] not in chosen_ids]

            if len(available) < needed:
                return None

            extra = random.sample(available, needed)
            chosen_runbacks.extend(extra)

        if len(chosen_runbacks) > runback_max:
            chosen_runbacks = random.sample(chosen_runbacks, runback_max)

        for p in chosen_runbacks:
            team_players.append(p)
    else:
        if runback_min > 0:
            return None

    # Maybe add DST from stack team
    dst_row = maybe_add_dst_to_stack(team, df)
    if dst_row is not None:
        team_players.append(dst_row)

    # ---------------- ATTEMPT TO FILL ROSTER ----------------
    for _ in range(MAX_ATTEMPTS_PER_LINEUP):
        entries: List[Dict] = []
        used_ids = {p["ID"] for p in team_players}

        # Mini-stack correlation
        corr_players: List[pd.Series] = []
        if mini_rule is not None:
            picked = pick_mini_stack_players(
                mini_rule,
                df,
                used_ids,
                STACK_TEAMS,
                STACK_RUNBACK_TEAMS,
            )
            if picked is None:
                continue
            for p in picked:
                corr_players.append(p)
                used_ids.add(p["ID"])

        combo_players = team_players + corr_players

        stack_qbs = [p for p in combo_players if p["Position"] == "QB"]
        stack_rbs = [p for p in combo_players if p["Position"] == "RB"]
        stack_wrs = [p for p in combo_players if p["Position"] == "WR"]
        stack_tes = [p for p in combo_players if p["Position"] == "TE"]
        stack_dsts = [p for p in combo_players if p["Position"] == "DST"]

        count_primary_team = sum(1 for p in team_players if p["TeamAbbrev"] == team)
        if not (min_p <= count_primary_team <= max_p):
            continue

        stack_teams_set = set(STACK_TEAMS)
        runback_teams_set = set(STACK_RUNBACK_TEAMS.values())

        filler_df = df[~df["TeamAbbrev"].isin(stack_teams_set.union(runback_teams_set))]

        def pool(pos: str) -> pd.DataFrame:
            _df = filler_df[filler_df["Position"] == pos]
            return _df[~_df["ID"].isin(used_ids)].reset_index(drop=True)

        # QB
        if stack_qbs:
            qb_row = stack_qbs[0]
        else:
            p = pool("QB")
            if p.empty:
                continue
            qb_row = p.sample(n=1).iloc[0]
        entries.append({"Slot": "QB", "Player": qb_row})
        used_ids.add(qb_row["ID"])

        # RB1, RB2
        rbs = stack_rbs.copy()
        while len(rbs) < 2:
            p = pool("RB")
            if p.empty:
                break
            row = p.sample(n=1).iloc[0]
            rbs.append(row)
            used_ids.add(row["ID"])
        if len(rbs) < 2:
            continue
        entries.append({"Slot": "RB1", "Player": rbs[0]})
        entries.append({"Slot": "RB2", "Player": rbs[1]})

        # WR1, WR2, WR3
        wrs = stack_wrs.copy()
        while len(wrs) < 3:
            p = pool("WR")
            if p.empty:
                break
            row = p.sample(n=1).iloc[0]
            wrs.append(row)
            used_ids.add(row["ID"])
        if len(wrs) < 3:
            continue
        entries.append({"Slot": "WR1", "Player": wrs[0]})
        entries.append({"Slot": "WR2", "Player": wrs[1]})
        entries.append({"Slot": "WR3", "Player": wrs[2]})

        # TE
        if stack_tes:
            te_row = stack_tes[0]
        else:
            p = pool("TE")
            if p.empty:
                continue
            te_row = p.sample(n=1).iloc[0]
        entries.append({"Slot": "TE", "Player": te_row})
        used_ids.add(te_row["ID"])

        # DST
        if stack_dsts:
            dst_row = stack_dsts[0]
        else:
            p = pool("DST")
            if p.empty:
                continue
            dst_row = p.sample(n=1).iloc[0]
        entries.append({"Slot": "DST", "Player": dst_row})
        used_ids.add(dst_row["ID"])

        # FLEX
        flex_pool = filler_df[
            (filler_df["Position"].isin(FLEX_ELIGIBLE)) &
            (~filler_df["ID"].isin(used_ids))
        ]
        if flex_pool.empty:
            continue
        flex_row = flex_pool.sample(n=1).iloc[0]
        entries.append({"Slot": "FLEX", "Player": flex_row})
        used_ids.add(flex_row["ID"])

        total_salary = salary_of_lineup(entries)
        if MIN_SALARY <= total_salary <= SALARY_CAP:
            return entries

    return None


# ================================================================
#                    OUTPUT HELPERS (NO CONSOLE)
# ================================================================

def lineups_to_df(lineups: List[List[Dict]]) -> pd.DataFrame:
    records = []
    for i, lineup in enumerate(lineups, start=1):
        slot_map = {e["Slot"]: e["Player"] for e in lineup}
        rec = {"LineupID": i}
        tot = 0
        for slot in SLOT_ORDER:
            p = slot_map[slot]
            rec[slot] = p["Name + ID"]
            tot += p["Salary"]
        rec["Total_Salary"] = tot
        records.append(rec)
    return pd.DataFrame.from_records(records)


# ================================================================
#                         STREAMLIT APP
# ================================================================

def run_app():
    global NUM_LINEUPS, SALARY_CAP, MIN_SALARY, RANDOM_SEED
    global STACK_TEAMS, STACK_EXPOSURES, STACK_REQUIRED, STACK_OPTIONAL, STACK_MIN_MAX
    global STACK_RUNBACK_TEAMS, STACK_RUNBACKS, STACK_RUNBACK_MIN_MAX
    global STACK_INCLUDE_DST, STACK_DST_PERCENT
    global MINI_STACKS
    global EXCLUDED_PLAYERS, EXCLUDED_TEAMS
    global EXCEPT_PLAYERS_FROM_EXCLUDED_TEAMS, EXCEPT_DST_FROM_EXCLUDED_TEAMS
    global MAX_OVERALL_ATTEMPTS

    st.title("🏈 Main Slate Stacking Lineup Builder")

    uploaded = st.file_uploader("Upload DKSalaries.csv", type=["csv"])
    if not uploaded:
        st.info("Please upload a `DKSalaries.csv` file.")
        return

    # Sidebar global settings
    st.sidebar.header("Global Build Settings")
    num_lineups = st.sidebar.number_input("Number of Lineups", 1, 150, NUM_LINEUPS)
    salary_cap = st.sidebar.number_input("Salary Cap", 0, 50000, SALARY_CAP, 500)
    min_salary = st.sidebar.number_input("Minimum Salary", 0, salary_cap, MIN_SALARY, 500)
    rand_seed = st.sidebar.number_input("Random Seed (-1 for None)", value=RANDOM_SEED)

    # Load DF (raw)
    df_raw = load_player_pool(uploaded)
    all_names = sorted(df_raw["Name"].unique().tolist())
    all_teams = sorted(df_raw["TeamAbbrev"].unique().tolist())

    # Top-level tabs
    tab_global, tab_stacks, tab_runbacks, tab_minis, tab_excl, tab_build = st.tabs(
        ["Global Filters", "Stack Teams", "Run-backs", "Mini-stacks", "Exclusions", "Build Lineups"]
    )

    # ------------- GLOBAL FILTERS TAB -------------
    with tab_global:
        st.subheader("Global Player Pool Filter")
        global_removed = st.multiselect(
            "Remove players from the global player pool (optional):",
            options=all_names,
            default=[],
            help="Players selected here are removed entirely before building lineups.",
        )
        st.caption(
            "Optional: Remove any players you do not want included in the slate. "
            "Removed players will not appear in any lineups or rules."
        )

    # DF after global removed players (used for config UIs)
    df_cfg = df_raw[~df_raw["Name"].isin(global_removed)].reset_index(drop=True)
    cfg_names = sorted(df_cfg["Name"].unique().tolist())
    cfg_teams = sorted(df_cfg["TeamAbbrev"].unique().tolist())

    # ------------- STACK TEAMS TAB -------------
    stack_teams_selected: List[str] = []
    stack_exposures_ui: Dict[str, float] = {}
    stack_required_ui: Dict[str, List[str]] = {}
    stack_optional_ui: Dict[str, Dict[str, float]] = {}
    stack_minmax_ui: Dict[str, Tuple[int, int]] = {}

    with tab_stacks:
        st.subheader("Primary Stack Teams")
        stack_teams_selected = st.multiselect(
            "Select Stack Teams (offenses you want to build around):",
            options=cfg_teams,
            default=[],
            help="Choose any teams you want to stack lineups with.",
        )

        st.caption(
            "For each stack team, set how many lineups (exposure), which players are required, "
            "which are optional, and min/max number of players from that team in each stacked lineup."
        )

        for team in stack_teams_selected:
            with st.expander(f"Stack Settings: {team}", expanded=False):
                exp_pct = st.slider(
                    f"{team} Stack Exposure (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    key=f"exp_{team}",
                )
                stack_exposures_ui[team] = exp_pct / 100.0

                col_min, col_max = st.columns(2)
                with col_min:
                    min_players = st.number_input(
                        f"Min players from {team}",
                        min_value=1,
                        max_value=9,
                        value=2,
                        step=1,
                        key=f"min_{team}",
                    )
                with col_max:
                    max_players = st.number_input(
                        f"Max players from {team}",
                        min_value=min_players,
                        max_value=9,
                        value=5,
                        step=1,
                        key=f"max_{team}",
                    )
                stack_minmax_ui[team] = (min_players, max_players)

                team_players = sorted(
                    df_cfg[df_cfg["TeamAbbrev"] == team]["Name"].unique().tolist()
                )

                required_sel = st.multiselect(
                    f"Required players from {team} (optional):",
                    options=team_players,
                    default=[],
                    key=f"req_{team}",
                )
                st.caption(
                    "Optional: These players will appear in every lineup where this team is the primary stack."
                )
                stack_required_ui[team] = required_sel

                optional_sel = st.multiselect(
                    f"Optional sprinkle players from {team} (optional):",
                    options=team_players,
                    default=[],
                    key=f"opt_{team}",
                )
                st.caption(
                    "Optional: These players can be included in some of this team's stack lineups. "
                    "Set a per-player chance below."
                )

                opt_map: Dict[str, float] = {}
                for pname in optional_sel:
                    pct = st.slider(
                        f"{pname} inclusion chance in {team} stacks (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=1.0,
                        key=f"opt_{team}_{pname}",
                    )
                    opt_map[pname] = pct / 100.0
                stack_optional_ui[team] = opt_map

    # ------------- RUN-BACKS & DST TAB -------------
    stack_runback_teams_ui: Dict[str, str] = {}
    stack_runbacks_ui: Dict[str, Dict[str, float]] = {}
    stack_runback_minmax_ui: Dict[str, Tuple[int, int]] = {}
    stack_include_dst_ui: Dict[str, bool] = {}
    stack_dst_percent_ui: Dict[str, float] = {}

    with tab_runbacks:
        st.subheader("Run-backs & DST Settings")

        st.caption(
            "Configure opposing-team run-backs for each stack (correlated bring-backs), "
            "and whether to include the stacked team's DST in some lineups."
        )

        for team in stack_teams_selected:
            with st.expander(f"Run-backs & DST for {team}", expanded=False):

                opp_team = st.selectbox(
                    f"Run-back (opposing) team for {team} (optional):",
                    options=[""] + [t for t in cfg_teams if t != team],
                    index=0,
                    key=f"rb_team_{team}",
                    help="Choose the opposing team whose players can be used as run-backs. Leave blank to disable run-backs for this team.",
                )
                stack_runback_teams_ui[team] = opp_team

                if opp_team:
                    opp_players = sorted(
                        df_cfg[df_cfg["TeamAbbrev"] == opp_team]["Name"].unique().tolist()
                    )
                    rb_sel = st.multiselect(
                        f"Run-back players from {opp_team} (optional):",
                        options=opp_players,
                        default=[],
                        key=f"rb_players_{team}",
                    )
                    st.caption(
                        "Optional: These players are eligible as run-backs in lineups where "
                        f"{team} is the primary stack. Set per-player chances below."
                    )

                    rb_map: Dict[str, float] = {}
                    for pname in rb_sel:
                        pct = st.slider(
                            f"{pname} run-back chance in {team} stacks (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            step=1.0,
                            key=f"rb_{team}_{pname}",
                        )
                        rb_map[pname] = pct / 100.0
                    stack_runbacks_ui[team] = rb_map

                    col_min, col_max = st.columns(2)
                    with col_min:
                        rb_min = st.number_input(
                            f"Min run-backs in {team} stacks",
                            min_value=0,
                            max_value=3,
                            value=0,
                            step=1,
                            key=f"rbmin_{team}",
                        )
                    with col_max:
                        rb_max = st.number_input(
                            f"Max run-backs in {team} stacks",
                            min_value=rb_min,
                            max_value=3,
                            value=1,
                            step=1,
                            key=f"rbmax_{team}",
                        )
                    stack_runback_minmax_ui[team] = (rb_min, rb_max)
                else:
                    stack_runbacks_ui[team] = {}
                    stack_runback_minmax_ui[team] = (0, 999)

                st.markdown("---")
                include_dst = st.checkbox(
                    f"Allow {team} DST in some {team} stacks? (optional)",
                    value=False,
                    key=f"dst_inc_{team}",
                )
                stack_include_dst_ui[team] = include_dst

                dst_pct = st.slider(
                    f"{team} DST chance in {team} stacks (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    key=f"dst_pct_{team}",
                )
                stack_dst_percent_ui[team] = dst_pct / 100.0
                st.caption(
                    "Optional: If enabled, the stacked team's DST may appear in some of its stack lineups. "
                    "Use the slider to control how often."
                )

    # ------------- MINI-STACKS TAB -------------
    with tab_minis:
        st.subheader("Mini-stacks (Secondary Correlations)")

        st.caption(
            "Mini-stacks create additional correlations separate from your primary stacks. "
            "For example RB + DST on the same team, or WR vs WR from opposing teams."
        )

        if "mini_stacks_ui" not in st.session_state:
            st.session_state["mini_stacks_ui"] = []

        mini_list = st.session_state["mini_stacks_ui"]

        col_add1, col_add2 = st.columns(2)
        with col_add1:
            if st.button("➕ Add same-team mini-stack rule"):
                mini_list.append(
                    {
                        "type": "same_team",
                        "exposure_pct": 0.0,
                        "team": "",
                        "pos1": "RB",
                        "pos2": "DST",
                    }
                )
        with col_add2:
            if st.button("➕ Add opposing-teams mini-stack rule"):
                mini_list.append(
                    {
                        "type": "opposing_teams",
                        "exposure_pct": 0.0,
                        "team1": "",
                        "team2": "",
                        "pos1": "WR",
                        "pos2": "WR",
                    }
                )

        positions = ["QB", "RB", "WR", "TE", "DST"]
        remove_indices = []

        for i, rule in enumerate(mini_list):
            label_type = "Same-team" if rule["type"] == "same_team" else "Opposing-teams"
            with st.expander(f"Mini-stack #{i+1} ({label_type})", expanded=False):
                col_top1, col_top2 = st.columns([3, 1])
                with col_top1:
                    exposure_pct = st.slider(
                        "Mini-stack exposure (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(rule.get("exposure_pct", 0.0)),
                        step=1.0,
                        key=f"mini_exp_{i}",
                    )
                with col_top2:
                    if st.button("🗑 Delete rule", key=f"mini_del_{i}"):
                        remove_indices.append(i)

                rule["exposure_pct"] = exposure_pct

                if rule["type"] == "same_team":
                    team_val = st.selectbox(
                        "Team for same-team mini-stack:",
                        options=[""] + cfg_teams,
                        index=([""] + cfg_teams).index(rule.get("team", "")) if rule.get("team", "") in ([""] + cfg_teams) else 0,
                        key=f"mini_team_{i}",
                    )
                    rule["team"] = team_val

                    pos1_val = st.selectbox(
                        "Position 1:",
                        options=positions,
                        index=positions.index(rule.get("pos1", "RB")) if rule.get("pos1", "RB") in positions else 1,
                        key=f"mini_pos1_{i}",
                    )
                    pos2_val = st.selectbox(
                        "Position 2:",
                        options=positions,
                        index=positions.index(rule.get("pos2", "DST")) if rule.get("pos2", "DST") in positions else 4,
                        key=f"mini_pos2_{i}",
                    )
                    rule["pos1"] = pos1_val
                    rule["pos2"] = pos2_val

                elif rule["type"] == "opposing_teams":
                    team1_val = st.selectbox(
                        "Team 1:",
                        options=[""] + cfg_teams,
                        index=([""] + cfg_teams).index(rule.get("team1", "")) if rule.get("team1", "") in ([""] + cfg_teams) else 0,
                        key=f"mini_t1_{i}",
                    )
                    team2_val = st.selectbox(
                        "Team 2:",
                        options=[""] + cfg_teams,
                        index=([""] + cfg_teams).index(rule.get("team2", "")) if rule.get("team2", "") in ([""] + cfg_teams) else 0,
                        key=f"mini_t2_{i}",
                    )
                    rule["team1"] = team1_val
                    rule["team2"] = team2_val

                    pos1_val = st.selectbox(
                        "Position from Team 1:",
                        options=positions,
                        index=positions.index(rule.get("pos1", "WR")) if rule.get("pos1", "WR") in positions else 2,
                        key=f"mini_pos1_opp_{i}",
                    )
                    pos2_val = st.selectbox(
                        "Position from Team 2:",
                        options=positions,
                        index=positions.index(rule.get("pos2", "WR")) if rule.get("pos2", "WR") in positions else 2,
                        key=f"mini_pos2_opp_{i}",
                    )
                    rule["pos1"] = pos1_val
                    rule["pos2"] = pos2_val

                st.caption(
                    "Optional: This rule will be applied to approximately the selected percentage "
                    "of all built lineups, as long as it doesn't conflict with stack isolation."
                )

        # Delete rules marked for removal
        if remove_indices:
            mini_list = [r for idx, r in enumerate(mini_list) if idx not in remove_indices]
            st.session_state["mini_stacks_ui"] = mini_list

    # ------------- EXCLUSIONS TAB -------------
    excluded_players_extra: List[str] = []
    excluded_teams_ui: List[str] = []
    except_players_ui: List[str] = []
    except_dst_teams_ui: List[str] = []

    with tab_excl:
        st.subheader("Player & Team Exclusions")

        excluded_players_extra = st.multiselect(
            "Additional players to exclude (optional):",
            options=cfg_names,
            default=[],
            help="These players will be removed from the pool after global filters.",
        )
        st.caption(
            "Optional: Use this to remove players you do not want in any lineup."
        )

        excluded_teams_ui = st.multiselect(
            "Teams to exclude entirely (optional):",
            options=cfg_teams,
            default=[],
            help="All players from these teams will be removed, except for any whitelisted below.",
        )
        st.caption(
            "Optional: Remove teams you do not want to use in lineups, while optionally whitelisting certain players or DSTs."
        )

        except_players_ui = st.multiselect(
            "Players to keep even if their team is excluded (optional):",
            options=cfg_names,
            default=[],
        )
        st.caption(
            "Optional: These players will remain in the pool even if their team is excluded above."
        )

        except_dst_teams_ui = st.multiselect(
            "DST teams to keep even if excluded (optional):",
            options=cfg_teams,
            default=[],
        )
        st.caption(
            "Optional: DSTs from these teams remain usable even if their team is excluded."
        )

    # ------------- BUILD LINEUPS TAB -------------
    with tab_build:
        st.subheader("Build Lineups")

        st.markdown("**Summary of Config:**")
        st.write(f"- Number of lineups: **{num_lineups}**")
        st.write(f"- Stack teams selected: **{', '.join(stack_teams_selected) if stack_teams_selected else 'None'}**")

        if st.button("🚀 Build Lineups"):
            # Set global config values from UI
            NUM_LINEUPS = int(num_lineups)
            SALARY_CAP = int(salary_cap)
            MIN_SALARY = int(min_salary)
            RANDOM_SEED = int(rand_seed) if rand_seed >= 0 else None
            MAX_OVERALL_ATTEMPTS = NUM_LINEUPS * 100

            if RANDOM_SEED is not None:
                random.seed(RANDOM_SEED)

            if not stack_teams_selected:
                st.error("You must select at least one stack team.")
                return

            if all(stack_exposures_ui.get(t, 0.0) == 0.0 for t in stack_teams_selected):
                st.error("All stack exposures are zero. Set at least one stack team exposure > 0.")
                return

            STACK_TEAMS = list(stack_teams_selected)
            STACK_EXPOSURES = dict(stack_exposures_ui)
            STACK_REQUIRED = dict(stack_required_ui)
            STACK_OPTIONAL = dict(stack_optional_ui)
            STACK_MIN_MAX = dict(stack_minmax_ui)

            STACK_RUNBACK_TEAMS = {
                t: stack_runback_teams_ui.get(t, "") for t in STACK_TEAMS
            }
            STACK_RUNBACKS = dict(stack_runbacks_ui)
            STACK_RUNBACK_MIN_MAX = dict(stack_runback_minmax_ui)

            STACK_INCLUDE_DST = dict(stack_include_dst_ui)
            STACK_DST_PERCENT = dict(stack_dst_percent_ui)

            # Build MINI_STACKS config from UI
            mini_stacks_conf: List[Dict] = []
            for rule in st.session_state.get("mini_stacks_ui", []):
                exp_fraction = rule.get("exposure_pct", 0.0) / 100.0
                if exp_fraction <= 0:
                    continue
                if rule["type"] == "same_team":
                    if not rule.get("team"):
                        continue
                    mini_stacks_conf.append(
                        {
                            "type": "same_team",
                            "exposure": exp_fraction,
                            "team": rule["team"],
                            "pairs": [[rule["pos1"], rule["pos2"]]],
                        }
                    )
                elif rule["type"] == "opposing_teams":
                    if not rule.get("team1") or not rule.get("team2"):
                        continue
                    mini_stacks_conf.append(
                        {
                            "type": "opposing_teams",
                            "exposure": exp_fraction,
                            "team1": rule["team1"],
                            "team2": rule["team2"],
                            "pairs": [[rule["pos1"], rule["pos2"]]],
                        }
                    )
            MINI_STACKS = mini_stacks_conf

            EXCLUDED_PLAYERS = list(set(global_removed) | set(excluded_players_extra))
            EXCLUDED_TEAMS = list(excluded_teams_ui)
            EXCEPT_PLAYERS_FROM_EXCLUDED_TEAMS = list(except_players_ui)
            EXCEPT_DST_FROM_EXCLUDED_TEAMS = list(except_dst_teams_ui)

            # Now build lineups using final df
            df = load_player_pool(uploaded)
            df = apply_exclusions(df, EXCLUDED_PLAYERS)
            df = apply_team_exclusions(
                df,
                EXCLUDED_TEAMS,
                EXCEPT_PLAYERS_FROM_EXCLUDED_TEAMS,
                EXCEPT_DST_FROM_EXCLUDED_TEAMS,
            )

            # Re-derive teams after exclusions (just for safety)
            pos_groups = position_split(df)

            # stack lineup counts per team
            stack_counts = {
                team: int(NUM_LINEUPS * STACK_EXPOSURES.get(team, 0.0))
                for team in STACK_TEAMS
            }
            total_planned = sum(stack_counts.values())

            if total_planned == 0:
                st.error("Total planned stack lineups is 0. Increase some stack exposures.")
                return

            st.write(f"Target stack lineups per team: {stack_counts} (sum = {total_planned})")

            # mini rules state
            mini_rules_state = init_mini_stack_state(NUM_LINEUPS, MINI_STACKS)

            def choose_mini_rule_for_team(stack_team: str) -> Dict | None:
                for r in mini_rules_state:
                    if r["remaining"] <= 0:
                        continue
                    if mini_rule_applicable_to_team(r, stack_team, STACK_TEAMS, STACK_RUNBACK_TEAMS):
                        return r
                return None

            lineups: List[List[Dict]] = []
            seen = set()

            for team in STACK_TEAMS:
                needed = stack_counts.get(team, 0)
                built = 0
                attempts = 0

                while built < needed and attempts < MAX_OVERALL_ATTEMPTS:
                    attempts += 1
                    active_mini = choose_mini_rule_for_team(team)
                    lineup = build_stack_lineup(df, pos_groups, team, active_mini)
                    if lineup is None:
                        continue

                    key = lineup_to_key(lineup)
                    if key in seen:
                        continue

                    lineups.append(lineup)
                    seen.add(key)
                    built += 1

                    if active_mini is not None:
                        active_mini["remaining"] -= 1

                st.write(f"Built {built}/{needed} lineups for stack team {team}.")

            if not lineups:
                st.error("No valid lineups could be built. Try loosening constraints or exposures.")
                return

            df_out = lineups_to_df(lineups)
            st.success(f"Built {len(lineups)} lineups successfully!")
            st.dataframe(df_out)

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Lineups CSV",
                data=csv_bytes,
                file_name="stack_lineups.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    run_app()
