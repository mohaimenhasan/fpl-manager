"""Shared, pre-deadline forecast representation for training and inference."""

import json

SYSTEM_PROMPT = (
    "You are an FPL points forecaster. Use only the supplied pre-deadline information. "
    "Future outcomes are unknown. Return only JSON with a players list containing "
    "player_id and points (your predicted gameweek points) for every requested player."
)
FORECAST_INSTRUCTION = (
    "Forecast each requested player's final individual FPL points for this gameweek. "
    "Points are before captain multipliers or automatic substitutions. Historical "
    "training answers are realized supervised labels, not information available before "
    "the deadline. Return only JSON: "
    '{"players":[{"player_id":123,"points":4.0}]}.'
)


def forecast_input(snapshot):
    """Build inference input exclusively from one archived pre-deadline snapshot."""
    bootstrap = snapshot["bootstrap"]
    players = {p["id"]: p for p in bootstrap["elements"]}
    predictions = {p["player_id"]: p for p in snapshot["predictions"]}
    fields = (
        "web_name", "team", "element_type", "status", "chance_of_playing_next_round",
        "now_cost", "form", "points_per_game", "minutes", "total_points",
        "goals_scored", "assists", "clean_sheets", "expected_goals",
        "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    )
    requested = []
    for player_id in snapshot["squad_ids"]:
        player = players[player_id]
        requested.append({
            "player_id": player_id,
            **{key: player[key] for key in fields if key in player},
            "baseline_xpts": predictions[player_id]["xpts"],
            "price": predictions[player_id]["price"],
        })
    return {
        "season": snapshot["season"],
        "gameweek": snapshot["gameweek"],
        "as_of": snapshot["fetched_at"],
        "deadline_time": snapshot["deadline_time"],
        "players": requested,
        "teams": [
            {key: team[key] for key in ("id", "name", "short_name")}
            for team in bootstrap["teams"]
        ],
        "fixtures": [
            {key: fixture.get(key) for key in (
                "id", "event", "team_h", "team_a", "kickoff_time",
                "team_h_difficulty", "team_a_difficulty",
            )}
            for fixture in snapshot["fixtures"]
            if fixture.get("event") == snapshot["gameweek"]
        ],
    }


def user_prompt(instruction, inputs):
    return instruction + "\n\n" + json.dumps(inputs, ensure_ascii=False, sort_keys=True)
