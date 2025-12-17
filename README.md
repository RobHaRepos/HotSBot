# replay_parser (POC)

This microservice parses Heroes of the Storm replay files (.StormReplay) using the Blizzard `heroprotocol` CLI tool.  
This folder contains a small Proof-of-Concept (POC) parser implemented in `app/replay_parser/src/parser.py`. The POC wrapper calls the `heroprotocol` CLI and parses the newline-delimited JSON objects that `--json` emits.

## Requirements
- Python 3.7–3.11 recommended (3.11 preferred).
- heroprotocol; `mpyq`, `six`, `protobuf` as dependencies.

Install quick (pip):

```powershell
python -m pip install -r app/replay_parser/requirements.txt
```

To use locally, install the required packages including `heroprotocol`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r app/replay_parser/requirements.txt
```

## CLI usage (heroprotocol)
Basic form:

```powershell
python -m heroprotocol [FLAGS] replay_file.StormReplay
```

Flags (as exposed by `heroprotocol` CLI):
- `--gameevents` — print game events (coordinates included)
- `--messageevents` — print message events (pings, chat messages)
- `--trackerevents` — print tracker events (unit births / deaths / stats)
- `--attributeevents` — print attribute events (attr id and values)
- `--header` — print protocol header (build id, elapsedGameLoops)
- `--details` — print replay details (players, teams, heroes)
- `--initdata` — print initialization data (player settings / cache handles)
- `--stats` — print stats summary for the event stream
- `--json` — output JSON objects (newline-delimited streaming JSON)

Examples:

```powershell
# Print details
python -m heroprotocol --header "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-header.txt
python -m heroprotocol --details "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-details.txt
python -m heroprotocol --initdata "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-initdata.txt
python -m heroprotocol --stats "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-stats.txt
python -m heroprotocol --gameevents "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-gameevents.txt
python -m heroprotocol --messageevents "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-messageevents.txt
python -m heroprotocol --trackerevents "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-trackerevents.txt
python -m heroprotocol --attributeevents "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-attributeevents.txt

# Combined: all flags into one file (useful for full dump)
python -m heroprotocol --header --details --initdata --stats --gameevents --messageevents --trackerevents --attributeevents "data\replays\2025-12-09 17.29.31 Silver City.StormReplay" > output-all.txt

# Stream JSON objects for details and tracker events
python -m heroprotocol --details --trackerevents --json "data/replays/example.StormReplay" > output.json
```

Note: `--json` produces a stream of JSON objects (newline-delimited/sequence of objects), not a single array. When consuming programmatically, parse line-by-line or use `jq -s` to combine into a single JSON array.

## Implemented POC
- `app/replay_parser/src/parser.py` — a small wrapper that runs `python -m heroprotocol --json <flags> <file>` via `subprocess`, then parses the emitted newline-delimited JSON into a list of Python objects. The primary helper is `parse_replay_with_cli(path, flags)`.

## Project structure (where to find things)
- `app/replay_parser/src/parser.py` — local POC CLI wrapper and parsing helpers (invokes `heroprotocol` CLI or library depending on your env).
- `data/replays/` — sample replay files used for local POC testing.
- `tests/` — repo unit tests (runs against parse helpers / POC scripts).

## POC script
- `app/replay_parser/src/parser.py` — the POC wrapper that invokes `heroprotocol` via subprocess and returns parsed JSON objects; call `parse_replay_with_cli(path, flags)` to parse a replay and receive a list of objects.

## Running examples
- Run `heroprotocol` directly:

```powershell
python -m heroprotocol --details --trackerevents --json "data/replays/2025-12-09 17.29.31 Silver City.StormReplay" > output.txt
```

- Run the POC wrapper from Python:

```powershell
python -c "from replay_parser.parser import parse_replay_with_cli; print(parse_replay_with_cli('data/replays/2025-12-09 17.29.31 Silver City.StormReplay', ['--details','--trackerevents','--json']))"
```

## Notes on the `heroprotocol` CLI JSON format
The `--json` output is a stream of newline-delimited JSON objects representing header/details/init/game/tracker events. For programmatic consumption, parse line-by-line and collect objects of interest (e.g. `details`, `header`, or `tracker` events). Example snippet:

```python
import json
with open('output.txt', 'r', encoding='utf-8') as fh:
    for line in fh:
        try:
            obj = json.loads(line)
            # handle obj
        except json.JSONDecodeError:
            pass
```

## Credits & License
- This project integrates Blizzard's `heroprotocol` library: https://github.com/Blizzard/heroprotocol
- Ship responsibly and follow the upstream MIT license.

## Next steps
- Add a FastAPI wrapper to accept replay uploads and return parsed JSON.
- Normalize parsed output into a stable JSON schema for downstream services.

---
Short and ready for copy/paste usage. For additional guidance or to wire up an HTTP microservice, tell me which option you want next (FastAPI wrapper / summary extraction / queue consumer).
