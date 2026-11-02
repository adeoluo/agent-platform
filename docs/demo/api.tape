# Records docs/demo/api.gif from a real run: `vhs docs/demo/api.tape` from the repo root.
Output docs/demo/api.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' AGENT_PLATFORM_DB=$(mktemp -d)/demo.db AGENT_PLATFORM_URL=http://127.0.0.1:18095 && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `export AGENT_PLATFORM_API_KEY=$(agent-platform keys create --tenant acme | awk '/api key/{print $3}')` Enter
Sleep 2s
Type `agent-platform serve --port 18095 >/dev/null 2>&1 &` Enter
Sleep 2s
Type `sleep 4 && agent-platform ingest docs/demo/acme.json >/dev/null && clear` Enter
Sleep 1s
Show
Type `agent-platform ask "How long is the warranty?" | jq -c '{answer, citations, grounded}'` Enter
Sleep 3.5s
Type `agent-platform jobs submit -q "How long is the warranty?" -q "What is 17 * 23?" | jq -c` Enter
Sleep 3s
Hide
Type `kill %1` Enter
Sleep 1s
