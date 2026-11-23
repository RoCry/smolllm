#!/bin/bash
# Derive expected results from the raw SSE with jq, independent of any smolllm code.
set -eu
for f in *.sse; do
n=${f%.sse}
sed -n 's/^data: //p' "$f" | grep -v '^\[DONE\]$' | jq -s '
  [.[].choices[]? | .delta // {}] as $d
  | ([.[].usage? | select(. != null)] | last) as $u
  | {
    content: ([$d[] | .content? // empty] | join("")),
    reasoning: ([$d[] | .reasoning_content? // empty] | join("")),
    tool_calls: ([$d[] | .tool_calls[]?] | group_by(.index) | map({
      id: (map(.id // empty | select(. != "")) | first),
      name: (map(.function.name // empty | select(. != "")) | first),
      arguments: (map(.function.arguments // "") | join(""))
    })),
    reasoning_items: [$d[] | .reasoning_items[]?],
    finish_reason: ([.[].choices[]? | .finish_reason | select(. != null)] | last),
    usage: {
      prompt_tokens: $u.prompt_tokens,
      completion_tokens: $u.completion_tokens,
      cached_tokens: ($u.prompt_tokens_details.cached_tokens // 0),
      reasoning_tokens: ($u.completion_tokens_details.reasoning_tokens // 0)
    }
  }' > "$n.expected.json"
done
