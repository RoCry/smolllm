
Format: `provider/model_name`

## custom keys
the lib will try find keys from `{PROVIDER.upper()}_API_KEY` in env

e.g. `gemini/gemini-2.0-flash` -> `GEMINI_API_KEY`
the keys supports multiple keys separated by comma

## custom base url
`{PROVIDER.upper()}_BASE_URL` same as multiple keys supports

## combine with multiple keys and base url
they can be combined together, e.g.
1. One key and multiple base url
```
OLLAMA_API_KEY=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1,http://rocry-ubuntu.local:11434/v1
```

2. Multiple keys and one base url
```
GEMINI_API_KEY=1,2
GEMINI_BASE_URL=https://api.gemini.com/v1
```

3. Multiple keys and multiple base url
> In this case, the count of keys and base url must be the same, so the lib will choose a random pair of key and base url
> e.g. in below case, the lib will choose `1` and `https://api.gemini.com/v1` or `2` and `https://api.gemini.com/v2`
```
GEMINI_API_KEY=1,2
GEMINI_BASE_URL=https://api.gemini.com/v1,https://api.gemini.com/v2
```


