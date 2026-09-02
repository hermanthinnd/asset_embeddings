# API and document setup

## OpenAI API key

Do not paste the key into chat, a notebook, `.env.example`, source code, or a committed file.

For a temporary terminal session on macOS:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_EXPOSURE_MODEL="model-id-frozen-for-the-pilot"
```

Start VS Code/Jupyter from that terminal, or restart the notebook kernel after setting the variables. The notebook checks only whether the variables are present; it never prints their values.

For a persistent local setup, add the exports to a private shell configuration file such as `~/.zshrc`, then restart VS Code. Do not place the key anywhere inside the project folder.

The production extractor will use the OpenAI Responses API, strict JSON-schema output, `store=False`, a frozen model ID and prompt hash, and local request/response auditing. API data-retention settings should be reviewed before sending licensed transcripts.

Official documentation:

- https://platform.openai.com/docs/quickstart/make-your-first-api-request
- https://developers.openai.com/api/reference/cli/resources/beta/subresources/responses
- https://developers.openai.com/api/docs/guides/your-data

## Small transcript test

Place a small set of legally obtained transcript text files under:

```text
geopolitical_exposure/documents/raw/earnings_calls/
```

Use one UTF-8 `.txt` file per transcript. Do not rename away the source identifiers. Provide a separate local metadata file with these fields:

```text
document_id
firm_id
company_name
ticker
document_date
available_date
source
source_uri
filename
```

`firm_id` can initially be CIK, FactSet entity ID, or another stable source ID. We will link it to PERMCO in a separate point-in-time crosswalk.

For the Russian-invasion pilot, the first useful test is 10–20 firms across several channels—not only obvious exposed companies. Include some likely controls so extraction precision can be evaluated.

## SEC-filing substitute

If transcripts are unavailable, the same adapter can ingest pre-event 10-K and 10-Q sections. SEC filings are a suitable proof-of-concept corpus because they provide public filing timestamps and evidence text. They are not assumed to be equivalent to earnings calls; `document_type` remains part of every record so the two sources can later be compared.
