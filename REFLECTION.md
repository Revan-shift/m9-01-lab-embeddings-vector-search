# Reflection

Results below are from an actual run of `search_by_meaning.py` using
the `gemini-embedding-001` model.

## Query 1: "my laptop won't switch on"
- Top match: **kb-02** (score = 0.6638) - "To power up a device that
  won't turn on, hold the power button for ten seconds..."
- Shared words with the query: none of the content words overlap
  ("laptop" vs "device", "switch on" vs "turn on", "won't" is the
  only shared word and it's a function word, not a content word).
- What this shows: the embedding matched on *meaning* (a device that
  fails to power on), not on shared vocabulary. The model treated
  "laptop" as a kind of "device" and "switch on" as equivalent to
  "turn on" purely from the meaning of the words, not their spelling.

## Query 2: "how do I stop being billed every month?"
- Top match: **kb-05** (score = 0.6915) - "To cancel your
  subscription, open Account Settings and choose End Plan..."
- Shared words with the query: none ("billed every month" vs "cancel
  your subscription" / "billing period" share no exact words).
- What this shows: the model recognized that "stop being billed" and
  "cancel subscription" describe the same real-world action, even
  though "billed" and "billing" appear nowhere together and the
  phrasing is completely different.

## Query 3: "access denied error when saving a file"
- Top match: **kb-08** (score = 0.7584) - "The error code 0x80070005
  means 'access denied'..."
- Shared words with the query: "access denied" appears in both, so
  this is the one case with real word overlap. Note it also has the
  highest score of all four queries (0.7584) - a useful contrast
  showing that overlap can boost the score further, but isn't
  required for a strong match (see queries 1, 2, and 4).

## Query 4: "where do I leave my car in the evening?"
- Top match: **kb-01** (score = 0.7061) - "Employees may park in lot
  B after 6pm on weekdays..."
- Shared words with the query: none ("leave my car" vs "park",
  "evening" vs "after 6pm" - completely different vocabulary).
- What this shows: the embedding captured the concept of a parking
  location and a time-of-day constraint, despite zero vocabulary
  overlap between query and passage.

## Pattern across all four queries
In 3 out of 4 queries (1, 2, 4) the best match shares **no** content
words with the query, yet the correct passage still ranks #1 by a
clear margin. This is the core point of the lab: cosine similarity
between embeddings retrieves by *meaning*, not by keyword overlap.
It's also worth noting that **kb-09** (laptop cloud backups) shows up
as a runner-up in three different queries (1, 3, 4) purely because it
mentions "laptops" and is generally workplace/IT-flavored - a reminder
that embedding similarity is about overall semantic closeness, not a
single matching concept.

## Optional stretch: "what's the wifi password?"
- Top match: **kb-01** (score = 0.5903) - "Employees may park in lot
  B after 6pm..." (the parking policy, completely unrelated to wifi)
- Comparison: 0.5903 is lower than every in-scope top-1 score (0.66,
  0.69, 0.76, 0.71), but the gap is smaller than expected - it's not
  dramatically low, just noticeably below the others.
- Takeaway: a similarity threshold could still be useful here. For
  example, a rule like "if the top score is below ~0.62, treat it as
  'no confident answer in the knowledge base'" would correctly flag
  this wifi query while still accepting all four real queries above.
  In a production system you'd tune this threshold on a labeled set
  of in-scope vs out-of-scope queries rather than picking it by eye,
  since a single example isn't enough to set a reliable cutoff.
