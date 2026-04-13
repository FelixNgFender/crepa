---
name: read-arxiv-paper
description:
  Use this skill when when asked to read an arxiv paper given an arxiv URL
---

You will be given a ID of an arxiv paper, for example 2601.07372.

### Part 1: Fetch the paper

Use `hf` CLI to fetch it like so
`hf papers read <ID> >> ./knowledge/{arxiv_id}.txt`

### Part 2: Read and report on the paper

Read the contents. Once you've read the paper, produce a summary of the paper
into a markdown file at `./knowledge/summary_{tag}.md`. Generate some reasonable
`tag` like e.g. `conditional_memory` or whatever seems appropriate given the
paper. Make sure that the tag doesn't exist yet so you're not overwriting files.

As for the summary itself, remember that you're processing this paper within the
context of this repository, so most often we we will be interested in how to
apply the paper and its lessons to the corrupted-jepa project. Therefore, you
should feel free to "remind yourself" of the related pea code by reading the
relevant parts, and then explicitly make the connection of how this paper might
relate to us or what are things we might be inspired about or try.
