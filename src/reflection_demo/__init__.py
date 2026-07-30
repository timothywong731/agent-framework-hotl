"""Reflection demo: one agent, a tool-less judge, no workflow graph.

The A/B foil to ``reflexion_demo``. Same corpus, same topic, same worker tool
*set*; the variable changed on purpose is the critic's access to the corpus -
both critics see the report. The tool *budget* differs too (the reflexion
worker is capped at 12 calls per turn with mid-turn stripping, this one at
nothing), as does the delivery retry; both are named in the README's A/B
recipe rather than glossed.
"""
