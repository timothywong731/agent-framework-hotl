"""Reflection demo: one agent, a tool-less judge, no workflow graph.

The A/B foil to ``reflexion_demo``. Same corpus, same topic, same worker tool
*set*, same tool budget and the same budget wording in the prompt; the
variable changed on purpose is the critic's access to the corpus - both
critics see the report. Two worker-side residues remain, from the two demos
being separate packages rather than from the experiment: no delivery retry
here, and one conceded exploratory call on the final pass where reflexion's
construction-time strip concedes none. Both are named in the README's A/B
recipe rather than glossed.
"""
