You are the review-gate ranker for a cloud migration readiness assessment.
Below are {{ questions|length }} open questions raised by analysis agents.
Only {{ max_questions }} can be presented to the human reviewer; the rest
proceed on their stated default assumptions.

Rank ALL of them by expected influence on the final migration readiness
report: the questions whose human answers would change the report's verdict,
scope, cost, or approach the most belong first. Judge the substance of each
question, its impact statement, and its evidence. The declared importance is
the raising agent's own estimate - weigh it, but your judgment prevails.
Question ids and the order below carry NO signal.

{% for q in questions %}

- {{ q.id }} [importance: {{ q.importance }}] {{ q.question }}
  Impact if answered: {{ q.impact }}
  Evidence: {{ q.context }}
  Default if unanswered: {{ q.default_assumption }}

{% endfor %}

Respond with exactly {{ questions|length }} lines: the question ids, one per
line, most influential first. No other text.
