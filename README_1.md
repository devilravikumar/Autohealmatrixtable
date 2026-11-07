# AutoHeal-RAG evaluation: SIMULATION results 

Reproduce: `python3 -I run_experiment.py out 5000 42,1,2,3,4`

What ran: a seeded simulator of 12 REST services (Train-Ticket service names), 8 error classes, 5,000 injected
errors per seed with ground-truth labels (`out/seed_*/injected_errors_ground_truth.jsonl`). 30% of messages are
generic (uninformative), 35% of specific ones carry misleading noise, 15% of faults are transient (self-heal).
Four systems: rule-based regex; TF-IDF top-5 RAG over a separate 640-record KB; agentic RAG (query reformulation);
AutoHeal-RAG (agentic + confidence + self-reflection below 0.85 using runtime metrics + policy gate).
Remediation = simulated: correct action fixes in 3-20 s, wrong action does nothing, health checked at 60 s.

What did NOT run: real Train-Ticket (Docker registry blocked here), any LLM, real remediation on real services.
"Timing" is wall-clock of this Python code (~0.01-0.65 ms), NOT LLM or recovery latency; do not compare to the slide's 11.4 s / 42.7 s.
Results depend on the simulator's difficulty settings (noise rates, metric overlap), which I chose and did not tune to hit any target.
