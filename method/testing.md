# The testing method

How a change to a model server was measured and decided on the Volta builds. It is written to be reused: nothing in
it is specific to the V100 except the numbers in the examples. This is the long form of
[part 5 of the front page](../index.md#part-5-the-whole-process-of-testing).

The method exists because of one fact about this work: a change that makes a large language model server faster
can also make it quietly wrong, and the people (and coding agents) making the change want it to have worked. Every
rule below removes one way of fooling yourself. Most were added after the failure they prevent.

## 1. Three labels, and one word that is never a pass

Every number carries one of three labels.

- **MEASURED**: observed on the hardware, with the date, the power limit, the clock ceiling and what was running.
- **ESTIMATE**: arithmetic or a model built on measurements. The basis is stated.
- **UPSTREAM**: someone else's published figure that we have not reproduced. The source is linked.

An estimate is never written as a result, and a plan is never written as done. A configuration that has not passed
its own rule says **"quality not established"** next to every speed it reports.

A check has three outcomes: PASS, FAIL and **INCONCLUSIVE**. "Could not check" and "found nothing wrong" are
different results. INCONCLUSIVE is never a pass; it is the start of the next test, and the report names the missing
input.

## 2. The unit is the round, not tokens per second

With a speculative drafter, throughput is `(1 + accepted drafts per round) / round time`. The first factor depends
on the text: code accepts three or four of five drafts, prose one. Tokens per second therefore moves with the
prompt set, and a change can look like a win because the test text changed.

The engine's own speed is the time of one verify round, decode only (first to last streamed token, divided by the
rounds). Every decode change is decided on milliseconds per round. Prompt-reading changes are decided on time to
first token at fixed prompt lengths.
Source: [Volta sPLIt, docs/method.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/method.md).

## 3. A rule is written before its data

Before a test runs, a file states: the arms, how each is measured, the figure that decides, the noise floor and how
it is computed, the outcome that means ACCEPT, the outcome that means "no measurable change", and what makes the
test INCONCLUSIVE. The file has a time stamp earlier than the first measurement, and it is not changed afterwards.

- If one replication will be allowed, the rule says so in advance. In one published case a kernel showed -4.12 ms
  per round against a noise floor of 3.43 and was "no measurable change" by its rule; the single replication,
  declared beforehand, showed -4.94 ms against a floor of 1.14 and was accepted. Both windows are published
  ([the rule](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/acceptance/WINDOW11-accept-rule.md),
  [the replication](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/docs/acceptance/WINDOW11-replication.md)).
- When a rule turns out to ask the wrong question, it stays FAILED as written and a new rule is written for the
  next test. In one case a rule asked that a restored chat answer token for token like a cold read of the same
  text. The engine does not give that (see section 8). The line stayed failed; the finding was recorded.
- **Predictions are written down too, and scored.** Before each test the expected result is recorded with its
  basis. A prediction that was wrong is recorded as wrong. On the twelve-card build two memory predictions and one
  timing model were wrong and stand in the plan as wrong; the corrected model was used for the next test.

For a quality comparison the same idea is a **preregistration**: the items, the scoring code and the verdict rules
are fixed and hashed before any output is captured
([example](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/blob/main/quality/closeness-test/PREREGISTRATION.txt)).
Anything counted after scoring is labelled as outside the preregistered rules.

## 4. Mirrored arms, a measured noise floor

- Each arm is a fresh boot of the server, with untimed warm-up requests first. (The first request after a boot
  compiles kernels; first runs of a boot carried 2 to 3 ms per round of noise on one build.)
- Arms run in mirrored order: OFF, ON, ON, OFF. Small changes run eight arms. Drift then cancels.
- Prompts are paired across arms when the generated texts are identical; paired noise was about 0.05 ms per round.
- **The noise floor is measured, not assumed.** The same configuration is measured twice (A against A, and a
  repeat inside one boot). A gain counts only if the pairs agree in sign and the mean difference clears the floor.
- Nothing else runs on the machine during an arm: no compiler, no file copy.
- Know the test's power. With a spread of about 0.12 ms per comparison, a true 0.15 ms gain passes a 0.15 ms rule
  about 40 % of the time, and a 0.25 ms gain about 90 %. A small change that "loses" was not detected; that is not
  proof that it does nothing.
- **Judge a change inside one boot or one window.** The same engine bytes gave different greedy answers on two
  boots of a test machine. Speeds from different days, power limits or boots are not a controlled pair, and the
  tables say so.

## 5. The fenced window on a machine that also serves

The eight- and twelve-card servers were in daily use. A test on them is a **window**:

1. The serving build is stopped. The window's script checks that the cards are empty.
2. **The reference arm boots in the same window.** A speed or an output is compared with a reference measured
   minutes earlier on the same cards at the same power limit, never with last week's number.
3. Test arms boot one at a time, one change per boot. Each must log its switch as active on every worker; an OFF
   arm must log none.
4. Each arm is judged by the rule. A boot that fails is not booted twice in the same window.
5. The serving build is restored, and the restore is itself checked.

On the twelve-card build the last step became a two-phase rule: every check (no expert left in host memory, the
memory limit active on all workers, seven long prompts token for token as the accepted capture, a memory rehearsal
with one grown chat per slot) runs on a **test port** and writes a record of exactly which code and configuration
passed. The serving boot on the real port sends no request and starts only if the record matches what is installed.
A client can never reach a server that is still being judged, and a client's request can never fail a check.

After a card error (an Xid line) or an unplanned reset, the rule is: stop nothing, start nothing, send nothing,
report, wait.

The DGX Spark builds use the same idea; its harness is published as
[dsv41-window-harness](https://github.com/ProprietaryLegal/dsv41-window-harness).

## 6. Switch off means unchanged

Every new code path sits behind a switch that defaults off. Before any speed test:

- a CPU test and a comparison of the syntax tree show that with the switch off the old code runs unchanged;
- a kernel is checked on one card and on four against the kernel it replaces, over every shape it will see, into
  output buffers pre-filled with NaN;
- `compute-sanitizer` runs its memory, race and synchronisation checks.

Only then is a window spent on it.

## 7. Two judges, and what each can show

A change either keeps the same arithmetic or it does not. The two cases need different judges.

**The identity judge.** For a change that claims to compute the same thing: the same prompts, greedy decoding,
compared token id by token id with a reference from the same window.

- It can show that a change does not move answers on those prompts.
- It cannot show that answers are *good*, and it cannot be used across a change of rounding.
- It must be strict. See section 9: ours was not, for a while.

**The near-tie gate.** For a change that reorders floating-point sums (a fused kernel, a different split): outputs
may differ in their last bits. The gate, fixed in advance, accepts a changed token only where the model's top two
candidates were within a small logit margin, and reports those separately. A confident token that moves is a FAIL.
Several of the fastest kernels in these builds failed here and were not installed
([list](../optimizations/what-did-not-work.md#2-it-was-faster-and-changed-the-answers)).

**The closeness judge.** For a change that is meant to change the arithmetic (another quantization, another chunk
cut): there is no identity to check, so the build is compared with a reference build on many short contexts.

- Measures: how often it picks the same next token as the reference; the gap in log-probability on the
  reference's token; KL divergence over the top 20 tokens (a lower bound of the true value); how long a greedy
  continuation stays identical; how many multiple-choice answers change.
- **The reference is run twice.** Its own repeat is the noise floor (100 % identical in the published test).
- A third build is the yardstick, so the answer is "closer than X", not a bare number.
- It can show closeness to the reference. It cannot show closeness to the unquantized model if the reference is
  itself a quantization, and it says nothing about sampling above temperature 0, thinking, or long answers.
- With 513 questions an accuracy difference under about 4 points cannot be resolved; "not distinguishable at this
  sample size" does not mean equal.

Published example with its limits stated:
[Volta Max uPLIft, "Accuracy"](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4#accuracy-how-close-to-nvfp4).

## 8. What "identical" cannot mean on this engine

Three findings limit every token-exact test. All three hold on the unmodified engine.

1. **A cache hit is not a cold read.** At near-tied tokens (top-two logit gaps of 0.125 to 0.47 in the cases seen)
   a greedy reply can differ between a cold read of a prompt and a read that resumes from the prefix cache, and
   between hits at different positions. A token-exact test of cache reuse compares a hit with a hit at the same
   position in the same boot.
2. **A different chunk cut rounds differently.** Large sums across the four cards of a board go through NCCL in
   half precision, in an order that depends on the buffer's size; and the dense multiply library picks its split
   count from a chunk's row count. A 64-row shift of the cut moved long-prompt answers as far as halving the chunk.
3. **Two boots can differ.** Before one fix, sampled output varied by boot because an all-reduce's order did.

So the acceptance test for "the same answers" is: same prompts, same cache state, same chunk cut, same boot or the
same window's reference. Anything else is a closeness question.

## 9. What went wrong with the method itself

These are the most useful part of this page.

- **The test driver's own checks spoiled test arms five times on one build.** An out-of-memory pattern matched the
  memory limit's own announcement line, refused a good boot and cost a second one. A worker count was doubled by
  an extra log line from a new change, and the window stopped at its first boot. A witness read each worker's
  environment from the process table, which a renamed process blanks. A probe replayed answers without their
  reasoning, so its turns missed the cache and it measured cold reads. A witness failed a boot on the API server's
  warning about an unregistered switch. **Rule since: every pattern a check applies to a server's log is first run
  against a real log of that kind of boot, and a new change's new log lines go through every pattern before its
  first window.**
- **CPU tests that did not go through the real entry point.** Two engine changes passed their CPU tests and failed
  on their first boot, because the tests called the new function directly and the server reached it through a
  loader and a scheduler that the tests skipped. Rule since: a table lists which entry point reaches each changed
  file, and a test goes through it.
- **The equality judge accepted cut-off answers.** It compared two answers only through the shorter one. An
  independent review removed one token from each of twelve real references and the judge still printed "identical
  12 of 12". A second judge could pass with keys it could not decide. Both were made strict: equal length, equal
  set of keys, a missing or unstable key is INCONCLUSIVE, and token ids are compared, not text digests. The
  historical results were re-checked on whole sequences.
- **A cache returned wrong tokens after an edit.** A token cache under construction returned wrong token ids on a
  hit when an edit removed the special token at its cut. Its check compared token *counts*, which were equal. It
  was repaired and re-proven on complete sequences under edits by two independent fuzz tests before it was allowed
  near a server.
- **A rule scaled from the wrong data.** A long-prompt limit was taken from short prompts and closed a good
  configuration for one window.
- **A sum is not a rate.** "Four requests together: 306.6 tok/s" was the sum of the four requests' own rates. By
  wall clock the same run was 242.6 tok/s. Both are now stated.
- **A profile overstated a cause.** A time budget attributed all of the host's allocation time per prompt chunk
  to idle cards (16 to 19 % of a read). Intersecting the intervals showed about half of it overlapped a working
  card. The prediction was cut to about 9 % before the test; the test measured +9.9 %.
- **A checker failed a correct run on a rounding artifact** (a probability mass that summed to 1.0000001). The
  checker was fixed and the saved run re-judged.

## 10. The independent review

After the twelve-card build was serving, a second model was given the whole record, read-only, with one question:
what did we measure wrong, and what are we missing? Its findings are the equality judge, the token cache, the
overstated cause and the sum-versus-rate items above, plus a correction to the order of work. The owner of the
build re-ran or re-read each finding before acting on it.

Two things made the review work. It was read-only, so it could not "fix" what it found. And it was handed the code
and the receipts, not the authors' conclusions.

## 11. A checklist

Before the test:

- [ ] The rule is written, with its noise floor and its three outcomes, and has a time stamp.
- [ ] The prediction is written with its basis.
- [ ] The switch defaults off, and switch-off is proven unchanged.
- [ ] Every log pattern the driver uses has been run on a real log of this kind of boot.
- [ ] The change's test goes through the server's real entry point.
- [ ] Every prompt the test sends is public text.

During:

- [ ] The reference boots in the same window, at the same power limit.
- [ ] One change per boot. The switch is logged active on every worker.
- [ ] Nothing else runs on the machine.

After:

- [ ] The verdict is read from the rule, not from the result.
- [ ] The prediction is scored.
- [ ] The number is published with its label, its conditions, and the losing arms.
- [ ] "Quality not established" stays on the row until a judge has run.
