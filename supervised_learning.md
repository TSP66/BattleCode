Supervised learning (just after every single gate we will run a pass of this, so it learns this before going onto the next round of RL), ideally freshly generated perfect play examples each time. We won't train aggressively, maybe just 32 batches of 32 each with lr=0.001 and KL against unmodified bot (new anchor) of 0.02 maybe? Also don't run this until after the first round of RL, as otherwise we risk collapsing the policy.

The idea is their is a small subset of moves that is perfect play which we should help PPO converge to.

We should be able to generate many hundreds of instances of each of these, using the alowed augementations mentioned.

Aggressively augemented situations should help the bot note overfit to specific maps. 

The following situations constitute perfect play:

- Blank board: go straight forwards so long as no pearls/other dragons visible in immediate view point (augement any memory features, kelp in view port, at any point in game)
- Any move by non-queen that kills visible queen (literally any allowed sprint that kills opponent queen at any point in game) you will need to check that all allowed sprints within the 7x7 view port are avaliable in the action space, we should be able to generate many thousands of these
- Late game suicide, if only OUR queen is visible and no opponent is visible (other dragons of our own kind are fine) and our queen is within 2 squares, commit suicide if it is after round 480. (so only give examples for this with round >480).
- Trapped our queen, it is possible to imagine a situation where we see that we've accidentially trapped our own queen, in this case we should promptly commit suicide.
- If there are other instances of 'perfect play' you can imagine please let me know, I will want to assess these myself.