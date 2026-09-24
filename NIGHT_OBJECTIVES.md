You are to work all night tonight as there a couple huge changes that have been made:

Spawn sub agents as required to handle tasks. This is a very large refactor. 
"""
Sonars can send / receive 64 bit messages
A dragon may choose to broadcast a different message for each cardinal direction each turn
A dragon can now hear the echoes of a sonar, indicating what the sonars hit
"""
Read the docs if you are confused.

Your objectives for tonight are: 
- Upgrade all code to be up to standard with the new version
- Upgrade the critic architecture to a large privildge CNN
- Ensure the upgrade worked to improve the architecture of the agents network, previously it was flat packed shit. The memories in particular need good structure.
- There is a huge upgrade with the SONAR, the most important thing is after a split we need to send the child information. We have 64 bits to do this. I think it makes sense to try and send the most meaningful memories to it. Maybe even just four latents from the network? Have a think - it doesn't need to be perfect - but it should make sense. We should stretch those 64 bits as much as possible. On turn zero we can gareentee where it came from - but if it recieves multiple SONARS at once it will need to error check them to work out which came from the parent. Maybe just allocate 8 bits to confirm this hardcoded. 
- Assuming it doesn't cost too many operations we should also send a sonar in every direction every turn, as we now get that information back. This comes in the form of how many hit what - this is useful information. That the model needs.

Phase 2 (not high priority, but could be useful down the line and think about this when making the agent CNN):

- Another use of SONAR that is important - when we can be confident that our message will be recieved (i.e., the heads are inline with each other) - I say we send a message (with some check-sum) that gives useful location information to the agent. These should be encoded in a similar way to memories - which is why we need a super high quality CNN artichture.
- An example of high quality information is some child dragons are spawned as their parent went down a one way street. And just share that this is to be avoided. 


Now we need to do a huge redistillation.
- There are now new teams on top, we need to redistill these with fresh scrapes potentially.
- We also need to confirm the new CNN architecture works for the agents by comparing them like for like (only authoritive way to do this is round robin against other agents we have saved - slow but strong).
- We also need to retrain them with the SONAR upgraded features. You will likely need to implement them yourself, use the SONAR style I've laid out. Ignore the distillation's own use of SONAR. We will always use our own.
- Lastly I want to distill the best agent again, but use an LTSM this time. You will need to ensure the LTSM actaully fits within the compute budget - but if it does it is likely to be the best way to store memories. Keep the LTSM layer structured if possible. I want to know if an LTSM better paramatrizes memories.

Once we have found the best distillation, we need to begin a giant retraining of the PPO process. Note lessons buried in the repo about the best way to do this. Note that previously trained bots are still useful and valid in the new update, just likely suboptimal. 

Obviously the privledged critic will require a full retraining. We've previously had problems with the critic overfitting and than collapsing. Perhaps pretrain on a huge number of games. Conditioning the network on who both the players are (reasonably small learnable embeddings for each team + sinuisoidal movement based on iteration number). It can than evolve during PPO fitting as normal. 

Ideally have the new PPO training underway by tomorrow mid morning. See if you can reach this objective. 
