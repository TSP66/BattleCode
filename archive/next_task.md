There is an agent working on the C++ submission, you are not to interfere with them.

We need some idea of locally how good an agent is. To do this we need to make a couple basic strategies (greedy pearl, some bot that gets in the way, another that aggresively pursues portals, one that never splits, one that always splits, etc). They don't need to be particulary excellent individaully, but ideally their strategies are largely orthogonal. We will than update the dashboard to include winrates against these bots on different maps.

Don't run this eval too often maybe every ~10-15 million turns. As this eval will not be trained on - it is merely an assessment yardstick.

There are a couple more metrics I want to add (assuming they don't have to much cost). 
 - Portal usage (percentage of turns that use a portal)
 - Dragons killed per game


I also want to upweight the smaller boards, as I suspect they will train quicker. Keep the larger boards in play - just make them rarer.

The maps we see are not necessarily the ones we will see in elimination, so we need to make some basic augmentations to the ones we have. With time we will make some more and we will also record any used in battles. 

Just randomly add and delete kelp according to some parameter and also vary pearl spawning and the exact starting poisition. You can also randomly crop a board or add a blank row/column/both to get some variability going. Obviously maintain symmentry rules.

Also make sure entropy penalty is slowly decaying with time.

Additionally how are we preventing knowledge disappearing - or a least checking that its not forgetting anything. In the evaulation mix add a couple past versions so we can be confident it is getting better.

Of the current run - try and retain progress - I'm very curious how it fares against these metrics. 

For the next run - upweight kills a tad - I want to try and see some more aggressive behaviour.