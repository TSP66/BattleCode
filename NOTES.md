Couple observations: 

There are a bunch of extra maps. We need to download them and start playing battles on them.

However before we do that we need to distill the best teams off it, to form strong competitors and baseline on these maps.

The PPO run over the weekend should produce a (very strong) baseline on the rest of the maps. I suggest we start from the strongest out of new distillations and previous bots, and retrain the critic from scratch so it understands the more macro heavy maps.

I believe pearl distribution also changed over weekend (more entropy), (also meaning critic should be retrained).

New critic should be retrained using the updates over the weekend, mostly on games over the weekend.

We need to study the two new 'macro heavy' maps and see if we can make some more like this, the organizers have emphasised that we are likely to see similar maps in the elimination rounds. We should drop glut. Reduce share of current maps to 50% I think.

Whenever we remove a oppodent due to beating it >80% this increase should largely be allocated towards more self play.

We need to check that the action space includes splitting: parent->2 and child going to total-2. I've noticed maps often have dead ends where this is (likely) the optimial approach. 

We need to check we are using all the action space. No point predicting anything we don't use. 

We need to allow stupid operations like suicide. As sometimes it is optimial